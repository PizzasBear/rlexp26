import datetime as dt
from argparse import ArgumentParser
from itertools import count
from typing import Any

import ale_py
import gymnasium as gym
import jax
import numpy as np
import numpy.typing as npt
import optax
from etils.epath import Path
from flax import nnx
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.vector import VectorEnv
from orbax.checkpoint import v1 as ocp
from tensorboardX import SummaryWriter

gym.register_envs(ale_py)

from expreplay import ReplayBuffer

from . import btr

SEED = 0
FIRE_ACTION = 1  # ALE action 1 launches the ball in Breakout
ALE_PROTOCOL = dict[str, Any](
    repeat_action_probability=0.25,  # sticky actions (Machado et al. 2018)
    frameskip=4,
    stack_num=btr.FRAME_STACK,
    noop_max=30,
    use_fire_reset=True,
    episodic_life=False,  # True changes the task; report it if used
    reward_clipping=True,  # clip to [-1, 1] for training
    img_height=84,
    img_width=84,
)


def evaluate() -> None:
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="stop after this many episodes; runs until interrupted by default",
    )
    args = parser.parse_args()

    checkpoint_path: Path = args.checkpoint_path
    episodes: int | None = args.episodes

    # Rebuilds ALE_PROTOCOL out of wrappers, since the native vectoriser that applies it in
    # main() cannot render. Two deliberate departures from the protocol: rewards stay unclipped,
    # because eval reports the game score, and sticky actions are off, because deterministic
    # transitions are easier to watch. Both mean these numbers are NOT comparable with the
    # training curve or with Machado-protocol published scores.
    env: gym.Env[npt.NDArray[np.uint8], np.int64] = gym.make(
        "ALE/Breakout-v5",
        repeat_action_probability=0,  # deliberate, see above
        frameskip=1,  # AtariPreprocessing does the skipping and the maxpool
        render_mode="human",
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=ALE_PROTOCOL["frameskip"],
        noop_max=ALE_PROTOCOL["noop_max"],
        screen_size=(ALE_PROTOCOL["img_width"], ALE_PROTOCOL["img_height"]),
    )
    env = gym.wrappers.FrameStackObservation(env, ALE_PROTOCOL["stack_num"])

    assert isinstance(env.action_space, Discrete)
    assert isinstance(env.observation_space, Box)

    num_actions: int = int(env.action_space.n)

    rngs = nnx.Rngs(SEED)
    obs_stack: int = env.observation_space.shape[0]
    qnet = btr.QNet(num_actions, obs_stack=obs_stack, rngs=rngs)
    # The abstract state matters: SpectralNorm keys its batch_stats by tuple, and a
    # structure-free load hands those back stringified, which corrupts the graph.
    nnx.update(qnet, ocp.load(checkpoint_path.absolute(), nnx.state(qnet)))
    qnet.eval()  # type: ignore[no-untyped-call]

    # Unlike main()'s vectoriser, a single env does not autoreset: it keeps reporting terminated
    # until reset() is called, so every episode here starts with an explicit pair of calls. The
    # reset observation is the one before the ball is launched, so it is the FIRE step's
    # observation that the policy is given, not reset()'s.
    def start_episode() -> npt.NDArray[np.uint8]:
        env.reset()
        # ALE_PROTOCOL trains with use_fire_reset=True, which fires on reset and not on a lost
        # life, so the policy has to relaunch the ball itself mid-episode. Match that here.
        obs, _reward, terminated, truncated, _info = env.step(FIRE_ACTION)
        assert not (terminated or truncated), "the episode ended on its opening FIRE"
        return obs

    obs = start_episode()
    returns, played = 0.0, 0
    try:
        while episodes is None or played < episodes:
            action = jax.device_get(btr.act(qnet, obs, rngs=rngs))

            next_obs, reward, terminated, truncated, _info = env.step(action)

            returns += float(reward)
            if terminated or truncated:
                print("EPISODE " + ("TRUNCATED" if truncated else "TERMINATED"))
                print(f"  TOTAL REWARDS = {returns}")
                returns, played = 0.0, played + 1
                obs = start_episode()
            else:
                obs = next_obs
    except KeyboardInterrupt:
        print(f"\nINTERRUPTED after {played} episodes")
    finally:
        env.close()


def main() -> None:
    env_name = "breakout"
    env: VectorEnv[npt.NDArray[np.uint8], npt.NDArray[np.uint64], npt.NDArray[Any]] = (
        gym.make_vec("ALE/Breakout-v5", num_envs=btr.NUM_ENVS, **ALE_PROTOCOL)
    )

    assert isinstance(env.action_space, MultiDiscrete)
    assert isinstance(env.single_observation_space, Box)
    assert env.single_observation_space.dtype is not None

    num_actions: int = env.action_space.nvec[0]
    assert (env.action_space.nvec == num_actions).all()

    rngs = nnx.Rngs(SEED)
    obs_stack: int = env.single_observation_space.shape[0]
    qnet = btr.QNet(num_actions, obs_stack=obs_stack, rngs=rngs)
    target_qnet = nnx.clone(qnet)
    inference_qnet = nnx.clone(qnet)

    opt = nnx.Optimizer(
        qnet,
        optax.chain(
            optax.clip_by_global_norm(btr.GRADIENT_CLIPPING_MAX_NORM),
            optax.adam(btr.LEARNING_RATE, eps=btr.ADAM_EPS),
        ),
        wrt=nnx.Param,
    )

    buf = ReplayBuffer(
        env.num_envs,
        btr.BUFFER_SIZE // env.num_envs,
        obs_stack=obs_stack,
        obs_shape=env.single_observation_space.shape[1:],
        obs_dtype=env.single_observation_space.dtype,
        act_dtype=np.uint8,
        act_shape=(),
        use_prios=True,
        seed=SEED,
    )

    now_str = dt.datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = Path(f"./checkpoints/{env_name}_{now_str}_qnet").absolute()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(f"./logs/{env_name}_{now_str}")

    obs, _info = env.reset()
    buf.reset(obs)

    actions = jax.device_get(btr.act(qnet, obs, rngs=rngs))

    # Performance related TODOs:
    # TODO: Should I separate actor and learner to make train_step independent?
    # TODO: Should I save to the replay buffer in a worker thread to parallelise?
    # TODO: Should I sample batches in a background thread and then queue them for parallel CPU-GPU transfer?
    # TODO: Should I queue priority updates to parallelise GPU-CPU transfer?
    #       (this is the cheapest of the four -- see the overlap entry in btr.py, and note it
    #        also fixes the recycled-slot priority bug flagged at the bottom of this loop)

    # Measured 2026-09-06, 3080 + 24-thread host, 64 envs, batch 256:
    #   pre-train-start: ~1000 env steps/s
    #   steady state:    ~910 env steps/s (~3.6k ALE frames/s), i.e. ~15h for 50M env steps
    #   GPU 80% util, 324W of a 370W cap, 84C -- power/thermally limited, so it is the wall
    #   host CPU ~1.4 cores of 24 (main thread ~40%, 16 ALE threads ~5% each): not the wall

    # General:
    # TODO: Every once in a while run an evaluation run to display progress

    avg_returns = 0
    curr_returns = 0
    for step in count():
        should_train = btr.TRAIN_START_BUF_SIZE < len(buf)

        # TODO: replace this garbage with real tracking
        if step % 25 == 0:
            num_env_steps = env.num_envs * step
            print(f"Num environment steps: {num_env_steps}")
            print(f"Average returns: {avg_returns}")
            if should_train:
                writer.add_scalar(
                    "avg_training_returns", avg_returns, global_step=num_env_steps
                )

        next_obs, rewards, terminated, truncated, _info = env.step(actions)

        curr_returns += rewards[0]
        if terminated[0] or truncated[0]:
            factor = 0.99
            avg_returns = factor * avg_returns + (1 - factor) * curr_returns
            curr_returns = 0

        next_actions = btr.act(inference_qnet, next_obs, rngs=rngs)
        next_actions.copy_to_host_async()

        if step % btr.INFERENCE_SYNC_FREQ == 0:
            btr.sync_qnet(qnet, inference_qnet)

        if should_train:
            (
                batch_indices,
                batch_prios,
                batch_obs,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_obs,
            ) = buf.sample(btr.BATCH_SIZE, n_steps=btr.N_STEP, discount=btr.DISCOUNT)

            new_prios = btr.train_step(
                qnet=qnet,
                target_qnet=target_qnet,
                sample_prios=batch_prios,
                opt=opt,
                obs=batch_obs,
                actions=batch_actions,
                rewards=batch_rewards,
                dones=batch_dones,
                next_obs=batch_next_obs,
                rngs=rngs,
            )
            new_prios.copy_to_host_async()

        buf.save_step(
            actions.astype(np.uint8),
            rewards.astype(np.float32),
            terminated,
            truncated,
            next_obs,
        )

        # TODO: the target update is keyed on the env-loop counter, which only equals the
        #       gradient-step count while the act/train ratio stays 1:1. Count updates instead.
        if should_train and step % btr.TARGET_NETWORK_UPDATE_FREQ == 0:
            btr.sync_qnet(qnet, target_qnet)

            # TODO: check if this is correct, improve per run naming, throw this into a dedicated directory, etc.
            ocp.save_async(checkpoint_path, nnx.state(qnet), overwrite=True)

        obs = next_obs
        actions = jax.device_get(next_actions)

        # TODO: priorities land after this iteration's save_step, so once the buffer has
        #       wrapped a slot sampled above can already have been recycled and gets the
        #       previous occupant's priority. Either update before save_step (costs the
        #       GPU->CPU overlap) or have the buffer reject indices written since the draw.
        if should_train:
            buf.update_prios(batch_indices, jax.device_get(new_prios))
