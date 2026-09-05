from itertools import count
from typing import Any

import ale_py
import gymnasium as gym
import jax
import numpy as np
import numpy.typing as npt
import optax
from flax import nnx
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.vector import VectorEnv

gym.register_envs(ale_py)

from expreplay import ReplayBuffer

from . import btr

SEED = 0
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


def main() -> None:
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

    obs, _info = env.reset()
    buf.reset(obs)

    actions = jax.device_get(btr.act(qnet, obs, rngs=rngs))

    # Performance related TODOs:
    # TODO: Should I separate actor and learner to make train_step independent?
    # TODO: Should I save to the replay buffer in a worker thread to parallelise?
    # TODO: Should I sample batches in a background thread and then queue them for parallel CPU-GPU transfer?
    # TODO: Should I queue priority updates to parallelise GPU-CPU transfer?

    # Current pre-train-start CPU util: 60%

    # Current normal CPU util: 10%-15%
    # Current normal GPU util: 70%-80%

    # General:
    # TODO: Every once in a while run an evaluation run to display progress
    # TODO: `env.num_envs * step` counts env steps, not stored transitions -- under
    #       next-step autoreset the post-termination step stores nothing. Print len(buf).
    for step in count():
        if not step % 100:
            print(f"Num transitions: {env.num_envs * step}")
        next_obs, rewards, terminated, truncated, _info = env.step(actions)

        next_actions = btr.act(qnet, next_obs, rngs=rngs)
        next_actions.copy_to_host_async()

        should_train = btr.TRAIN_START_BUF_SIZE < len(buf)

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
            btr.update_target(qnet, target_qnet)

        obs = next_obs
        actions = jax.device_get(next_actions)

        # TODO: priorities land after this iteration's save_step, so once the buffer has
        #       wrapped a slot sampled above can already have been recycled and gets the
        #       previous occupant's priority. Either update before save_step (costs the
        #       GPU->CPU overlap) or have the buffer reject indices written since the draw.
        if should_train:
            buf.update_prios(batch_indices, jax.device_get(new_prios))
