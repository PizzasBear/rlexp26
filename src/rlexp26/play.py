"""Watch a checkpoint play, rendered."""

from argparse import ArgumentParser

import gymnasium as gym
import jax
import numpy as np
import numpy.typing as npt
from etils.epath import Path
from gymnasium.spaces import Box, Discrete
from orbax.checkpoint import v1 as ocp

from . import ale, btr
from .agent import Policy


def play() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=Path,
        help="a run's checkpoint directory, ./checkpoints/<run>",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="gradient step to load; the newest checkpoint in run_dir by default",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="stop after this many episodes; runs until interrupted by default",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seeds the NoisyNet draws this watches the policy make",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    step: int | None = args.step
    episodes: int | None = args.episodes
    seed: int = args.seed

    # ale.PROTOCOL rebuilt from wrappers, since the native vectoriser cannot render. Rewards
    # are unclipped and sticky actions off, so scores here are not comparable with training's.
    env: gym.Env[npt.NDArray[np.uint8], np.int64] = gym.make(
        ale.ENV_ID,
        repeat_action_probability=0,
        frameskip=1,  # AtariPreprocessing does the skipping and the maxpool
        render_mode="human",
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=ale.PROTOCOL["frameskip"],
        noop_max=ale.PROTOCOL["noop_max"],
        screen_size=(ale.PROTOCOL["img_width"], ale.PROTOCOL["img_height"]),
    )
    env = gym.wrappers.FrameStackObservation(env, ale.PROTOCOL["stack_num"])

    assert isinstance(env.action_space, Discrete)
    assert isinstance(env.observation_space, Box)

    num_actions: int = int(env.action_space.n)

    obs_stack, obs_height, obs_width = env.observation_space.shape
    policy: Policy = btr.QNetPolicy(
        num_actions,
        (obs_stack, obs_height, obs_width),
        frames_per_step=ale.PROTOCOL["frameskip"],
    )

    # Plain defaults: a training run may still be writing here, and main()'s cleanup and
    # preservation policies would delete its work. The abstract state is needed because
    # SpectralNorm keys batch_stats by tuple, which a structure-free load stringifies.
    with ocp.training.Checkpointer(run_dir.absolute()) as ckptr:
        if ckptr.latest is None:
            raise SystemExit(f"no checkpoint in {run_dir}")
        policy.load(
            ckptr.load_checkpointables(step, policy.checkpointables()), seed=seed
        )

    obs, _info = env.reset()
    returns, played = 0.0, 0
    try:
        while episodes is None or played < episodes:
            # At frame 0: a checkpoint does not carry the frame count.
            action, _ = policy.act(obs, num_env_steps=0, evaluation=True)
            action = jax.device_get(action)

            next_obs, reward, terminated, truncated, _info = env.step(action)

            returns += float(reward)
            if terminated or truncated:
                print("EPISODE " + ("TRUNCATED" if truncated else "TERMINATED"))
                print(f"  TOTAL REWARDS = {returns}")
                returns, played = 0.0, played + 1
                obs, _info = env.reset()
            else:
                obs = next_obs
    finally:
        env.close()
