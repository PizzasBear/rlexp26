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

FIRE_ACTION = 1  # ALE action 1 launches the ball in Breakout


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

    # Rebuilds ale.PROTOCOL out of wrappers, since the native vectoriser main() uses cannot
    # render. Two deliberate departures: rewards stay unclipped (this reports a game score) and
    # sticky actions are off (easier to watch). Both mean these numbers are NOT comparable with
    # the training curve or with Machado-protocol published scores.
    env: gym.Env[npt.NDArray[np.uint8], np.int64] = gym.make(
        ale.ENV_ID,
        repeat_action_probability=0,  # deliberate, see above
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

    obs_stack: int = env.observation_space.shape[0]
    # A policy rather than the agent it was trained by: watching needs the acting weights, and
    # building the rest would allocate a target network and an optimizer for nothing.
    policy: Policy = btr.QNetPolicy(
        num_actions, obs_stack, frames_per_step=ale.PROTOCOL["frameskip"]
    )

    # Plain defaults, unlike main()'s: this reads a directory a training run may still be writing
    # to, and cleanup_tmp_directories or a preservation policy would delete the other process's
    # work. The abstract state matters too -- SpectralNorm keys batch_stats by tuple, and a
    # structure-free load hands those back stringified, corrupting the graph.
    with ocp.training.Checkpointer(run_dir.absolute()) as ckptr:
        if ckptr.latest is None:
            raise SystemExit(f"no checkpoint in {run_dir}")
        policy.load(
            ckptr.load_checkpointables(step, policy.checkpointables()), seed=seed
        )

    # Unlike main()'s vectoriser a single env does not autoreset, so every episode here starts
    # with an explicit pair of calls. The policy is given the FIRE step's observation, not
    # reset()'s, which is the one before the ball is launched.
    def reset_and_fire() -> npt.NDArray[np.uint8]:
        env.reset()
        # Matches training, where use_fire_reset launches the ball at an episode start and
        # never on a lost life -- so the policy has to relaunch it itself mid-episode.
        obs, _reward, terminated, truncated, _info = env.step(FIRE_ACTION)
        assert not (terminated or truncated), "the episode ended on its opening FIRE"
        return obs

    obs = reset_and_fire()
    returns, played = 0.0, 0
    try:
        while episodes is None or played < episodes:
            # The evaluation policy at frame 0, since a checkpoint does not carry the frame
            # count a schedule would be read at. Not a greedy policy: it soft-samples, which is
            # what training and evaluation both do, and the point of watching it.
            action, _ = policy.act(obs, num_env_steps=0, evaluation=True)
            action = jax.device_get(action)

            next_obs, reward, terminated, truncated, _info = env.step(action)

            returns += float(reward)
            if terminated or truncated:
                print("EPISODE " + ("TRUNCATED" if truncated else "TERMINATED"))
                print(f"  TOTAL REWARDS = {returns}")
                returns, played = 0.0, played + 1
                obs = reset_and_fire()
            else:
                obs = next_obs
    finally:
        env.close()
