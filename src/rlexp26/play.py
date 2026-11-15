"""Watch a run's checkpoint play, rendered."""

import gymnasium as gym
import jax
import jax.random as jr
import numpy as np
import numpy.typing as npt
from etils import epath
from orbax.checkpoint import v1 as ocp

from .ale import PROTOCOL
from .cli import CHECKPOINT_ROOT, Run
from .recipes import make_agent


def play(
    run: Run, run_name: str, *, step: int | None, episodes: int | None, seed: int
) -> None:
    """
    Play the run ``run_name``'s checkpoint at gradient step ``step``, its newest by default,
    under the policy its evaluation scored at that point, for ``episodes`` episodes or until
    interrupted.
    """
    # PROTOCOL rebuilt from wrappers, since ALE's native vectoriser cannot render.
    env: gym.Env[npt.NDArray[np.uint8], np.int64] = gym.wrappers.FrameStackObservation(
        gym.wrappers.AtariPreprocessing(
            gym.make(
                run.task.env_id,
                repeat_action_probability=PROTOCOL["repeat_action_probability"],
                frameskip=1,
                render_mode="human",
            ),
            frame_skip=PROTOCOL["frameskip"],
            noop_max=PROTOCOL["noop_max"],
            screen_size=(PROTOCOL["img_width"], PROTOCOL["img_height"]),
        ),
        PROTOCOL["stack_num"],
    )
    agent = make_agent(
        run.recipe,
        env.observation_space,
        env.action_space,
        frames_per_step=run.task.frames_per_step,
        seed=run.seed,
    )
    checkpoint_dir = epath.Path(CHECKPOINT_ROOT / run_name).absolute()
    # Plain defaults: a run may still be writing here, and its policies would delete its work.
    with ocp.training.Checkpointer(checkpoint_dir) as reader:
        if reader.latest is None:
            raise SystemExit(f"no checkpoint in {checkpoint_dir}")
        restored = reader.load_checkpointables(
            step, agent.checkpointables() | {"num_env_steps": 0}
        )
    agent.restore(restored)
    num_env_steps = int(restored["num_env_steps"])
    policy = agent.policy(num_env_steps=num_env_steps)
    print(
        f"PLAYING {run_name} at {agent.num_updates} gradient steps, "
        f"{num_env_steps} env steps"
    )

    state = policy.initial_state(1, key=jr.key(seed))
    obs, _info = env.reset(seed=seed)
    episode_start = np.ones(1, bool)
    returns, length, played = 0.0, 0, 0
    try:
        while episodes is None or played < episodes:
            actions, state = policy.act(obs[None], state, episode_start=episode_start)
            obs, reward, terminated, truncated, _info = env.step(
                jax.device_get(actions)[0]
            )
            returns, length = returns + float(reward), length + 1
            episode_start[0] = terminated or truncated
            if episode_start[0]:
                ending = "truncated" if truncated else "terminated"
                print(f"EPISODE {played} {ending}: {returns:g} in {length} steps")
                returns, length, played = 0.0, 0, played + 1
                obs, _info = env.reset()
    finally:
        env.close()
