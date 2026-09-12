"""The Atari task: which game is played, under which ALE settings, and the type of the
vectorised env that results. Everything here is a property of the environment, not of whatever
is learning in it."""

from typing import Any

import ale_py
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.vector import VectorEnv

# Beside the id every path builds from, so that importing this module is what makes ALE's envs
# constructible.
gym.register_envs(ale_py)

type AtariVecEnv = VectorEnv[
    npt.NDArray[np.uint8], npt.NDArray[np.uint64], npt.NDArray[Any]
]

# Two names for one env: the gym id everything is constructed from, and the short one that
# labels run directories. The id also goes in the hparams, where it is what says which task a
# curve belongs to.
ENV_ID = "ALE/Breakout-v5"
ENV_NAME = "breakout"
PROTOCOL = dict[str, Any](
    repeat_action_probability=0.25,  # sticky actions (Machado et al. 2018)
    frameskip=4,
    stack_num=4,
    noop_max=30,
    use_fire_reset=True,
    episodic_life=False,  # True changes the task; report it if used
    reward_clipping=True,  # clip to [-1, 1] for training
    img_height=84,
    img_width=84,
)
EVAL_PROTOCOL = PROTOCOL | dict[str, Any](
    reward_clipping=False,
    # Already False above; restated so that flipping it there cannot reach evaluation, where a
    # per-life return is not a game score.
    episodic_life=False,
)
# btr's epsilon schedules are written in ALE frames; the loop counts env steps.
FRAMES_PER_STEP: int = PROTOCOL["frameskip"]
