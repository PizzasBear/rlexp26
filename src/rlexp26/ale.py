"""The Atari task: which game is played, under which ALE settings, and what counts as its end.
Everything here is a property of the environment, not of whatever is learning in it."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import ale_py
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from ale_py import roms
from cappa import Arg, Env
from gymnasium.vector import VectorEnv

gym.register_envs(ale_py)

# Games played from a patched ROM, written by ../rlexp26's scripts/patch_<game>_rom.py: Phoenix
# without its wave-transition hang, Qbert without its endless round-end bonus. ale-py loads only
# ROMs whose MD5 is in its md5.json, so its lookup is replaced for these games.
PATCHED_ROMS = {
    name: Path(__file__).resolve().parents[2] / "ale_roms" / f"{name}.bin"
    for name in ("phoenix", "qbert")
}
_get_rom_path = roms.get_rom_path
roms.get_rom_path = lambda name: PATCHED_ROMS.get(name) or _get_rom_path(name)

type AtariVecEnv = VectorEnv[
    npt.NDArray[np.uint8], npt.NDArray[np.int64], npt.NDArray[Any]
]

PROTOCOL = dict[str, Any](
    repeat_action_probability=0.25,  # sticky actions (Machado et al. 2018)
    frameskip=4,
    stack_num=4,
    noop_max=30,
    use_fire_reset=False,
    episodic_life=False,
    img_height=84,
    img_width=84,
)


class TrainingTerminal(StrEnum):
    GAME_OVER = "game-over"
    # Every life lost terminates a training transition while the game goes on, as Dopamine's
    # terminal_on_life_loss. The step after one is real, and the replay memory drops it as an
    # autoreset step.
    LIFE_LOSS = "life-loss"


@dataclass(frozen=True)
class Task:
    # A Gymnasium ALE id, played under PROTOCOL. Training rewards are clipped to [-1, 1];
    # evaluation's are not, and evaluation always plays to the game's end.
    env_id: Annotated[str, Arg(default=Env("ENV_ID"))] = "ALE/BattleZone-v5"
    training_terminal: Annotated[
        TrainingTerminal, Arg(default=Env("TRAINING_TERMINAL"))
    ] = TrainingTerminal.GAME_OVER

    @property
    def name(self) -> str:
        """The ROM's name, which also labels the run."""
        game: str = gym.spec(self.env_id).kwargs["game"]
        return game

    @property
    def frames_per_step(self) -> int:
        frameskip: int = PROTOCOL["frameskip"]
        return frameskip

    def make_env(self, num_envs: int, *, evaluation: bool) -> AtariVecEnv:
        life_loss = (
            not evaluation and self.training_terminal is TrainingTerminal.LIFE_LOSS
        )
        env: AtariVecEnv = gym.make_vec(
            self.env_id,
            num_envs=num_envs,
            **PROTOCOL,
            reward_clipping=not evaluation,
            life_loss_info=life_loss,
        )
        return env

    def episode_over(
        self, terminated: npt.NDArray[np.bool_], info: dict[str, Any]
    ) -> npt.NDArray[np.bool_]:
        """Which of a training env's terminations end the game, not just a life."""
        match self.training_terminal:
            case TrainingTerminal.GAME_OVER:
                return terminated
            case TrainingTerminal.LIFE_LOSS:
                lives: npt.NDArray[np.int32] = info["lives"]
                over: npt.NDArray[np.bool_] = np.logical_and(terminated, lives == 0)
                return over

    def hyperparameters(self) -> dict[str, bool | int | float | str]:
        rom = roms.get_rom_path(self.name)
        assert rom is not None and rom.exists(), f"no ROM for {self.name}: {rom}"
        return {
            "env/id": self.env_id,
            "env/rom_md5": hashlib.md5(rom.read_bytes()).hexdigest(),
            "env/training_terminal": self.training_terminal.value,
        } | {f"env/{name}": value for name, value in PROTOCOL.items()}
