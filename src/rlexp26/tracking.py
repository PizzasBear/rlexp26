"""What a run records: scalars against environment steps, its hyperparameters, and profiles."""

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams

from .agent import Agent


class Tracker(ABC):
    """
    A run's record, whatever stores it. A backend implements ``log_hyperparameters``, ``log``
    and ``close``; scalars still on the device go through ``log_later`` and ``drain`` so that
    their transfer never waits on the device.

    A backend opens at the env step its process starts from, and supersedes whatever an
    earlier process of the same run logged from there on: a resume goes back to its
    checkpoint, so the steps logged after it are replayed.
    """

    def __init__(self) -> None:
        self._pending = deque[tuple[dict[str, jax.Array], int]]()

    @abstractmethod
    def log_hyperparameters(
        self, hyperparameters: Mapping[str, bool | int | float | str]
    ) -> None:
        """Called once, at the start of the process, resumed or not."""

    @abstractmethod
    def log(self, scalars: Mapping[str, float], env_steps: int) -> None: ...

    def log_later(self, scalars: dict[str, jax.Array], env_steps: int) -> None:
        """Start the scalars' transfer and log them once ``drain`` reaches ``env_steps``."""
        jax.copy_to_host_async(scalars)  # type: ignore[no-untyped-call]
        self._pending.append((scalars, env_steps))

    def drain(self, up_to_env_steps: int) -> None:
        while self._pending and self._pending[0][1] <= up_to_env_steps:
            scalars, env_steps = self._pending.popleft()
            self.log(
                {name: float(value) for name, value in jax.device_get(scalars).items()},
                env_steps,
            )

    def close(self) -> None:
        """Log everything still pending. A backend extends this to flush and close."""
        while self._pending:
            self.drain(self._pending[-1][1])


class TensorBoardTracker(Tracker):
    """
    One event file in ``logdir`` per process. ``purge_step`` makes TensorBoard drop the earlier
    files' events from ``start_env_steps`` on. The hyperparameters are an HParams session
    written into the run's own event file: unlike ``add_hparams``, which writes a separate run, this survives a
    run that dies, and the HParams tab reads ``metrics`` by tag from the run's scalars. A resume
    writes a second session start, and the plugin reads the later one.
    """

    def __init__(
        self, logdir: Path, metrics: tuple[str, ...], start_env_steps: int
    ) -> None:
        super().__init__()
        self._writer = SummaryWriter(str(logdir), purge_step=start_env_steps)
        self._metrics = metrics
        self._session_end: Any = None

    def log_hyperparameters(
        self, hyperparameters: Mapping[str, bool | int | float | str]
    ) -> None:
        # hparams reads only the metrics' names.
        experiment, session_start, self._session_end = hparams(
            dict(hyperparameters), dict.fromkeys(self._metrics, 0.0)
        )
        assert self._writer.file_writer is not None
        self._writer.file_writer.add_summary(experiment)
        self._writer.file_writer.add_summary(session_start)

    def log(self, scalars: Mapping[str, float], env_steps: int) -> None:
        for name, value in scalars.items():
            self._writer.add_scalar(name, value, global_step=env_steps)

    def close(self) -> None:
        super().close()
        if self._session_end is not None:
            assert self._writer.file_writer is not None
            self._writer.file_writer.add_summary(self._session_end)
        self._writer.close()


class TraceWindow:
    """
    A jax.profiler trace of ``count`` gradient steps into ``logdir``, opening ``start`` steps
    after the first one this process takes, so that a resume's refill is not traced. Both ends
    wait for the device to go idle.
    """

    def __init__(self, logdir: Path, start: int, count: int) -> None:
        self._logdir = logdir
        self._start, self._count = start, count
        self._opens_at: int | None = None
        self._tracing = False

    def update(self, agent: Agent) -> None:
        """Call after every learn_step that took a gradient step."""
        if self._opens_at is None:
            self._opens_at = agent.num_updates + self._start
        opens_at = self._opens_at
        if not self._tracing and self._count and agent.num_updates >= opens_at:
            jax.block_until_ready(agent.checkpointables())  # type: ignore[no-untyped-call]
            jax.profiler.start_trace(str(self._logdir))
            self._tracing = True
        elif self._tracing and agent.num_updates >= opens_at + self._count:
            self.close(agent)

    def close(self, agent: Agent) -> None:
        if not self._tracing:
            if self._count:
                print("PROFILE NOT TAKEN -- the run ended before the window opened")
            return
        jax.block_until_ready(agent.checkpointables())  # type: ignore[no-untyped-call]
        jax.profiler.stop_trace()  # type: ignore[no-untyped-call]
        print(f"PROFILED {self._opens_at}..{agent.num_updates} into {self._logdir}")
        self._tracing, self._count = False, 0
