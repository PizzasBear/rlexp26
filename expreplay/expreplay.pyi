"""
Experience replay for vectorised environments, as independent parts that agree on one thing: a
handle names an observation by its environment and its timestep there.

- `Timeline` knows where episodes begin and end. It holds no data.
- `Column` holds one field per timestep. It knows nothing of episodes.
- `PrioTree` holds one weight per timestep and draws in proportion to them.
- `Uniform` draws over everything a timeline has written.

npt.NDArray[np.int64]
-------
A batch of handles is an ``int64`` array whose last axis is ``(env, t)``: ``t`` is the count of
observations the environment had written before this one. A timestep is never reused for a
different observation, so a handle stays meaningful after the ring has overwritten what it named,
and every reader can tell that it has. The action taken from an observation and the reward it
earned belong to the same handle; ``handles - (0, 1)`` is the timestep before.

Every method takes handles of any leading shape and answers in that shape.

Nothing here computes on a stored value. Returns, discounts and priorities are the caller's.
"""

import enum
from typing import Any, final

import numpy as np
import numpy.typing as npt

class EpisodeEnd(enum.Enum):
    """What a rollout does on reaching the end of its episode."""

    PAD = 0
    """
    Stay inside the episode: the steps past a terminal are padding. A truncation ends the rollout
    only as its last step, so every rollout bootstraps after the same number of steps unless it
    terminated.
    """
    CONTINUE = 1
    """Run on into the following episodes, with `Rollout.kinds` marking each boundary."""

class Kind(enum.IntEnum):
    """What a timestep holds beyond its observation."""

    STEP = 0
    """A transition whose episode carries on into the next timestep."""
    TERMINAL = 1
    """A transition that ended its episode: nothing is bootstrapped past it."""
    TRUNCATED = 2
    """
    A transition cut short by a time limit or a mid-run reset. The next timestep holds the
    observation to bootstrap from.
    """
    FINAL = 3
    """An observation with no action: the one an episode ended on."""

@final
class Span:
    """
    The stretch of timesteps a draw reads: ``length`` steps forward from the drawn handle, and
    the ``stack - 1`` observations before it.
    """

    def __new__(
        cls, length: int, *, stack: int = 1, episode_end: EpisodeEnd
    ) -> Span: ...
    @property
    def length(self) -> int: ...
    @property
    def stack(self) -> int: ...
    @property
    def episode_end(self) -> EpisodeEnd: ...

@final
class Rollout:
    """
    The timesteps a batch of draws read, for gathering out of any `Column`. ``lead`` is the
    leading shape of the handles it was made from.
    """

    @property
    def steps(self) -> npt.NDArray[np.int64]:
        """
        ``[*lead, length, 2]``: the transitions, in order. With `EpisodeEnd.PAD` the steps past a
        terminal repeat it.
        """
    @property
    def frames(self) -> npt.NDArray[np.int64]:
        """
        ``[*lead, stack + length, 2]``: one window of observations, oldest first, covering the
        stack of every step and of the observation after the last. Observation ``j``'s stack is
        elements ``j : j + stack``, to be sliced on the accelerator. Before the episode's start the
        window repeats its first observation, and with `EpisodeEnd.PAD` it repeats the final one
        past the end.

        With `EpisodeEnd.CONTINUE` only the window's own start is held to its episode: a stack
        that straddles a later boundary mixes two episodes, and `kinds` says where.
        """
    @property
    def kinds(self) -> npt.NDArray[np.uint8]:
        """``[*lead, length]``: the `Kind` of each of `steps`."""
    @property
    def lengths(self) -> npt.NDArray[np.uint32]:
        """``[*lead]``: the real steps, short of ``length`` only where the episode terminated."""

@final
class Timeline:
    """
    The episode structure of ``num_envs`` environments, each a ring of ``env_capacity`` timesteps.

    It follows Gymnasium's next-step autoreset: the step that ends an episode carries its final
    observation, and the call after it carries the next episode's first observation with an
    action and reward the environment ignored. That call completes no transition.
    """

    def __new__(cls, num_envs: int, env_capacity: int) -> Timeline: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def env_capacity(self) -> int: ...
    @property
    def newest(self) -> npt.NDArray[np.int64]:
        """
        ``[num_envs]``: each environment's newest written timestep, -1 before its first. A
        timestep's age is its distance behind this.
        """
    def reset(self) -> npt.NDArray[np.int64]:
        """
        Starts an episode in every environment and returns the handles, ``[num_envs, 2]``, their
        opening observations take.

        Normally called once: autoreset boundaries come from `step`'s flags. Called mid-run it
        abandons the current episodes. The newest observation is then the one the transition
        before it bootstraps from, so that transition becomes a truncation and the new episode
        starts a timestep later. A newest observation no transition leads to is overwritten
        instead, and its handle is handed out again.
        """
    def step(
        self, terminated: npt.NDArray[np.bool_], truncated: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        """
        Records one environment step each and returns ``(obs, stepped)``: the handles,
        ``[num_envs, 2]``, the incoming observations take, and a ``[num_envs]`` mask.

        Where ``stepped`` is set, the call completed the transition at ``obs - (0, 1)``, and its
        action and reward belong there. It is clear on the call after an episode's end, whose
        action and reward the environment ignored.
        """
    def valid(
        self, handles: npt.NDArray[np.int64], span: Span
    ) -> npt.NDArray[np.bool_]:
        """
        Whether a draw of ``span`` may start at each handle: the one rule every sampler rejects by.

        The handle must be live, and so must the ``stack - 1`` observations before it, back to the
        start of its episode at the furthest. An episode's start is read off the timestep before
        it, so that one must be live as well. With `EpisodeEnd.PAD` the handle must be a
        transition, and the rollout must reach its last step or a terminal without meeting the
        newest observation or a truncation with steps to go. With `EpisodeEnd.CONTINUE` the
        ``length`` timesteps after it must all be written.
        """
    def rollout(self, handles: npt.NDArray[np.int64], span: Span) -> Rollout:
        """
        The timesteps each draw of ``span`` reads, in the leading shape of ``handles``.

        Raises `ValueError` on a start that `valid` refuses.
        """

@final
class Column:
    """
    A field stored per timestep, in a ring of ``env_capacity`` timesteps per environment. A row is
    ``shape`` elements of ``dtype``, which may be any NumPy dtype without Python objects.

    A column follows its own writes and no timeline, so it can hold a field written on some other
    schedule, such as a recurrent state written back after training. Reads and writes release the
    GIL.

    Raises `MemoryError` where the storage cannot be allocated.
    """

    def __new__(
        cls,
        num_envs: int,
        env_capacity: int,
        *,
        dtype: npt.DTypeLike,
        shape: tuple[int, ...] = (),
    ) -> Column: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def env_capacity(self) -> int: ...
    @property
    def dtype(self) -> np.dtype[Any]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def write(
        self, handles: npt.NDArray[np.int64], values: npt.NDArray[Any]
    ) -> npt.NDArray[np.bool_]:
        """
        Writes ``values``, of shape ``[*handles.shape[:-1], *shape]``, one row per handle and in
        order, and returns which landed.

        A handle the ring has moved past is skipped: a value computed from a draw may arrive after
        its timestep was overwritten. A handle beyond the newest timestep moves the ring up to it,
        zeroing the timesteps in between.

        ``values`` must already be an array of the column's dtype, or it is a `TypeError`: nothing
        is cast on the way in.
        """
    def read(self, handles: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        """
        The row at each handle, as ``[*handles.shape[:-1], *shape]``.

        Raises `IndexError` on a handle the column no longer holds, or does not hold yet.
        """

@final
class PrioTree:
    """
    A weight per timestep, in a ring of ``env_capacity`` per environment, with a sum tree over
    them.

    It is a column of weights and follows its own writes, as `Column` does. What a weight means
    is the caller's: a priority, a priority scaled by age, or one for a uniform draw. A timestep
    with no weight written is never drawn, so the caller writes one as each transition completes.
    """

    def __new__(
        cls, num_envs: int, env_capacity: int, *, seed: int | None = None
    ) -> PrioTree: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def env_capacity(self) -> int: ...
    @property
    def total(self) -> float:
        """The sum of every weight. A draw's probability is its weight over this."""
    def write(
        self, handles: npt.NDArray[np.int64], weights: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.bool_]:
        """
        Sets the weight at each handle, in order, and returns which landed.

        Stale and skipped handles are treated as `Column.write` treats them, a skipped timestep's
        weight being zero. A weight that is not finite and non-negative is a `ValueError`, raised
        before anything is written.
        """
    def read(self, handles: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        """The weight at each handle. Raises `IndexError` as `Column.read` does."""
    def sample(
        self, batch: int, timeline: Timeline, span: Span, *, stratified: bool = False
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """
        Draws ``batch`` handles with replacement, in proportion to weight among those ``span`` can
        start at on ``timeline``, and returns ``(handles, weights)``: ``[batch, 2]`` and each
        one's weight.

        ``stratified`` cuts the weight mass into ``batch`` equal slices and draws one handle from
        each, falling back to the whole mass for a slice that keeps proposing invalid handles.

        Raises `RuntimeError` when every weight is zero, or when a draw gives up: nearly all the
        weight then sits on timesteps the span cannot start at.
        """

@final
class Uniform:
    """Draws over every timestep a timeline holds, equally likely."""

    def __new__(cls, *, seed: int | None = None) -> Uniform: ...
    def sample(
        self, batch: int, timeline: Timeline, span: Span
    ) -> npt.NDArray[np.int64]:
        """
        Draws ``batch`` handles, ``[batch, 2]``, uniformly and with replacement over those
        ``span`` can start at on ``timeline``.

        Raises `RuntimeError` when a draw gives up: the timeline is empty, or too young for the
        span.
        """
