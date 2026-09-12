"""
A replay buffer for vectorised environments, with optional prioritised sampling.

Every array is stored as ``[env, slot, ...]``: each of the ``num_envs`` environments owns an
independent circular buffer of ``env_capacity`` slots. Transitions are addressed by the flat
index ``env * env_capacity + slot``, which is what :meth:`ReplayBuffer.sample` returns and what
:meth:`ReplayBuffer.update_prios` takes back.

Successor observations are not stored: the transition at slot ``i`` reads its ``next_obs`` from
slot ``i + 1``, and frame stacks read backwards from ``i``. This halves the memory needed, at the
cost of leaving a window of slots around each environment's write head unsamplable.

Episode boundaries are inferred from the ``terminated`` and ``truncated`` flags of each
:meth:`ReplayBuffer.save_step`, under Gymnasium's default ``AutoresetMode.NEXT_STEP``: the step that
ends an episode returns that episode's *final* observation, and the call after it returns the new
episode's first observation together with an action and a reward the environment ignored. This
buffer follows the same shape, and :meth:`ReplayBuffer.reset` is needed only for the opening
observation and for abandoning an episode by hand.

``obs_stack`` is a property of the model rather than of a draw, so it is fixed at construction;
``n_steps`` is a per-draw argument, since it is the kind of thing a schedule moves during
training. Only one frame per slot is ever stored -- the rest of a stack is already in the slots
before it -- so :meth:`ReplayBuffer.save_step` takes either a single new frame or the environment's
whole stacked observation, ``(num_envs, obs_stack, *obs_shape)``, keeping only the newest frame of
it, and :meth:`ReplayBuffer.sample` reassembles the stack on the way out.

Frames lead rather than trail so that each one is a contiguous run of bytes on both paths: ``save_step``
picks its frame out with a memcpy per environment instead of a stride-``obs_stack`` byte scatter,
and ``sample`` builds a stack as ``obs_stack`` back-to-back memcpys. A model wanting frames
trailing should transpose on the accelerator, where that move is cheap and is usually folded into
the first layer.

Sampling is proportional to the values last passed to :meth:`ReplayBuffer.update_prios`, with no
exponent applied on the way in, so a caller wanting the usual ``p ** alpha`` weighting applies
``alpha`` before calling. ``alpha`` cannot be applied after sampling: the buffer draws in
proportion to whatever it stores. :meth:`ReplayBuffer.sample` likewise hands back the stored
priorities untouched rather than normalising them, which leaves the choice of denominator -- and
so of ``beta`` and of what the weights are scaled against -- entirely on the caller's side.
"""

from typing import Any, Self

import numpy as np
import numpy.typing as npt

class ReplayBuffer:
    def __new__(
        cls,
        num_envs: int,
        env_capacity: int,
        *,
        obs_shape: tuple[int, ...],
        obs_dtype: npt.DTypeLike = np.float32,
        act_shape: tuple[int, ...] = (),
        act_dtype: npt.DTypeLike = np.uint8,
        obs_stack: int | None = None,
        use_prios: bool = False,
        max_prio: float = 1.0,
        max_prio_decay: float = 0.999,
        stratified: bool | None = None,
        seed: int | None = None,
    ) -> Self:
        """
        Allocate a buffer holding ``num_envs * env_capacity`` transitions.

        ``obs_shape`` and ``act_shape`` describe a single observation and action; the environment
        and slot axes are prepended internally. Either dtype may be any of ``uint8``, ``uint16``,
        ``uint32``, ``uint64``, ``int8``, ``int16``, ``int32``, ``int64``, ``float16``,
        ``float32`` or ``float64``; anything else raises ``TypeError``. Note that ``float16`` is
        the one dtype with no element-by-element fallback on the way in -- :meth:`reset` and
        :meth:`save_step` take a real ``numpy`` array of it and nothing else.

        ``obs_stack`` makes :meth:`sample` return that many consecutive frames per observation
        and lets :meth:`save_step` accept an environment's stacked observation directly. Only one frame
        per slot is stored either way, so it does not change how much memory the buffer needs.

        ``stratified`` splits the priority mass into ``batch_size`` equal slices and takes one
        draw from each, so a batch spreads over the distribution instead of clumping; the per-draw
        marginal is unchanged, only the draws' correlation. It defaults to ``use_prios`` and
        requires it: passing it without priorities raises ``ValueError``, since a uniform draw is
        already spread over every samplable transition and there is no mass to slice.

        A slice holding nothing samplable is drawn from the whole mass instead, rather than
        failing the call. That happens when ``sum_prios / batch_size`` falls below the priority
        on the handful of too-recent transitions behind a write head -- a small buffer, a batch
        comparable to the number of stored transitions, or a ``max_prio`` far above the rest of
        the distribution -- and costs only the draws it applies to their stratification.

        ``use_prios`` enables prioritised sampling; see :meth:`update_prios` for
        ``max_prio_decay``. ``max_prio`` seeds the priority new transitions are given, so that a
        run resuming from a checkpoint does not refill its buffer at the scale of a fresh one; it
        is held to the same rule a priority is in :meth:`update_prios`, finite and non-negative,
        and is read-only once the buffer exists. n-step returns are asked for per draw rather than
        here -- see :meth:`sample`.

        ``env_capacity`` must be at least three: the write head and the slot holding its
        ``next_obs`` are always spoken for, so anything smaller cannot hold a samplable transition.

        Raises ``MemoryError`` if the buffer does not fit, ``OverflowError`` if its flat index
        would not fit in a ``uint32``, and ``ValueError`` for out-of-range parameters.
        """

    def __len__(self) -> int:
        """
        Number of transitions stored, summed across environments.

        This counts everything written, which is more than :meth:`sample` can draw: the window
        around each write head is excluded from sampling but counted here.
        """

    @property
    def num_envs(self) -> int: ...
    @property
    def env_capacity(self) -> int: ...
    @property
    def total_capacity(self) -> int: ...
    @property
    def obs_dtype(self) -> np.dtype: ...
    @property
    def obs_shape(self) -> tuple[int, ...]: ...
    @property
    def act_dtype(self) -> np.dtype: ...
    @property
    def act_shape(self) -> tuple[int, ...]: ...
    @property
    def obs_stack(self) -> int | None:
        """Frames per observation that :meth:`sample` returns, or ``None`` without a stack."""

    @property
    def use_prios(self) -> bool: ...
    @property
    def stratified(self) -> bool:
        """
        Whether a prioritised batch is spread over the priority mass rather than drawn
        independently.

        Settable: it changes only how the draws correlate, never what is stored, so there is
        nothing to keep consistent across a change. Setting it on a buffer built without
        ``use_prios`` raises ``ValueError``.
        """

    @stratified.setter
    def stratified(self, value: bool) -> None: ...
    @property
    def max_prio(self) -> float:
        """
        The priority a newly written transition is given; see :meth:`update_prios`.

        This is the scale :meth:`sample` reports its priorities against, so an
        importance-sampling correction normalised against the whole buffer rather than against
        the batch that came back reads its denominator from here.

        Its starting value is a constructor argument; unlike :attr:`max_prio_decay` it is not
        settable, since every priority already stored was written against whatever value was in
        force at the time.
        """

    @property
    def max_prio_decay(self) -> float:
        """
        How fast :attr:`max_prio` decays, applied once per :meth:`update_prios` call.

        Settable, so that a schedule can track a changing replay ratio -- the half-life is
        measured in gradient steps. Raises ``ValueError`` outside ``(0, 1]``.
        """

    @max_prio_decay.setter
    def max_prio_decay(self, value: float) -> None: ...
    def reset(self, obs: npt.NDArray[Any]) -> None:
        """
        Seed every environment with its initial observation.

        ``obs`` is indexed by environment -- shape ``(num_envs, *obs_shape)``, or
        ``(num_envs, obs_stack, *obs_shape)`` where the buffer has a frame stack, in which case
        only the newest frame (the last along the stack axis) is kept. It must already be in the
        dtype the buffer was built with; nothing is cast on the way in.

        Normally called once, before the first :meth:`save_step`: autoreset boundaries during play are
        inferred from the ``terminated`` and ``truncated`` flags rather than from further ``reset``
        calls. Calling it again mid-run is still well defined, and means what ``gym.Env.reset``
        means -- abandon the current episode and start a new one.

        Because successor observations are not stored separately, the observation at an
        environment's write head *is* the ``next_obs`` of the transition before it. Where that
        transition exists and is still mid-episode, ``reset`` does not overwrite the head; it closes
        the slot off as a truncation -- which is what an abandoned episode is, cut short with its
        final observation known -- and starts the new episode one slot later, costing a single slot.
        Where nothing can reach the head observation, it is overwritten and no slot is spent: the
        opening ``reset``, back-to-back ``reset`` calls, and a ``reset`` straight after a terminated
        or truncated ``save_step``.

        Raises ``TypeError`` on a dtype mismatch and ``ValueError`` on a shape mismatch, in both
        cases before anything is written.
        """

    def save_step(
        self,
        actions: npt.NDArray[Any],
        rewards: npt.NDArray[np.float32],
        terminated: npt.NDArray[np.bool],
        truncated: npt.NDArray[np.bool],
        next_obs: npt.NDArray[Any],
    ) -> None:
        """
        Record one transition per environment and advance each write head.

        Every argument is indexed by environment. ``next_obs`` takes the same two shapes ``obs``
        does in :meth:`reset`, and is whatever the environment returned alongside the reward: under
        Gymnasium's default ``AutoresetMode.NEXT_STEP`` that is the episode's *final* observation
        wherever ``terminated`` or ``truncated`` is set. The call after such a step carries the new
        episode's first observation together with an action and a reward the environment ignored:
        it completes no transition and does not grow ``len(self)``.

        Raises ``TypeError`` on a dtype mismatch and ``ValueError`` on a shape mismatch, in both
        cases before anything is written.
        """

    def sample(
        self,
        batch_size: int,
        *,
        n_steps: int = 1,
        discount: float | None = None,
    ) -> tuple[
        npt.NDArray[np.uint32],  # indices
        npt.NDArray[np.float32],  # prios
        npt.NDArray[Any],  # obs
        npt.NDArray[Any],  # act
        npt.NDArray[np.float32],  # rewards
        npt.NDArray[np.bool],  # terminals
        npt.NDArray[Any],  # next_obs
    ]:
        """
        Draw ``batch_size`` transitions.

        ``obs`` and ``next_obs`` are ``(batch_size, obs_stack, *obs_shape)``, or
        ``(batch_size, *obs_shape)`` where the buffer has no frame stack. Frames run oldest to
        newest, and a stack that would reach past the start of an episode repeats its oldest frame,
        as the environment's own frame-stack wrapper does. ``act`` is ``(batch_size, *act_shape)``
        and the rest are ``(batch_size,)``.

        ``indices`` are flat indices to hand back to :meth:`update_prios`. ``prios`` holds each
        drawn transition's stored priority as it stands, and ``1.0`` throughout for a buffer
        sampling uniformly -- which is the right answer there, since a uniform draw needs no
        importance-sampling correction at all. It comes back raw rather than normalised so that
        the caller picks its own denominator: :attr:`sum_prios` turns it into the sampling
        probability, :attr:`max_prio` into a weight scaled against the whole buffer instead of
        against the batch. Transitions too close to a write head to assemble are redrawn rather
        than returned; that does not touch the priorities themselves.

        ``rewards`` is the accumulated ``n_steps`` return and ``next_obs`` the observation
        ``n_steps`` later, or wherever the episode ended if that came first. ``terminals`` marks
        only true episode ends; a transition cut short by a time limit should still be bootstrapped
        from, so it is not marked terminal. A rollout that would meet such a truncation before its
        ``n_steps`` are up is redrawn instead, since the caller applies one ``discount ** n_steps``
        to the whole batch.

        Raises ``ValueError`` if ``discount`` is missing or out of range, if ``batch_size`` is
        zero, or if no environment yet holds enough transitions to draw from, and ``RuntimeError``
        if the priorities have collapsed onto slots that cannot be drawn.
        """

    def update_prios(
        self,
        indices: npt.NDArray[np.uint32],
        prios: npt.NDArray[np.float32],
    ) -> None:
        """
        Replace the priorities at ``indices`` with ``prios``.

        Both arrays must be the same length and the priorities finite and non-negative. The whole
        batch is checked before any of it is applied, so a bad index or priority leaves the buffer
        untouched. Duplicate indices are applied in order, so the last value wins. No exponent is
        applied here.

        This also refreshes the priority given to newly written transitions, as the larger of the
        highest value in ``prios`` and the previous value decayed by ``max_prio_decay``. Decaying
        rather than keeping a running maximum matters over a long run: TD errors shrink as the
        agent improves, so a maximum that never falls would keep handing new transitions a priority
        drawn from early training and heavily oversample them. The decay applies per call, so its
        half-life is measured in gradient steps and shifts with the replay ratio.

        For speed, pass real ``numpy`` arrays of exactly these dtypes. Anything else -- a list, or
        a JAX array -- is converted element by element through the Python sequence protocol, which
        costs far more than the update itself.

        Raises ``ValueError`` if the buffer was built without priorities or a priority is invalid,
        and ``IndexError`` if an index is out of range.
        """

    @property
    def sum_prios(self) -> float:
        """
        Sum of every stored priority: the constant that turns :meth:`sample`'s priorities into
        sampling probabilities.

        Raises ``ValueError`` if the buffer was built without priorities.
        """
