"""
A replay buffer for vectorised environments, with optional prioritised sampling.

Every array is stored as ``[env, slot, ...]``: each of ``num_envs`` environments owns a circular
buffer of ``env_capacity`` slots. Transitions are addressed by the flat index
``env * env_capacity + slot``, which :meth:`ReplayBuffer.sample` returns and
:meth:`ReplayBuffer.update_prios` takes back.

Successor observations are not stored: slot ``i`` reads its ``next_obs`` from slot ``i + 1``, and a
frame stack reads backwards from ``i``. That halves the memory, at the cost of an unsamplable
window around each write head.

Episode boundaries come from :meth:`ReplayBuffer.save_step`'s ``terminated`` and ``truncated``
flags, under Gymnasium's default ``AutoresetMode.NEXT_STEP``. :meth:`ReplayBuffer.reset` is only
for the opening observation and for abandoning an episode by hand.

Only one frame is stored per slot; :meth:`ReplayBuffer.sample` reassembles the stack. Frames lead
the observation shape, ``(num_envs, obs_stack, *obs_shape)``, so each is one contiguous memcpy; a
model wanting them trailing transposes on the accelerator.

Sampling is proportional to the values last passed to :meth:`ReplayBuffer.update_prios`, so a
caller wanting ``p ** alpha`` applies ``alpha`` first. :meth:`ReplayBuffer.sample` returns the
stored priorities unnormalised, leaving the importance-sampling denominator, and ``beta``, to the
caller.
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

        ``obs_shape`` and ``act_shape`` describe one observation and action. Either dtype may be
        ``uint8``, ``uint16``, ``uint32``, ``uint64``, ``int8``, ``int16``, ``int32``, ``int64``,
        ``float16``, ``bfloat16``, ``float32`` or ``float64``; anything else raises ``TypeError``,
        as does ``bfloat16`` where ``ml_dtypes`` has not registered it with ``numpy``.
        :meth:`reset` and :meth:`save_step` take ``float16`` and ``bfloat16`` only as real
        ``numpy`` arrays of that dtype.

        ``obs_stack`` makes :meth:`sample` return that many consecutive frames per observation and
        lets :meth:`save_step` take a stacked observation. It does not change the memory needed.

        ``use_prios`` enables prioritised sampling. ``stratified`` splits the priority mass into
        ``batch_size`` equal slices and takes one draw from each, which changes how the draws
        correlate but not their marginal. It defaults to ``use_prios`` and requires it. A slice
        holding nothing samplable is drawn from the whole mass instead.

        ``max_prio`` seeds the priority new transitions are given, so a resumed run can refill at
        the scale training reached. It must be finite and non-negative, and is read-only once the
        buffer exists. See :meth:`update_prios` for ``max_prio_decay``, and :meth:`sample` for
        n-step returns.

        ``env_capacity`` must be at least three: the write head and the slot holding its
        ``next_obs`` are always spoken for.

        Raises ``MemoryError`` if the buffer does not fit, ``OverflowError`` if its flat index
        would not fit in a ``uint32``, and ``ValueError`` for out-of-range parameters.
        """

    def __len__(self) -> int:
        """
        Number of transitions written, summed across environments. Includes the unsamplable
        window around each write head, so it is more than :meth:`sample` can draw.
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
        independently. Settable, since it changes nothing stored; setting it on a buffer built
        without ``use_prios`` raises ``ValueError``.
        """

    @stratified.setter
    def stratified(self, value: bool) -> None: ...
    @property
    def max_prio(self) -> float:
        """
        The priority a newly written transition is given (see :meth:`update_prios`), and so the
        denominator of an importance-sampling weight normalised against the whole buffer.

        Not settable, unlike :attr:`max_prio_decay`: every stored priority was written against the
        value in force at the time.
        """

    @property
    def max_prio_decay(self) -> float:
        """
        How fast :attr:`max_prio` decays, once per :meth:`update_prios` call, so its half-life is
        in gradient steps. Settable; raises ``ValueError`` outside ``(0, 1]``.
        """

    @max_prio_decay.setter
    def max_prio_decay(self, value: float) -> None: ...
    def reset(self, obs: npt.NDArray[Any]) -> None:
        """
        Seed every environment with its initial observation.

        ``obs`` is ``(num_envs, *obs_shape)``, or ``(num_envs, obs_stack, *obs_shape)`` of which
        only the newest (last) frame is kept, and already in the buffer's dtype; nothing is cast.

        Normally called once: autoreset boundaries come from :meth:`save_step`'s flags. Called
        mid-run it abandons the current episode, as ``gym.Env.reset`` does. The head's observation
        is the ``next_obs`` of the transition before it, so where that transition is mid-episode it
        is closed off as a truncation and the new episode starts one slot later. Otherwise the
        head is overwritten and no slot is spent.

        Raises ``TypeError`` on a dtype mismatch and ``ValueError`` on a shape mismatch, before
        anything is written.
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

        Every argument is indexed by environment, and ``next_obs`` takes the shapes ``obs`` does in
        :meth:`reset`. Where ``terminated`` or ``truncated`` is set it is the episode's final
        observation. The call after that carries the new episode's first observation with an
        action and reward the environment ignored; it completes no transition and does not grow
        ``len(self)``.

        Raises ``TypeError`` on a dtype mismatch and ``ValueError`` on a shape mismatch, before
        anything is written.
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
        ``(batch_size, *obs_shape)`` without a frame stack, frames oldest first. A stack reaching
        past an episode's start repeats its oldest frame, as a frame-stack wrapper does. ``act`` is
        ``(batch_size, *act_shape)`` and the rest are ``(batch_size,)``.

        ``indices`` go back to :meth:`update_prios`. ``prios`` are the stored priorities,
        unnormalised, and ``1.0`` throughout without priorities: :attr:`sum_prios` turns them into
        sampling probabilities, :attr:`max_prio` into a buffer-wide weight. Transitions too close
        to a write head are redrawn, which leaves the priorities untouched.

        ``rewards`` is the ``n_steps`` return and ``next_obs`` the observation ``n_steps`` later,
        or where the episode ended if sooner. ``terminals`` marks true episode ends only, since a
        time-limit truncation is still bootstrapped from. A rollout meeting a truncation before
        its ``n_steps`` are up is redrawn, since the caller applies one ``discount ** n_steps`` to
        the whole batch.

        Raises ``ValueError`` if ``discount`` is missing or out of range, ``batch_size`` is zero,
        or nothing can be drawn yet, and ``RuntimeError`` if the priorities have collapsed onto
        slots that cannot be drawn.
        """

    def update_prios(
        self,
        indices: npt.NDArray[np.uint32],
        prios: npt.NDArray[np.float32],
    ) -> None:
        """
        Replace the priorities at ``indices`` with ``prios``.

        Both arrays must be the same length and the priorities finite and non-negative. The whole
        batch is checked first, so a bad one leaves the buffer untouched. The last value for a
        duplicate index wins. No exponent is applied.

        Also sets :attr:`max_prio` to the larger of the highest value in ``prios`` and its
        previous value times ``max_prio_decay``. A running maximum would keep handing new
        transitions the priority of early training's large TD errors.

        Pass real ``numpy`` arrays of exactly these dtypes: anything else is converted element by
        element, which costs far more than the update.

        Raises ``ValueError`` if the buffer was built without priorities or a priority is invalid,
        and ``IndexError`` if an index is out of range.
        """

    @property
    def sum_prios(self) -> float:
        """
        Sum of every stored priority, which turns :meth:`sample`'s priorities into probabilities.

        Raises ``ValueError`` if the buffer was built without priorities.
        """
