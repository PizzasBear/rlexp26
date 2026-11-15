"""The Q-learner's replay memory, host-side: what is stored, what is drawn and how often."""

import math
from collections import deque
from fractions import Fraction
from typing import Any, NamedTuple, assert_never

import jax
import numpy as np
import numpy.typing as npt

from expreplay import Column, EpisodeEnd, Kind, PrioTree, Span, Timeline

from .agent import EnvStep
from .config import Replay, ReplaySchedule

# Bursts whose priorities stay in flight after a write-back.
PRIO_QUEUE_SLACK = 2
# max_prio's factor per write-back, so that one outlier does not set the insert priority for good.
MAX_PRIO_DECAY = 0.999


class Batch(NamedTuple):
    """
    The batches of one burst of gradient steps, laid out ``[steps, batch, ...]``: one stratified
    draw of steps x batch, dealt out so that every step has one from each coarse stratum of the
    whole weight mass.
    """

    prios: npt.NDArray[np.float32]
    # What each was drawn in proportion to.
    weights: npt.NDArray[np.float32]
    # ``[steps, batch, obs_stack + n_step, ...]``: the observation is the first ``obs_stack``
    # frames and the one to bootstrap from the last ``obs_stack``.
    frames: npt.NDArray[Any]
    actions: npt.NDArray[np.uint8]
    # ``[steps, batch, n_step]``, of which the first ``lengths`` are real.
    rewards: npt.NDArray[np.float32]
    lengths: npt.NDArray[np.uint32]
    terminated: npt.NDArray[np.bool_]


class Draw(NamedTuple):
    """A burst's ``batch``, and what ``ReplayMemory.update`` needs to write back to it."""

    handles: npt.NDArray[np.int64]
    # Timesteps behind its environment's newest, at the draw.
    ages: npt.NDArray[np.int64]
    batch: Batch


class ReplayMemory:
    """
    Frames, actions and rewards per timestep, a priority each, and the sum tree transitions are
    drawn from, in proportion to priority (FIXED) or to priority / (1 + age / gper_half_age)
    (GPER). ``obs`` is the environments' first observations, ``[num_envs,
    obs_stack, ...]``, of which only the newest frame is stored.

    A new transition takes ``max_prio``, the largest priority written back lately. Priorities
    come back a burst at a time and late: ``update`` queues a burst's while they are still on
    the device, and ``observe`` writes back every burst but the PRIO_QUEUE_SLACK newest, so
    that it never waits on one still running. Until then a burst's transitions are drawn at the
    weight they were drawn on.
    """

    def __init__(
        self,
        hp: Replay,
        obs: npt.NDArray[Any],
        *,
        n_step: int,
        seed: int,
        max_prio: float = 1.0,
    ) -> None:
        num_envs, obs_stack = obs.shape[:2]
        env_capacity = hp.buffer_size // num_envs
        if hp.train_start_buf_size // num_envs < obs_stack + n_step - 1:
            raise ValueError(
                f"train_start_buf_size {hp.train_start_buf_size} over {num_envs} environments "
                f"is short of the {obs_stack + n_step - 1} transitions each that a "
                f"{obs_stack}-frame stack and a {n_step}-step rollout need"
            )
        self.hp = hp
        self.max_prio = max_prio
        self.span = Span(n_step, stack=obs_stack, episode_end=EpisodeEnd.PAD)
        self.timeline = Timeline(num_envs, env_capacity)
        self.frames = Column(
            num_envs, env_capacity, dtype=obs.dtype, shape=obs.shape[2:]
        )
        self.actions = Column(num_envs, env_capacity, dtype=np.uint8)
        self.rewards = Column(num_envs, env_capacity, dtype=np.float32)
        self.prios = Column(num_envs, env_capacity, dtype=np.float32)
        self.tree = PrioTree(num_envs, env_capacity, seed=seed)
        self.frames.write(self.timeline.reset(), np.ascontiguousarray(obs[:, -1]))
        self._queue = deque[
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], jax.Array]
        ]()
        self._steps_owed: Fraction | float = Fraction(0)

    def __len__(self) -> int:
        """Timesteps held, over all environments."""
        return int(
            np.minimum(self.timeline.newest + 1, self.timeline.env_capacity).sum()
        )

    def observe(self, step: EnvStep) -> None:
        """Writes back the priorities that have landed, stores the step and adds what it owes."""
        self.write_back()
        obs, stepped = self.timeline.step(step.terminated, step.truncated)
        self.frames.write(obs, np.ascontiguousarray(step.next_obs[:, -1]))
        transitions = (obs - (0, 1))[stepped]
        self.actions.write(transitions, step.actions[stepped].astype(np.uint8))
        self.rewards.write(transitions, step.rewards[stepped].astype(np.float32))
        prios = np.full(len(transitions), self.max_prio, np.float32)
        self.prios.write(transitions, prios)
        self.tree.write(transitions, prios)

        num_envs = self.timeline.num_envs
        match self.hp.replay_schedule:
            case ReplaySchedule.FIXED:
                self._steps_owed += self.hp.replay_ratio * num_envs
            case ReplaySchedule.GPER:
                # len saturates, so the ratio stops growing once the memory is full.
                self._steps_owed += (
                    self.hp.gper_reviews_per_doubling
                    * math.log1p(len(self) / self.hp.gper_half_age)
                    / (self.hp.batch_size * math.log(2))
                    * num_envs
                )
            case _:
                assert_never(self.hp.replay_schedule)

    def draw(self) -> Draw | None:
        """
        The batches of ``burst_steps`` of the gradient steps owed, or None while fewer are.
        Nothing is owed before ``train_start_buf_size`` timesteps are held: steps carried over
        the fill would come due at once on a single policy's data.
        """
        if len(self) < self.hp.train_start_buf_size:
            self._steps_owed *= 0
            return None
        if self._steps_owed < self.hp.burst_steps:
            return None
        self._steps_owed -= self.hp.burst_steps

        steps, batch_size = self.hp.burst_steps, self.hp.batch_size
        handles, weights = self.tree.sample(
            steps * batch_size, self.timeline, self.span, stratified=True
        )
        # The draws come in stratum order: step i takes every steps-th from the i-th.
        handles = handles.reshape(batch_size, steps, 2).swapaxes(0, 1)
        rollout = self.timeline.rollout(handles, self.span)
        return Draw(
            handles=handles,
            ages=self.timeline.newest[handles[..., 0]] - handles[..., 1],
            batch=Batch(
                prios=self.prios.read(handles),
                weights=weights.reshape(batch_size, steps).T,
                frames=self.frames.read(rollout.frames),
                actions=self.actions.read(handles),
                rewards=self.rewards.read(rollout.steps),
                lengths=rollout.lengths,
                # Past a terminal the steps repeat it.
                terminated=rollout.kinds[..., -1] == Kind.TERMINAL,
            ),
        )

    def update(self, draw: Draw, prios: jax.Array) -> None:
        """Queues ``draw``'s new priorities, laid out as it is and still on the device."""
        prios.copy_to_host_async()
        self._queue.append((draw.handles, draw.ages, prios))

    def write_back(self, keep: int = PRIO_QUEUE_SLACK) -> None:
        """
        Writes back every queued burst but the ``keep`` newest, later steps over earlier ones.
        A transition overwritten since its draw is skipped.
        """
        while keep < len(self._queue):
            handles, ages, in_flight = self._queue.popleft()
            prios: npt.NDArray[np.float32] = np.asarray(in_flight)
            match self.hp.replay_schedule:
                case ReplaySchedule.FIXED:
                    weights = prios
                case ReplaySchedule.GPER:
                    env_steps = ages * self.timeline.num_envs
                    weights = (prios / (1 + env_steps / self.hp.gper_half_age)).astype(
                        np.float32
                    )
                case _:
                    assert_never(self.hp.replay_schedule)
            self.prios.write(handles, prios)
            self.tree.write(handles, weights)
            self.max_prio = max(self.max_prio * MAX_PRIO_DECAY, float(prios.max()))
