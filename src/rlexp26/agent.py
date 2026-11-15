"""The interface the training loop is written against: everything the loop needs from whatever
is learning, and nothing about how it learns.

The loop owns the environment, the clock, the writer and the checkpoints. The agent owns its
networks, its optimizer and its own memory -- a replay buffer, a rollout buffer, or nothing at
all -- so what a timestep has to carry and when a gradient step is worth taking are its
decisions rather than the loop's.

Nothing here names a kind of environment. Observations and actions are PyTrees of arrays whose
leaves lead with the environment axis: one array for a ``Box`` or a ``Discrete`` space, a dict
of them for a ``Dict``. An agent is built from the environment's spaces and says there what it
supports.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import numpy as np
import numpy.typing as npt
from jaxtyping import PyTree


class EnvStep(NamedTuple):
    """
    What every environment answered to one batch of actions, host-side, indexed by environment
    throughout. The observations the actions were chosen from are the previous step's
    ``next_obs``, or ``Agent.init``'s.

    ``extras`` is what ``act`` returned alongside the actions, handed straight back and still on
    the device: the loop neither reads nor transfers it. It is the agent's channel from the
    moment it acted to the moment it stores: the behaviour policy's log-probability, a value
    estimate, the recurrent state it acted from, a latent of the observation.

    ``info`` is the environment's, untouched: what a task says beyond its reward.

    Under Gymnasium's next-step autoreset the step *after* an episode's end carries the new
    episode's first observation with actions and rewards the environment ignored, and completes
    no transition. Its ``extras`` still describe a real observation: the one the episode ended
    on.
    """

    actions: PyTree[npt.NDArray[Any]]
    rewards: npt.NDArray[np.floating[Any]]
    terminated: npt.NDArray[np.bool_]
    truncated: npt.NDArray[np.bool_]
    next_obs: PyTree[npt.NDArray[Any]]
    info: Mapping[str, Any]
    extras: PyTree[jax.Array]


class Policy(ABC):
    """
    An agent's acting weights at one moment, as the policy a score is reported under rather
    than the one that collects experience. Immutable and safe to hand to another thread.

    Whatever acting carries from one step to the next -- a random key, a recurrent state -- is
    in the ``state`` the caller threads through ``act``, not in the policy. So a newer snapshot
    of the same agent takes over mid-episode on the state the older one left.
    """

    @abstractmethod
    def initial_state(self, num_envs: int, *, key: jax.Array) -> PyTree[jax.Array]:
        """The state to act from before any observation, for this many environments."""

    @abstractmethod
    def act(
        self,
        obs: PyTree[npt.NDArray[Any]],
        state: PyTree[jax.Array],
        *,
        episode_start: npt.NDArray[np.bool_],
    ) -> tuple[PyTree[jax.Array], PyTree[jax.Array]]:
        """
        One action per environment and the state to pass to the next call, both still on the
        device.

        ``episode_start`` marks the environments whose ``obs`` opens an episode; whatever
        ``state`` carries across steps is started afresh there.
        """


class Agent(ABC):
    """
    A learner: its networks, its optimizer, its memory, and whatever state training needs between
    steps -- target networks, moments, exploration schedules -- all kept inside itself.

    The run drives it in one order and only that order: ``init`` once, then ``act`` /
    ``observe`` / ``learn_step`` per iteration, with the ``act`` on a step's ``next_obs``
    dispatched before that step is observed. ``restore`` comes before ``init``, so that an agent
    whose memory is sized or seeded from the checkpoint can build it knowing both.

    The last ``num_heldout`` environments are held out: the agent acts in them with the policy
    that collects experience, as in the rest, and never learns from them. They exist for its
    diagnostics, reported through ``learn_step``, of how it does on data it has not trained on.
    """

    @property
    @abstractmethod
    def num_updates(self) -> int:
        """
        Gradient steps taken over the life of the run, restores included. The agent's own
        schedules are written against it and the run paces evaluation and checkpointing by it,
        so it is checkpointed here rather than counted twice.
        """

    @property
    @abstractmethod
    def hyperparameters(self) -> dict[str, bool | int | float | str]:
        """
        Everything that defines this agent's configuration, keyed by the name it should appear
        under in the HParams table. Keys are namespaced by the agent, since the run merges them
        with the environment's.
        """

    @abstractmethod
    def init(self, obs: PyTree[npt.NDArray[Any]], *, num_heldout: int) -> None:
        """
        The environments' first observations, before any action is taken: the number of
        environments is their leading axis, the last ``num_heldout`` of them held out. An agent
        that cannot hold any out raises unless it is 0.
        """

    @abstractmethod
    def act(
        self,
        obs: PyTree[npt.NDArray[Any]],
        *,
        num_env_steps: int,
        episode_start: npt.NDArray[np.bool_],
    ) -> tuple[PyTree[jax.Array], PyTree[jax.Array]]:
        """
        One action per environment from the policy that collects experience, and the ``extras``
        the matching ``EnvStep`` hands back, both still on the device. The run transfers the
        actions and leaves the extras where they are.

        ``num_env_steps`` is the run's step count, which is what an agent whose exploration is
        scheduled reads its schedule from. ``episode_start`` marks the environments whose
        ``obs`` opens an episode, where a state carried across steps starts afresh.
        """

    @abstractmethod
    def observe(self, step: EnvStep) -> None:
        """
        Hand over one step of every environment. What is kept, and for how long, is the agent's.
        """

    @abstractmethod
    def learn_step(self) -> tuple[int, dict[str, jax.Array]]:
        """
        Whatever learning the data collected so far calls for, dispatched and left in flight:
        nothing while a replay buffer is still filling, one gradient step in the steady state, a
        whole pass over a rollout for an agent that learns in batches.

        Returns how many gradient steps this call took -- zero being the signal that there was
        nothing to do -- and the scalars to log for it, keyed by scalar name, empty on a step the
        agent does not report on. The cadence is the agent's because the cost of producing a
        diagnostic is: some are free beside the step and some are most of an event file. The
        scalars are still on the device, and the run transfers whatever it is handed.
        """

    @abstractmethod
    def stats(self) -> dict[str, float]:
        """
        Scalars describing the agent's own state rather than a gradient step's -- its memory,
        the actions it has been taking -- read on the run's logging cadence and written under
        the names given. Cheap: this is called whether or not the agent has learned anything.
        """

    @abstractmethod
    def policy(self, *, num_env_steps: int) -> Policy:
        """
        The acting weights as they stand, detached from further updates, as evaluated at the
        run's step count ``num_env_steps``.
        """

    @abstractmethod
    def checkpointables(self) -> dict[str, Any]:
        """
        Everything a resumed run cannot rebuild, read at the moment of the call, keyed for orbax.

        The same mapping serves as the abstract state a restore is read *into*, so it has to
        describe the tree even when the values in it are fresh.
        """

    @abstractmethod
    def restore(self, restored: Mapping[str, Any]) -> None:
        """Take back what ``checkpointables`` wrote, and re-derive anything that hangs off it."""
