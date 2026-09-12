"""The interface the training loop is written against: everything the loop needs from whatever
is learning, and nothing about how it learns.

The loop owns the environment, the clock, the writer and the checkpoints. The agent owns its
networks, its optimizer and its own memory -- a replay buffer, a rollout buffer, or nothing at
all -- so what a transition has to carry and when a gradient step is worth taking are its
decisions rather than the loop's."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import numpy as np
import numpy.typing as npt


class EnvStep(NamedTuple):
    """
    One step of every environment, host-side, indexed by environment throughout.

    ``extras`` is what ``act`` returned alongside the actions, transferred and handed straight
    back: the loop never looks inside it, so an agent that needs the behaviour policy's
    log-probability or a value estimate at training time can put it there and read it here.

    Under Gymnasium's next-step autoreset the step *after* a termination carries the new
    episode's first observation with an action and a reward the environment ignored, and
    completes no transition.
    """

    obs: npt.NDArray[Any]  # what the actions were chosen from
    actions: npt.NDArray[Any]
    rewards: npt.NDArray[Any]
    terminated: npt.NDArray[np.bool_]
    truncated: npt.NDArray[np.bool_]
    next_obs: npt.NDArray[Any]
    extras: dict[str, npt.NDArray[Any]]


class Policy(ABC):
    """
    An agent's behaviour, detached from its training: weights in, actions out.

    Built once and reloaded, so that the evaluator can keep a network and an environment alive
    across runs rather than rebuilding both every time.
    """

    @abstractmethod
    def checkpointables(self) -> dict[str, Any]:
        """
        The subset of a run's checkpoint this policy can be filled from, keyed for orbax, and
        the abstract state that subset is read into. The keys are the agent's own, so a policy
        loads either from ``Agent.policy_weights`` in memory or straight out of a checkpoint on
        disk, and only the algorithm ever names them.
        """

    @abstractmethod
    def load(self, weights: Mapping[str, Any], *, seed: int) -> None:
        """
        Fill this policy from the entries ``checkpointables`` names -- a live snapshot out of
        ``Agent.policy_weights``, or what orbax read back into that same tree -- and restart its
        own randomness at ``seed``, so that successive loads of different weights replay the
        same draws.
        """

    @abstractmethod
    def act(
        self, obs: npt.NDArray[Any], *, num_env_steps: int, evaluation: bool = False
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """
        One action per environment, still on the device.

        ``num_env_steps`` is the run's step count, which is what an agent whose exploration is
        scheduled reads its schedule from. ``evaluation`` asks for the policy a score should be
        reported under rather than the one that collects experience.

        The second return is whatever the agent wants said about the actions it just took, keyed
        by a plain name. The run logs each entry as ``run/action_<name>``, averaged over its
        logging window, and hands the whole dict back in the ``EnvStep``.
        """


class Agent(ABC):
    """
    A learner: its networks, its optimizer, its memory, and whatever state training needs between
    steps -- target networks, moments, exploration schedules -- all kept inside itself.

    The run drives it in one order and only that order: ``init`` once, then ``act`` /
    ``observe`` / ``learn_step`` per iteration. ``restore`` comes before ``init``, so that an
    agent whose memory is sized or seeded from the checkpoint can build it knowing both.
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
    def init(self, obs: npt.NDArray[Any]) -> None:
        """
        The environments' first observations, before any action is taken. An agent that stores
        transitions needs this as the ``obs`` half of the first one; the shapes and dtypes it
        sizes its memory from are here too.
        """

    @abstractmethod
    def act(
        self, obs: npt.NDArray[Any], *, num_env_steps: int, evaluation: bool = False
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """As ``Policy.act``, on the agent's own acting weights."""

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
        nothing to do -- and the scalars to log, keyed by scalar name. The scalars are still on
        the device: the run transfers only the ones it means to write.
        """

    @abstractmethod
    def stats(self) -> dict[str, float]:
        """
        Scalars describing the agent's own state rather than a gradient step's, read on the run's
        logging cadence and written under the names given. Cheap: this is called whether or not
        the agent has learned anything.
        """

    @abstractmethod
    def policy_weights(self) -> dict[str, Any]:
        """
        A snapshot of the acting weights, detached from further updates, for a ``Policy`` on
        another thread to load. Keyed as ``Policy.checkpointables`` describes, which is what
        lets the one ``load`` serve both a live handover and a restore from disk.
        """

    @abstractmethod
    def make_policy(self) -> Policy:
        """
        An empty policy shaped like this agent's, to be filled by ``Policy.load``. Called on the
        thread that will act with it, since it builds a network.
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
