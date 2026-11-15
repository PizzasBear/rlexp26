"""The orbax policies a run's ``ocp.training.Checkpointer`` saves and keeps checkpoints by."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from orbax.checkpoint import v1 as ocp


@dataclass(frozen=True)
class SinceLastSave(ocp.training.save_decision_policies.SaveDecisionPolicy):  # type: ignore[misc]
    """
    Save once ``interval`` steps have passed since the newest checkpoint, or at the first step
    if there is none, and never while a save is still writing. ``FixedIntervalPolicy``'s
    ``step % interval == 0`` is stepped over by an agent taking several steps an iteration.
    """

    interval: int

    def should_save(
        self,
        step: ocp.training.CheckpointMetadata[Any],
        previous_steps: Sequence[ocp.training.CheckpointMetadata[Any]],
        *,
        context: ocp.training.save_decision_policies.DecisionContext,
    ) -> bool:
        if context.is_saving_in_progress:
            return False
        return (
            not previous_steps or self.interval <= step.step - previous_steps[-1].step
        )


@dataclass(frozen=True)
class FirstPastEachMultiple(
    ocp.training.preservation_policies.PreservationPolicy  # type: ignore[misc]
):
    """
    Keep the first checkpoint at or past every positive multiple of ``interval`` steps.
    Checkpoints rarely land on a multiple, since ``SinceLastSave`` paces them from the
    previous one.
    """

    interval: int

    def should_preserve(
        self,
        checkpoints: Sequence[ocp.training.CheckpointMetadata[Any]],
        *,
        context: ocp.training.preservation_policies.PreservationContext,
    ) -> Sequence[bool]:
        first = dict[int, int]()
        for i, checkpoint in enumerate(checkpoints):
            if checkpoint.step >= self.interval:
                first.setdefault(checkpoint.step // self.interval, i)
        return [i in first.values() for i in range(len(checkpoints))]
