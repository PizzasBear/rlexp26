"""
The command lines: ``rlexp26`` starts a run and ``rlexp26-resume`` carries one on. Nothing
here imports JAX, which reads its ``JAX_*`` settings from the environment once, at import: the
loop is imported only after ``.env`` is loaded.
"""

import dataclasses
import datetime as dt
import json
from fractions import Fraction
from pathlib import Path
from typing import Annotated

import cappa
import cattrs
from cappa import Arg, Env
from cattrs.gen import make_dict_unstructure_fn, override
from cattrs.strategies import configure_tagged_union
from dotenv import find_dotenv, load_dotenv

from .ale import Task
from .config import Recipe

# A run is LOG_ROOT/<run> (events, config.json, profiles) and CHECKPOINT_ROOT/<run>.
LOG_ROOT = Path("logs")
CHECKPOINT_ROOT = Path("checkpoints")


@cappa.command(name="rlexp26", default_long=True)
@dataclasses.dataclass(frozen=True)
class Run:
    """Train a recipe on a task, evaluating as it goes."""

    recipe: cappa.Subcommands[Recipe]
    task: cappa.Destructured[Task] = Task()  # noqa: RUF009 -- frozen

    # The run is <task>_<name>_<start time>; name defaults to the recipe's.
    name: Annotated[str, Arg(default=Env("RUN_NAME"))] = ""
    seed: Annotated[int, Arg(default=Env("SEED"))] = 0
    # Environments stepped in lockstep: the batch the agent acts on.
    num_envs: Annotated[int, Arg(default=Env("NUM_ENVS"))] = 64
    # The run stops at this many environment steps, over all environments.
    env_steps: Annotated[int, Arg(default=Env("ENV_STEPS"))] = 50_000_000
    # Gradient steps between policy handovers to the evaluator.
    eval_freq: Annotated[int, Arg(default=Env("EVAL_FREQ"))] = 3125
    # Steps of the evaluator's environments per loop iteration, at most.
    eval_steps_per_iteration: Annotated[
        int, Arg(default=Env("EVAL_STEPS_PER_ITERATION"))
    ] = 4
    # Gradient steps between checkpoints. The newest checkpoint_keep stay, and the first one
    # past every multiple of checkpoint_milestone gradient steps.
    checkpoint_freq: Annotated[int, Arg(default=Env("CHECKPOINT_FREQ"))] = 3125
    checkpoint_keep: Annotated[int, Arg(default=Env("CHECKPOINT_KEEP"))] = 3
    checkpoint_milestone: Annotated[int, Arg(default=Env("CHECKPOINT_MILESTONE"))] = (
        62_500
    )
    # START COUNT: trace COUNT gradient steps, from START after this process's first, into
    # the run's logs. This process's only: a resume takes its own.
    profile: tuple[int, int] | None = None


@cappa.command(name="rlexp26-resume")
@dataclasses.dataclass(frozen=True)
class Resume:
    """Carry on a run from its newest checkpoint, under the configuration it started with."""

    # The run's name, its directory under logs/ and checkpoints/.
    run: str
    # A new budget in environment steps; the run's own by default.
    env_steps: Annotated[int | None, Arg(long=True)] = None
    # As rlexp26's --profile.
    profile: Annotated[tuple[int, int] | None, Arg(long=True)] = None


# config.json is {"git": <git describe at the start>, "run": <the Run, but its profile>}, the
# budget kept current by resumes. Unknown keys are refused, so a knob the code has since
# dropped stops a resume instead of vanishing from it.
converter = cattrs.Converter(forbid_extra_keys=True)
converter.register_unstructure_hook(Fraction, str)
converter.register_structure_hook(Fraction, lambda value, _: Fraction(value))
configure_tagged_union(
    Recipe,
    converter,
    tag_generator=lambda recipe: cappa.Command.get(recipe).real_name(),
    tag_name="name",
)
converter.register_unstructure_hook(
    Run, make_dict_unstructure_fn(Run, converter, profile=override(omit=True))
)


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True))
    run = cappa.parse(Run)
    recipe_name = cappa.Command.get(type(run.recipe)).real_name()
    started = dt.datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

    # Only now: JAX reads its settings from the environment at import.
    from .train import train

    train(
        run, "_".join((run.task.name, run.name or recipe_name, started)), resumed=False
    )


def resume() -> None:
    load_dotenv(find_dotenv(usecwd=True))
    args = cappa.parse(Resume)
    config_path = LOG_ROOT / args.run / "config.json"
    if not config_path.exists():
        raise SystemExit(f"no run {args.run!r}: {config_path} does not exist")
    config = json.loads(config_path.read_text())
    try:
        run: Run = converter.structure(config["run"], Run)
    except cattrs.ClassValidationError as error:
        raise SystemExit(
            "\n".join(
                [
                    f"{config_path} no longer fits the code:",
                    *cattrs.transform_error(error),
                ]
            )
        ) from error
    if args.env_steps is not None:
        run = dataclasses.replace(run, env_steps=args.env_steps)
        config["run"] = converter.unstructure(run)
        config_path.write_text(json.dumps(config, indent=2))
    run = dataclasses.replace(run, profile=args.profile)

    # Only now: JAX reads its settings from the environment at import.
    from .train import train

    train(run, args.run, resumed=True)
