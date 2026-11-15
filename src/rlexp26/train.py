"""A run: the single interleaved actor/learner loop, from a fresh start or a checkpoint."""

import json
import signal
import subprocess
import time
from collections.abc import Callable, Generator
from contextlib import ExitStack, closing, contextmanager
from functools import partial
from pathlib import Path
from types import FrameType

import jax
import numpy as np
from etils import epath
from orbax.checkpoint import v1 as ocp

from .agent import Agent, EnvStep
from .checkpoint import FirstPastEachMultiple, SinceLastSave
from .cli import CHECKPOINT_ROOT, LOG_ROOT, Run, converter
from .evaluate import EvalResult, Evaluator
from .recipes import make_agent
from .tracking import TensorBoardTracker, TraceWindow, Tracker

LOG_FREQ = 100  # loop iterations between the rate and agent.stats() writes
STATS_DRAIN_LAG = 2  # loop iterations a gradient step's scalars stay in flight
EVAL_NUM_ENVS = 8  # also the batch eval/returns_mean is taken over
# Scalars the HParams tab shows beside the hyperparameters.
HPARAM_METRICS = (
    "eval/returns",
    "eval/returns_mean",
    "eval/returns_max",
    "run/training_returns",
    "run/env_steps_per_second",
)


@contextmanager
def stop_requested(*signums: signal.Signals) -> Generator[Callable[[], bool]]:
    """
    Turn the first of these signals into a flag the loop reads between iterations, so the run
    stops with nothing half applied. The handlers are then restored, so a second signal
    aborts.
    """
    requested = False
    previous = {signum: signal.getsignal(signum) for signum in signums}

    def on_signal(signum: int, _frame: FrameType | None) -> None:
        nonlocal requested
        requested = True
        for restored, handler in previous.items():
            signal.signal(restored, handler)
        print(
            f"\n{signal.Signals(signum).name}: finishing the iteration and checkpointing "
            "(again to abort)"
        )

    for signum in signums:
        signal.signal(signum, on_signal)
    try:
        yield lambda: requested
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def train(run: Run, run_name: str, *, resumed: bool) -> None:
    """
    Train ``run`` as the run ``run_name``, from scratch or, ``resumed``, from its newest
    checkpoint. It stops at ``run.env_steps`` or at the first SIGINT or SIGTERM, and saves a
    final checkpoint either way. An exception saves nothing, leaving the newest periodic
    checkpoint as the state to resume from.
    """
    logdir = LOG_ROOT / run_name
    checkpoint_dir = epath.Path(CHECKPOINT_ROOT / run_name).absolute()
    git = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not resumed:
        logdir.mkdir(parents=True)
        (logdir / "config.json").write_text(
            json.dumps({"git": git, "run": converter.unstructure(run)}, indent=2)
        )

    with ExitStack() as stack:
        stop = stack.enter_context(stop_requested(signal.SIGINT, signal.SIGTERM))
        task = run.task
        env = stack.enter_context(
            closing(task.make_env(run.num_envs + run.heldout_envs, evaluation=False))
        )
        # The one place the algorithm is named.
        agent: Agent = make_agent(
            run.recipe,
            env.single_observation_space,
            env.single_action_space,
            frames_per_step=task.frames_per_step,
            seed=run.seed,
        )

        num_env_steps = 0
        if resumed:
            # A reader of its own: orbax's close does not wait for the tmp-directory cleanup
            # the writing checkpointer starts, so closing that at once fails at exit.
            with ocp.training.Checkpointer(checkpoint_dir) as reader:
                if reader.latest is None:
                    raise SystemExit(
                        f"no checkpoint to resume from in {checkpoint_dir}"
                    )
                restored = reader.load_checkpointables(
                    None, agent.checkpointables() | {"num_env_steps": num_env_steps}
                )
            agent.restore(restored)
            num_env_steps = int(restored["num_env_steps"])
            print(
                f"RESUMED {run_name} at {agent.num_updates} gradient steps, "
                f"{num_env_steps} env steps"
            )
        else:
            print(f"STARTED {run_name}")
        preservation = ocp.training.preservation_policies
        # orbax's v1 re-exports these two from v0, typed against v0's checkpoint info.
        kept = preservation.AnyPreservationPolicy(
            [
                preservation.LatestN(run.checkpoint_keep),
                FirstPastEachMultiple(run.checkpoint_milestone),  # type: ignore[list-item, unused-ignore]
            ]
        )
        checkpointer = stack.enter_context(
            ocp.training.Checkpointer(
                checkpoint_dir,
                save_decision_policy=SinceLastSave(run.checkpoint_freq),
                preservation_policy=kept,  # type: ignore[arg-type, unused-ignore]
                cleanup_tmp_directories=True,
            )
        )

        tracker: Tracker = stack.enter_context(
            closing(TensorBoardTracker(logdir, HPARAM_METRICS, num_env_steps))
        )
        tracker.log_hyperparameters(
            agent.hyperparameters
            | task.hyperparameters()
            | {
                "run/seed": run.seed,
                "run/num_envs": run.num_envs,
                "run/heldout_envs": run.heldout_envs,
                "run/git": git,
            }
        )
        # Seeded by where the run stands, so a resume plays new episodes, not the start's.
        # ALE's seeds are C ints, one per environment counting up from the one given.
        env_seed, eval_seed = (
            np.random.default_rng((run.seed, num_env_steps))
            .integers(2**30, size=2)
            .tolist()
        )
        evaluator = stack.enter_context(
            closing(
                Evaluator(
                    partial(task.make_env, EVAL_NUM_ENVS, evaluation=True),
                    seed=eval_seed,
                    steps_per_iteration=run.eval_steps_per_iteration,
                )
            )
        )
        trace = None if run.profile is None else TraceWindow(logdir, *run.profile)

        obs, _info = env.reset(seed=env_seed)
        agent.init(obs, num_heldout=run.heldout_envs)
        # Under next-step autoreset, the step after an episode's end opens the next one.
        episode_start = np.ones(len(obs), bool)
        autoreset = np.zeros(len(obs), bool)
        actions, extras = agent.act(
            obs, num_env_steps=num_env_steps, episode_start=episode_start
        )
        actions = jax.device_get(actions)
        evaluator.submit(agent.policy(num_env_steps=num_env_steps))
        last_eval_updates = agent.num_updates

        # A save's thread and event loop end only in its response's result(), which also
        # raises a failed save. should_save holds off while one is running, so the previous
        # save has finished by the next.
        saving: ocp.AsyncResponse[bool] | None = None
        training_returns = np.zeros(len(obs))
        eval_batch: list[EvalResult] = []
        last_log_time, last_log_env_steps = time.perf_counter(), num_env_steps

        try:
            while num_env_steps < run.env_steps and not stop():
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                num_env_steps += run.num_envs
                episode_start, autoreset = autoreset, terminated | truncated

                # Dispatched before anything waits on the device, and forced only after the
                # next env.step, so the device acts while the host steps.
                next_actions, next_extras = agent.act(
                    next_obs, num_env_steps=num_env_steps, episode_start=episode_start
                )
                jax.copy_to_host_async(next_actions)  # type: ignore[no-untyped-call]
                evaluator.advance()

                training_returns += rewards
                ended = task.episode_over(terminated, info) | truncated
                for name, envs in (
                    ("run/training_returns", slice(run.num_envs)),
                    ("heldout/returns", slice(run.num_envs, None)),
                ):
                    if ended[envs].any():
                        tracker.log(
                            {name: training_returns[envs][ended[envs]].mean()},
                            num_env_steps,
                        )
                training_returns[ended] = 0.0
                tracker.drain(num_env_steps - STATS_DRAIN_LAG * run.num_envs)

                agent.observe(
                    EnvStep(
                        actions, rewards, terminated, truncated, next_obs, info, extras
                    )
                )

                # Each episode lands where it ended.
                for result in evaluator.drain():
                    tracker.log({"eval/returns": result.episode_return}, num_env_steps)
                    eval_batch.append(result)
                    if len(eval_batch) == EVAL_NUM_ENVS:
                        batch = np.array([r.episode_return for r in eval_batch])
                        tracker.log(
                            {
                                "eval/returns_mean": batch.mean(),
                                "eval/returns_max": batch.max(),
                            },
                            num_env_steps,
                        )
                        print(
                            f"EVAL {agent.num_updates} gradient steps: "
                            f"{batch.mean():.1f} mean over {batch.size} episodes "
                            f"(min {batch.min():.0f}, max {batch.max():.0f}, "
                            f"{np.mean([r.seconds for r in eval_batch]):.0f}s each)"
                        )
                        eval_batch.clear()

                if num_env_steps - last_log_env_steps >= LOG_FREQ * run.num_envs:
                    now = time.perf_counter()
                    tracker.log(
                        {
                            "run/env_steps_per_second": (
                                num_env_steps - last_log_env_steps
                            )
                            / (now - last_log_time),
                            "run/num_updates": agent.num_updates,
                        }
                        | agent.stats(),
                        num_env_steps,
                    )
                    last_log_time, last_log_env_steps = now, num_env_steps

                num_steps, learn_stats = agent.learn_step()
                if num_steps:
                    if learn_stats:
                        tracker.log_later(learn_stats, num_env_steps)
                    if trace is not None:
                        trace.update(agent)
                    if agent.num_updates - last_eval_updates >= run.eval_freq:
                        evaluator.submit(agent.policy(num_env_steps=num_env_steps))
                        last_eval_updates = agent.num_updates
                    # Asked first, since gathering the checkpointables costs host time.
                    if checkpointer.should_save(agent.num_updates):
                        if saving is not None:
                            saving.result()
                        saving = checkpointer.save_checkpointables_async(
                            agent.num_updates,
                            agent.checkpointables() | {"num_env_steps": num_env_steps},
                        )

                actions, extras = jax.device_get(next_actions), next_extras
        finally:
            if trace is not None:
                trace.close(agent)

        if saving is not None:
            saving.result()
        # Unless no step was taken or the periodic save already has this one.
        latest = checkpointer.latest
        if agent.num_updates > (0 if latest is None else latest.step):
            checkpointer.save_checkpointables(
                agent.num_updates,
                agent.checkpointables() | {"num_env_steps": num_env_steps},
                force=True,
            )
            print(
                f"CHECKPOINTED {agent.num_updates} gradient steps to {checkpoint_dir}"
            )
        if num_env_steps >= run.env_steps:
            print(f"FINISHED {run_name} at {num_env_steps} env steps")
