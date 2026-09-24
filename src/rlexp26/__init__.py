"""The training run: the single interleaved actor/learner loop and everything it checkpoints."""

import datetime as dt
import signal
import time
from argparse import ArgumentParser
from collections import deque
from collections.abc import Callable, Generator, Sequence
from contextlib import ExitStack, closing, contextmanager
from types import FrameType
from typing import Any, NamedTuple, NewType

import gymnasium as gym
import jax
import numpy as np
from etils.epath import Path
from gymnasium.spaces import Box, MultiDiscrete
from orbax.checkpoint import v1 as ocp
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams

from . import ale, btr
from .agent import Agent, EnvStep
from .evaluate import EVAL_NUM_ENVS, EvalResult, Evaluator

# tensorboardX.summary.Summary comes from a generated protobuf module pyright cannot see into.
Summary = NewType("Summary", object)

SEED = 0
EVAL_SEED = SEED + 1
NUM_ENVS = 64  # environments stepped in lockstep, and the loop's batch of actions
LOG_FREQ = 100  # env-loop iterations between rate/throughput writes
CHECKPOINT_FREQ = 3125  # gradient steps between checkpoints
CHECKPOINT_KEEP = 3  # checkpoints kept on disk; the rest are garbage collected
EVAL_FREQ = 3125  # gradient steps between weight handovers to the evaluator
STATS_DRAIN_LAG = 2  # env-loop iterations a gradient step's diagnostics stay in flight


@contextmanager
def interruptible() -> Generator[Callable[[], bool]]:
    """
    Turn Ctrl-C into a flag the training loop reads between iterations, so the shutdown lands
    with nothing half applied. A raw ``KeyboardInterrupt`` inside a jitted call surfaces as a
    ``ValueError`` from JAX. A second Ctrl-C goes to the restored default handler and propagates.
    """
    interrupted = False
    previous = signal.getsignal(signal.SIGINT)

    def on_sigint(_signum: int, _frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True
        signal.signal(signal.SIGINT, previous)
        print()
        print(
            "INTERRUPTED: finishing the iteration and checkpointing (^C again to abort)"
        )

    signal.signal(signal.SIGINT, on_sigint)
    try:
        yield lambda: interrupted
    except KeyboardInterrupt:
        print()
        print("ABORTED")
        raise
    finally:
        signal.signal(signal.SIGINT, previous)


class SinceLastSavePolicy:
    """
    Orbax save decision policy: save once ``interval`` gradient steps have passed since the
    newest checkpoint, or at the first step if there is none, and never while a save is still
    writing. ``FixedIntervalPolicy``'s ``step % interval`` would be stepped over by an agent
    taking several gradient steps an iteration.
    """

    def __init__(self, interval: int) -> None:
        self.interval = interval

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


class PendingStats(NamedTuple):
    """
    One gradient step's diagnostics, still on the device, with the env step they were computed
    at: where they land on the curve, and what the drain measures their age by.
    """

    values: dict[str, jax.Array]
    env_steps: int


# Scalar tags the HParams tab shows beside the constants; they must be series the run writes.
HPARAM_METRICS = (
    "eval/returns",
    "eval/returns_mean",
    "eval/returns_max",
    "run/training_returns",
    "run/env_steps_per_second",
)


def write_hparams(
    writer: SummaryWriter, hyperparameters: dict[str, bool | int | float | str]
) -> Summary:
    """
    Write the HParams experiment and session start into the run's own event file, and return
    the session end for the caller to write at shutdown. Unlike ``add_hparams``, which writes a
    separate run in a sub-directory, this survives a run that dies, and the metrics are read by
    tag from the run's own scalars. A resume writes a second session start; the plugin reads the
    later one.
    """
    params: dict[str, bool | str | float | int] = dict(hyperparameters)
    params |= {
        f"env/{name}": value
        for name, value in ale.PROTOCOL.items()
        if isinstance(value, bool | int | float | str)
    }
    params["env/id"] = ale.ENV_ID
    params["run/seed"] = SEED
    params["run/num_envs"] = NUM_ENVS

    # ``hparams`` reads only the keys of the metrics.
    experiment, session_start, session_end = hparams(
        params, dict.fromkeys(HPARAM_METRICS, 0.0)
    )
    assert writer.file_writer is not None
    writer.file_writer.add_summary(experiment)
    writer.file_writer.add_summary(session_start)
    return session_end


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="carry on from the newest checkpoint in this run directory, ./checkpoints/<run>",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="START:COUNT",
        help="trace COUNT gradient steps, from START gradient steps into this run, to its logdir",
    )
    args = parser.parse_args()
    resume: Path | None = args.resume
    # In gradient steps taken by this process, so the window falls past the buffer fill.
    profile_start, profile_count = 0, 0
    if args.profile:
        start, _, count = args.profile.partition(":")
        if not (start.isdigit() and count.isdigit() and int(count)):
            parser.error("--profile takes START:COUNT, START >= 0 and COUNT >= 1")
        profile_start, profile_count = int(start), int(count)

    with ExitStack() as stack:
        # closing(): gymnasium's VectorEnv is not a context manager.
        env: ale.AtariVecEnv = stack.enter_context(
            closing(gym.make_vec(ale.ENV_ID, num_envs=NUM_ENVS, **ale.PROTOCOL))
        )

        assert isinstance(env.action_space, MultiDiscrete)
        assert isinstance(env.single_observation_space, Box)
        assert env.single_observation_space.dtype is not None

        num_actions: int = env.action_space.nvec[0]
        assert (env.action_space.nvec == num_actions).all()

        obs_stack, obs_height, obs_width = env.single_observation_space.shape
        # The one place the algorithm is named.
        agent: Agent = btr.BTR(
            num_actions,
            (obs_stack, obs_height, obs_width),
            frames_per_step=ale.FRAMES_PER_STEP,
            seed=SEED,
        )

        # A resumed run writes into the original run's directories, continuing one checkpoint
        # sequence and one curve.
        if resume is None:
            now_str = dt.datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = Path(f"./checkpoints/{ale.ENV_NAME}_{now_str}").absolute()
        else:
            run_dir = resume.absolute()
        run_name = run_dir.name

        # The pyright ignore is orbax's: its v0 policy classes do not nominally satisfy the v1
        # protocol the parameter is typed against.
        keep_policy = ocp.training.preservation_policies.LatestN(CHECKPOINT_KEEP)
        ckptr: ocp.training.Checkpointer = stack.enter_context(
            ocp.training.Checkpointer(
                run_dir,
                save_decision_policy=SinceLastSavePolicy(CHECKPOINT_FREQ),
                preservation_policy=keep_policy,  # pyright: ignore[reportArgumentType]
                cleanup_tmp_directories=True,
            )
        )

        num_env_steps = 0
        if resume is not None:
            if ckptr.latest is None:
                raise SystemExit(f"no checkpoint to resume from in {run_dir}")

            restored = ckptr.load_checkpointables(
                None, agent.checkpointables() | {"num_env_steps": num_env_steps}
            )
            agent.restore(restored)
            num_env_steps = restored["num_env_steps"]
            print(f"RESUMED {run_name} at {agent.num_updates} gradient steps")

        # Shared with the profiler, so TensorBoard shows the trace beside the scalars.
        logdir = f"./logs/{run_name}"
        writer = stack.enter_context(SummaryWriter(logdir))
        hparams_session_end = write_hparams(writer, agent.hyperparameters)
        evaluator = stack.enter_context(
            closing(Evaluator(agent.make_policy, EVAL_SEED))
        )

        # Seeded here, not in ale.PROTOCOL: ale_py 0.12's AtariVectorEnv takes no seed argument.
        obs, _info = env.reset(seed=SEED)
        # After the restore, which init may read.
        agent.init(obs)

        actions, extras = agent.act(obs, num_env_steps=num_env_steps)
        actions, extras = jax.device_get((actions, extras))

        stats_queue = deque[PendingStats]()
        num_updates = agent.num_updates
        last_eval_updates = num_updates
        eval_batch: list[EvalResult] = []
        log_every_env_steps = LOG_FREQ * env.num_envs
        curr_returns = np.zeros(env.num_envs)
        last_log_time, last_log_env_steps = time.perf_counter(), num_env_steps
        # agent.act's extras, averaged over the logging window.
        act_sums: dict[str, float] = {}
        act_count = 0

        def drain_stats(lag: int) -> None:
            """
            Write the diagnostics of every gradient step dispatched at least ``lag`` iterations
            ago. A step dispatched in iteration i is done by i + 2, once the act dispatched after
            it has been waited on.
            """
            while stats_queue and (
                lag * env.num_envs <= num_env_steps - stats_queue[0].env_steps
            ):
                stats = stats_queue.popleft()
                for name, value in jax.device_get(stats.values).items():
                    writer.add_scalar(name, float(value), global_step=stats.env_steps)

        def checkpointables() -> dict[str, Any]:
            """
            Everything a resumed run cannot rebuild, read at the call. Aliased, not cloned:
            ``save_checkpointables_async`` takes the arrays on the calling thread.
            """
            return agent.checkpointables() | {"num_env_steps": num_env_steps}

        def checkpoint_final() -> None:
            """
            The closing checkpoint, forced past the save decision policy. Skipped when the step
            is already on disk, since orbax refuses to write a step twice.
            """
            latest = ckptr.latest
            if num_updates == 0 or (latest is not None and num_updates <= latest.step):
                return

            try:
                ckptr.save_checkpointables(num_updates, checkpointables(), force=True)
                print(f"CHECKPOINTED {num_updates} gradient steps to {run_dir}")
            except KeyboardInterrupt:
                print("ABORTED -- keeping the newest complete checkpoint")

        def summarise_run() -> None:
            """Close the run's HParams session."""
            assert writer.file_writer is not None
            writer.file_writer.add_summary(hparams_session_end)

        stack.callback(summarise_run)
        # Unwound in reverse: checkpoint, then the last scalars, then the session end, all
        # while the writer is open.
        stack.callback(drain_stats, 0)
        stack.callback(checkpoint_final)

        profiling = False
        # Fixed at this process's first gradient step, so a resume's refill is not traced.
        profile_at: int | None = None

        def stop_profile() -> None:
            """
            Close the trace window, if one is open. Also on the stack, first to unwind, so an
            interrupted window is still written and the teardown stays outside it.
            """
            nonlocal profiling
            if not profiling:
                if profile_count:
                    print("PROFILE NOT TAKEN -- the run ended before the window opened")
                return
            jax.block_until_ready(checkpointables())  # type: ignore[no-untyped-call]
            jax.profiler.stop_trace()  # type: ignore[no-untyped-call]
            profiling = False
            print(f"PROFILED {profile_at}..{num_updates} into {logdir}")

        stack.callback(stop_profile)
        interrupted = stack.enter_context(interruptible())

        evaluator.submit(agent.policy_weights(), num_env_steps)

        while not interrupted():
            if profile_count and profile_at is not None:
                if not profiling and num_updates >= profile_at:
                    # Both ends of the window wait for the device to go idle.
                    # jax.effects_barrier would not: there are no ordered effects to wait on.
                    jax.block_until_ready(checkpointables())  # type: ignore[no-untyped-call]
                    jax.profiler.start_trace(str(logdir))
                    profiling = True
                elif profiling and num_updates >= profile_at + profile_count:
                    stop_profile()
                    profile_count = 0

            num_env_steps += env.num_envs
            with jax.profiler.TraceAnnotation("env.step"):
                next_obs, rewards, terminated, truncated, _info = env.step(actions)

            next_actions, next_extras = agent.act(next_obs, num_env_steps=num_env_steps)
            jax.copy_to_host_async((next_actions, next_extras))  # type: ignore[no-untyped-call]

            curr_returns += rewards
            done = np.logical_or(terminated, truncated)
            if done.any():
                writer.add_scalar(
                    "run/training_returns",
                    float(curr_returns[done].mean()),
                    global_step=num_env_steps,
                )
                curr_returns[done] = 0.0

            drain_stats(STATS_DRAIN_LAG)

            with jax.profiler.TraceAnnotation("agent.observe"):
                agent.observe(
                    EnvStep(
                        obs=obs,
                        actions=actions,
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        next_obs=next_obs,
                        extras=extras,
                    )
                )

            # Each episode lands where it ended.
            for result in evaluator.drain():
                writer.add_scalar(
                    "eval/returns", result.episode_return, global_step=num_env_steps
                )
                eval_batch.append(result)
                if len(eval_batch) == EVAL_NUM_ENVS:
                    batch = np.array([r.episode_return for r in eval_batch])
                    # Disjoint batches of EVAL_NUM_ENVS, the shape a snapshot evaluation had,
                    # so these stay comparable with runs from before the continuous fleet.
                    writer.add_scalar(
                        "eval/returns_mean", batch.mean(), global_step=num_env_steps
                    )
                    writer.add_scalar(
                        "eval/returns_max", batch.max(), global_step=num_env_steps
                    )
                    print(
                        f"EVAL {num_updates} gradient steps: {batch.mean():.1f} mean over "
                        f"{batch.size} episodes (min {batch.min():.0f}, "
                        f"max {batch.max():.0f}, "
                        f"{np.mean([r.seconds for r in eval_batch]):.0f}s each)"
                    )
                    eval_batch.clear()

            if num_env_steps - last_log_env_steps >= log_every_env_steps:
                now = time.perf_counter()
                fps = (num_env_steps - last_log_env_steps) / max(
                    now - last_log_time, 1e-9
                )
                last_log_time, last_log_env_steps = now, num_env_steps

                writer.add_scalar(
                    "run/env_steps_per_second", fps, global_step=num_env_steps
                )
                # About the actions taken, trailing by the one act still in flight.
                if act_count:
                    for name, total in act_sums.items():
                        writer.add_scalar(
                            f"run/action_{name}",
                            total / act_count,
                            global_step=num_env_steps,
                        )
                    act_sums, act_count = {}, 0

                for name, scalar in agent.stats().items():
                    writer.add_scalar(name, scalar, global_step=num_env_steps)

            with jax.profiler.TraceAnnotation("agent.learn_step"):
                grad_steps, learn_stats = agent.learn_step()
            if grad_steps:
                num_updates = agent.num_updates
                if profile_count and profile_at is None:
                    profile_at = num_updates + profile_start

                if learn_stats:
                    jax.copy_to_host_async(learn_stats)  # type: ignore[no-untyped-call]
                    stats_queue.append(
                        PendingStats(values=learn_stats, env_steps=num_env_steps)
                    )

                if num_updates - last_eval_updates >= EVAL_FREQ:
                    last_eval_updates = num_updates
                    evaluator.submit(agent.policy_weights(), num_env_steps)

                if ckptr.should_save(num_updates):
                    ckptr.save_checkpointables_async(num_updates, checkpointables())

            obs = next_obs
            with jax.profiler.TraceAnnotation("act transfer"):
                actions, extras = jax.device_get((next_actions, next_extras))
            for name, value in extras.items():
                act_sums[name] = act_sums.get(name, 0.0) + float(value.mean())
            act_count += 1
