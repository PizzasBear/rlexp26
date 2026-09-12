"""The training run: the single interleaved actor/learner loop and everything it checkpoints."""

import datetime as dt
import signal
import time
from argparse import ArgumentParser
from collections import deque
from collections.abc import Callable, Generator
from contextlib import ExitStack, closing, contextmanager
from types import FrameType
from typing import Any, NamedTuple

import gymnasium as gym
import jax
import numpy as np
from etils.epath import Path
from gymnasium.spaces import Box, MultiDiscrete
from orbax.checkpoint import v1 as ocp
from tensorboardX import SummaryWriter

from . import ale, btr
from .agent import Agent, EnvStep
from .evaluate import EvalResult, Evaluator

SEED = 0
EVAL_SEED = SEED + 1  # fixed, so every evaluation replays the same no-op starts
NUM_ENVS = 64  # environments stepped in lockstep, and the loop's batch of actions
LOG_FREQ = 25  # env-loop iterations between rate/throughput writes
TRAIN_LOG_FREQ = 25  # gradient steps between train_step diagnostic writes
CHECKPOINT_FREQ = 500  # gradient steps between checkpoints
CHECKPOINT_KEEP = 3  # checkpoints kept on disk; the rest are garbage collected
EVAL_FREQ = 3125  # gradient steps between evaluation runs
STATS_QUEUE_LEN = 3


@contextmanager
def interruptible() -> Generator[Callable[[], bool]]:
    """
    Turn Ctrl-C into a flag the training loop reads between iterations, so that the shutdown
    lands where the device queue is drained and nothing is half applied. A raised
    ``KeyboardInterrupt`` would not even arrive as itself: JAX catches it while hashing a jitted
    call's pytree metadata and re-raises it as a ``ValueError`` about unhashable fields.

    A second Ctrl-C hits the restored default handler and ends the loop body, but is caught here
    rather than re-raised, so the caller's teardown still runs.
    """
    interrupted = False
    previous = signal.getsignal(signal.SIGINT)

    def on_sigint(_signum: int, _frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True
        signal.signal(signal.SIGINT, previous)
        print(
            "\nINTERRUPTED -- finishing the iteration and checkpointing (^C again to abort)"
        )

    signal.signal(signal.SIGINT, on_sigint)
    try:
        yield lambda: interrupted
    except KeyboardInterrupt:
        print("\nABORTED")
    finally:
        signal.signal(signal.SIGINT, previous)


class PendingStats(NamedTuple):
    """
    One gradient step's diagnostics, still in flight on the device. Draining them on the *next*
    iteration is what lets the step overlap ``env.step``; the counter travels with them so they
    land on the curve where they were computed rather than where they were read.
    """

    values: dict[str, jax.Array]
    env_steps: int


def write_hparams(
    writer: SummaryWriter,
    hyperparameters: dict[str, bool | int | float | str],
    metrics: dict[str, float],
    global_step: int,
) -> None:
    """
    Record the run's hyperparameters and its closing metrics, so TensorBoard's HParams tab can
    line runs up against each other rather than leaving the constants that produced a curve to
    memory.

    The agent names its own hyperparameters and the env protocol goes in wholesale; everything
    else is named explicitly here, since the rest of the run's constants are logging and
    checkpointing knobs that do not change what a run means.

    ``add_hparams`` writes into a sub-directory of the writer's logdir, which is why this is
    called once at shutdown with real values rather than at startup with placeholders: the
    hparams plugin reads a session's metrics from that session's own run and the runs below it,
    never from the parent, so the numbers in the table are the ones passed here and nothing
    else. The scalars above go on living in the run proper; this is the summary row.
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

    # Fixed name, not add_hparams' default timestamp, so a resumed run rewrites the same session
    # rather than opening a second one beside it.
    writer.add_hparams(params, metrics, name="hparams", global_step=global_step)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="carry on from the newest checkpoint in this run directory, ./checkpoints/<run>",
    )
    args = parser.parse_args()
    resume: Path | None = args.resume

    with ExitStack() as stack:
        # closing() rather than enter_context: gymnasium gives Env a context manager but not
        # VectorEnv. Registered in the expression that builds it, so nothing below can leave
        # ALE's threads running.
        env: ale.AtariVecEnv = stack.enter_context(
            closing(gym.make_vec(ale.ENV_ID, num_envs=NUM_ENVS, **ale.PROTOCOL))
        )

        assert isinstance(env.action_space, MultiDiscrete)
        assert isinstance(env.single_observation_space, Box)
        assert env.single_observation_space.dtype is not None

        num_actions: int = env.action_space.nvec[0]
        assert (env.action_space.nvec == num_actions).all()

        obs_stack: int = env.single_observation_space.shape[0]
        # The one place the algorithm is named. Everything below asks the agent what it needs
        # rather than reaching into it.
        agent: Agent = btr.BTR(
            num_actions,
            obs_stack,
            frames_per_step=ale.FRAMES_PER_STEP,
            seed=SEED,
        )

        # A resumed run writes back into the directories the original one used, so that its
        # checkpoint steps stay one increasing sequence -- orbax refuses to write a step twice --
        # and TensorBoard shows one curve rather than two overlapping ones.
        if resume is None:
            now_str = dt.datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = Path(f"./checkpoints/{ale.ENV_NAME}_{now_str}").absolute()
        else:
            run_dir = resume.absolute()
        run_name = run_dir.name

        # Step-numbered directories rather than one path overwritten in place: each is written to
        # a tmp directory and renamed on completion, so an interrupt cannot destroy what is
        # already on disk the way an overwrite-in-place would.
        #
        # The pyright ignores are upstream's annotations, not these arguments: orbax types both
        # parameters against its v1 protocols, which its own v0 policy classes do not nominally
        # satisfy.
        save_policy = ocp.training.save_decision_policies.FixedIntervalPolicy(
            CHECKPOINT_FREQ
        )
        keep_policy = ocp.training.preservation_policies.LatestN(CHECKPOINT_KEEP)
        ckptr: ocp.training.Checkpointer = stack.enter_context(
            ocp.training.Checkpointer(
                run_dir,
                save_decision_policy=save_policy,  # pyright: ignore[reportArgumentType]
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

        writer = stack.enter_context(SummaryWriter(f"./logs/{run_name}"))
        evaluator = stack.enter_context(
            closing(Evaluator(agent.make_policy, EVAL_SEED))
        )

        # Seeded here rather than in ale.PROTOCOL: ale_py 0.12 dropped AtariVectorEnv's seed
        # argument, and gym.make_vec forwards the protocol to that constructor verbatim.
        obs, _info = env.reset(seed=SEED)
        # After the restore, so that an agent sizing or seeding its memory from the checkpoint
        # has it.
        agent.init(obs)

        actions, extras = agent.act(obs, num_env_steps=num_env_steps)
        actions, extras = jax.device_get((actions, extras))

        # Measured 2026-09-06, 3080 + 24-thread host, 64 envs, batch 256:
        #   pre-train-start: ~1000 env steps/s
        #   steady state:    ~910 env steps/s (~3.6k ALE frames/s), i.e. ~15h for 50M env steps
        #   GPU 80% util, 324W of a 370W cap, 84C -- power/thermally limited, so it is the wall
        #   host CPU ~1.4 cores of 24 (main thread ~40%, 16 ALE threads ~5% each): not the wall

        stats_queue = deque[PendingStats]()
        num_updates = agent.num_updates
        # Windows rather than modulo tests on num_updates, which an agent that learns in batches
        # steps over rather than landing on.
        last_train_log_updates = last_eval_updates = num_updates
        log_every_env_steps = LOG_FREQ * env.num_envs
        # Not checkpointed: the returns run into episodes a resumed run abandons at the env.reset
        # above, and the rate is measured from wherever this process started.
        curr_returns = np.zeros(env.num_envs)
        last_log_time, last_log_env_steps = time.perf_counter(), num_env_steps
        # Whatever agent.act asks to have logged about the actions, summed over the window
        # rather than sampled on the logging iteration: the transfer rides a sync the loop
        # already pays, so averaging every iteration is free.
        act_sums: dict[str, float] = {}
        act_count = 0
        # The closing row of the HParams table. last_eval is the only reason an EvalResult
        # outlives its drain; both stay None if the run ends before producing one.
        last_eval: EvalResult | None = None
        last_training_returns: float | None = None

        def drain_stats(keep: int) -> None:
            """
            Write the finished gradient steps' diagnostics, leaving ``keep`` still in flight.

            The loop leaves STATS_QUEUE_LEN - 1 outstanding so that a transfer never waits on the
            device; a shutdown leaves none, or the last steps before a checkpoint would be the
            ones missing from the curve.
            """
            while keep < len(stats_queue):
                stats = stats_queue.popleft()
                for name, value in jax.device_get(stats.values).items():
                    writer.add_scalar(name, float(value), global_step=stats.env_steps)

        def checkpointables() -> dict[str, Any]:
            """
            Everything a resumed run cannot rebuild for itself, read at the moment of the call.

            The agent's half is aliased rather than cloned, unlike ``Agent.policy_weights``:
            ``save_checkpointables_async`` walks the pytree on the calling thread and defers only
            the copy, so a later gradient step cannot change what is written. Verified by saving
            async, mutating immediately, and reading back the old values.
            """
            return agent.checkpointables() | {"num_env_steps": num_env_steps}

        def checkpoint_final() -> None:
            """
            The checkpoint that closes the run out, written past the save decision policy, since
            the newest on disk is up to CHECKPOINT_FREQ gradient steps old when the loop stops.

            Guarded because orbax refuses to write a step twice, which a run stopped before its
            first update would do. An interrupt mid-save costs only this checkpoint.
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
            """
            Close the run out in the HParams tab. Registered on the stack rather than run after
            the loop so that an interrupt -- which is how most runs end -- still lands the row,
            and registered before checkpoint_final so LIFO puts it after that write and while the
            writer above is still open.
            """
            metrics: dict[str, float] = {}
            if last_eval is not None:
                metrics["eval/returns"] = float(last_eval.returns.mean())
                metrics["eval/returns_max"] = float(last_eval.returns.max())
            if last_training_returns is not None:
                # Recorded rather than compared: it is the clipped mean over whichever envs
                # finished in the final window, so it says what the run was doing at the end,
                # not how well.
                metrics["run/training_returns"] = last_training_returns
            write_hparams(writer, agent.hyperparameters, metrics, num_env_steps)

        stack.callback(summarise_run)
        # Between the two on the unwind: the last scalars reach the curve before the closing
        # row is written, and after the checkpoint, which is the write worth losing least.
        stack.callback(drain_stats, 0)
        stack.callback(checkpoint_final)
        interrupted = stack.enter_context(interruptible())

        evaluator.submit(agent.policy_weights(), num_updates, num_env_steps)

        while not interrupted():
            num_env_steps += env.num_envs
            next_obs, rewards, terminated, truncated, _info = env.step(actions)

            next_actions, next_extras = agent.act(next_obs, num_env_steps=num_env_steps)
            jax.copy_to_host_async((next_actions, next_extras))  # type: ignore[no-untyped-call]

            curr_returns += rewards
            done = np.logical_or(terminated, truncated)
            if done.any():
                last_training_returns = float(curr_returns[done].mean())
                writer.add_scalar(
                    "run/training_returns",
                    last_training_returns,
                    global_step=num_env_steps,
                )
                curr_returns[done] = 0.0

            # Drained here, and only here until shutdown: the GPU has had the whole of
            # env.step to finish the step that produced them.
            drain_stats(STATS_QUEUE_LEN - 1)

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

            for result in evaluator.drain():
                last_eval = result
                writer.add_scalar(
                    "eval/returns",
                    result.returns.mean(),
                    global_step=result.num_env_steps,
                )
                writer.add_scalar(
                    "eval/returns_max",
                    result.returns.max(),
                    global_step=result.num_env_steps,
                )
                print(
                    f"EVAL {result.num_updates} gradient steps: "
                    f"{result.returns.mean():.1f} mean over {result.returns.size} episodes "
                    f"(min {result.returns.min():.0f}, max {result.returns.max():.0f}, "
                    f"{result.seconds:.0f}s)"
                )

            if num_env_steps - last_log_env_steps >= log_every_env_steps:
                now = time.perf_counter()
                fps = (num_env_steps - last_log_env_steps) / max(
                    now - last_log_time, 1e-9
                )
                last_log_time, last_log_env_steps = now, num_env_steps

                writer.add_scalar(
                    "run/env_steps_per_second", fps, global_step=num_env_steps
                )
                # The actions actually taken, so these are properties of the behaviour policy
                # rather than of the learned q. They trail by one iteration, the last act() of
                # the window still being in flight until the device_get at the foot of the loop.
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

            # Zero gradient steps while the agent is still collecting, which is also why every
            # frequency below is inside this branch: they are all counted in gradient steps.
            grad_steps, learn_stats = agent.learn_step()
            if grad_steps:
                num_updates = agent.num_updates

                # Only the logging iterations pay a transfer for the scalars.
                if num_updates - last_train_log_updates >= TRAIN_LOG_FREQ:
                    last_train_log_updates = num_updates
                    jax.copy_to_host_async(learn_stats)  # type: ignore[no-untyped-call]
                    stats_queue.append(
                        PendingStats(values=learn_stats, env_steps=num_env_steps)
                    )

                # An evaluation runs until its slowest env finishes, bounded only by ALE's
                # 108k-frame truncation, so it really can outlast EVAL_FREQ gradient steps.
                if num_updates - last_eval_updates >= EVAL_FREQ:
                    last_eval_updates = num_updates
                    if not evaluator.submit(
                        agent.policy_weights(), num_updates, num_env_steps
                    ):
                        print(
                            f"EVAL SKIPPED at {num_updates}: the previous one is still running"
                        )

                # should_save also answers False while a previous save is still writing, which is
                # right: dropping a checkpoint costs nothing, blocking on the disk costs a step.
                if ckptr.should_save(num_updates):
                    ckptr.save_checkpointables_async(num_updates, checkpointables())

            obs = next_obs
            actions, extras = jax.device_get((next_actions, next_extras))
            for name, value in extras.items():
                act_sums[name] = act_sums.get(name, 0.0) + float(value.mean())
            act_count += 1
