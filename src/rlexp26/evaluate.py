"""Unclipped evaluation of the training network, run beside the training loop."""

import queue
import threading
import time
import traceback
from collections.abc import Callable, Generator, Mapping
from typing import Any, NamedTuple

import gymnasium as gym
import jax
import numpy as np
import numpy.typing as npt

from . import ale
from .agent import Policy

EVAL_NUM_ENVS = 8  # also the episodes scored: one per env, see Evaluator._play


class EvalResult(NamedTuple):
    """
    One finished evaluation run, handed back to the training loop to log. The counters travel
    with it because it is drained an unknown number of iterations after it was submitted, and
    logging it against the counters at drain time would smear the curve right.
    """

    num_updates: int
    num_env_steps: int
    returns: npt.NDArray[np.float64]
    seconds: float


class Evaluator:
    """
    Unclipped evaluation of an agent's weights, played on a background thread.

    ``run/training_returns`` is a *clipped* return, so Breakout's 4- and 7-point bricks all count
    as 1; this replays the same weights under ``ale.EVAL_PROTOCOL`` to recover the game score.
    The policy is the training one, noise and all, asked for its evaluation behaviour.

    Threaded because an evaluation is thousands of sequential env steps, and ALE's vectoriser and
    the JAX dispatch both drop the GIL, so it really does overlap collection. The env and the
    policy are this object's own; only weights in and an ``EvalResult`` out cross the boundary.
    """

    def __init__(self, make_policy: Callable[[], Policy], seed: int) -> None:
        self._make_policy = make_policy
        self._seed = seed
        self._env: ale.AtariVecEnv | None = None
        self._policy: Policy | None = None
        self._results: queue.Queue[EvalResult] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def submit(
        self, weights: Mapping[str, Any], num_updates: int, num_env_steps: int
    ) -> bool:
        """
        Start an evaluation of an ``Agent.policy_weights`` snapshot; False if one is still
        running.
        """
        if self._thread is not None and self._thread.is_alive():
            return False

        self._thread = threading.Thread(
            target=self._run,
            args=(weights, num_updates, num_env_steps),
            name="evaluator",
            daemon=True,
        )
        self._thread.start()
        return True

    def drain(self) -> Generator[EvalResult]:
        """Yield whatever has finished since the last call, without blocking."""
        while True:
            try:
                yield self._results.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        """
        Stop an in-flight run and tear the thread's env down, from the training loop's thread.

        Joining before ``env.close()`` is not optional: closing ALE underneath a step in progress
        takes the process with it.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._env is not None:
            self._env.close()
            self._env = None

    def _run(
        self, weights: Mapping[str, Any], num_updates: int, num_env_steps: int
    ) -> None:
        """Thread body: play, then post the result. Nothing may escape to kill the thread."""
        started = time.perf_counter()
        try:
            returns = self._play(weights, num_env_steps)
        except Exception:  # noqa: BLE001 -- a thread body has nowhere to raise
            # Dropped so that a transient fault costs one point on the curve, not every
            # evaluation after it: the next submit tries again.
            print("EVAL FAILED")
            traceback.print_exc()
            return

        if returns.size:
            self._results.put(
                EvalResult(
                    num_updates=num_updates,
                    num_env_steps=num_env_steps,
                    returns=returns,
                    seconds=time.perf_counter() - started,
                )
            )

    def _play(
        self, weights: Mapping[str, Any], num_env_steps: int
    ) -> npt.NDArray[np.float64]:
        """
        Score one episode per env and return their unclipped returns.

        Counted by first finish rather than as a running total of N episodes, which would score
        the short ones twice: envs that die early get autoreset and would contribute again while
        the long ones are still going. Scored envs keep playing, since masking them out would
        cost a second act() shape and a retrace.

        Env and policy are built once and kept, both on this thread. The load reseeds the
        policy's own randomness, so successive evaluations replay the same draws rather than
        compounding sampling noise.
        """
        if self._env is None:
            self._env = gym.make_vec(
                ale.ENV_ID, num_envs=EVAL_NUM_ENVS, **ale.EVAL_PROTOCOL
            )
        if self._policy is None:
            self._policy = self._make_policy()
        env, policy = self._env, self._policy
        policy.load(weights, seed=self._seed)

        obs, _info = env.reset(seed=self._seed)
        curr_returns = np.zeros(env.num_envs)
        returns = np.zeros(env.num_envs)
        scored = np.zeros(env.num_envs, dtype=np.bool_)

        while not scored.all() and not self._stop.is_set():
            actions, _ = policy.act(obs, num_env_steps=num_env_steps, evaluation=True)
            actions = jax.device_get(actions)
            obs, rewards, terminated, truncated, _info = env.step(actions)

            # Safe under next-step autoreset: the extra step a termination forces carries a
            # reward of 0, so it adds nothing to the episode that is about to start.
            curr_returns += rewards
            done = np.logical_or(terminated, truncated) & ~scored
            if done.any():
                returns[done] = curr_returns[done]
                scored |= done
            curr_returns[np.logical_or(terminated, truncated)] = 0.0

        return returns[scored].astype(np.float64)
