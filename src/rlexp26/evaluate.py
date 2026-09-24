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

from . import ale
from .agent import Policy

EVAL_NUM_ENVS = 8  # also the batch eval/returns_mean is taken over


class _Snapshot(NamedTuple):
    """Weights waiting to be picked up, with the step count they were taken at."""

    weights: Mapping[str, Any]
    num_env_steps: int


class EvalResult(NamedTuple):
    """One finished evaluation episode. The loop drains every iteration, so it logs it at its own step count."""

    episode_return: float
    seconds: float


class Evaluator:
    """
    Unclipped evaluation of an agent's weights under ``ale.EVAL_PROTOCOL``, on a background
    thread. ALE's vectoriser and JAX dispatch both drop the GIL, so it overlaps collection.

    The fleet plays episode after episode without a common reset, posting each as it ends;
    ``submit`` hands it newer weights, which it picks up on its next action. So an episode may
    span several snapshots, and it scores the policy over that episode.
    """

    def __init__(self, make_policy: Callable[[], Policy], seed: int) -> None:
        self._make_policy = make_policy
        self._seed = seed
        self._env: ale.AtariVecEnv | None = None
        self._policy: Policy | None = None
        self._results: queue.Queue[EvalResult] = queue.Queue()
        self._pending: _Snapshot | None = None
        self._lock = threading.Lock()
        self._num_env_steps = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def submit(self, weights: Mapping[str, Any], num_env_steps: int) -> None:
        """
        Hand the fleet an ``Agent.policy_weights`` snapshot, starting it if it is not running.
        Only the newest pending snapshot is kept.
        """
        with self._lock:
            self._pending = _Snapshot(weights, num_env_steps)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run, name="evaluator", daemon=True
            )
            self._thread.start()

    def drain(self) -> Generator[EvalResult]:
        """Yield whatever has finished since the last call, without blocking."""
        while True:
            try:
                yield self._results.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        """
        Stop the fleet and close its env. The join has to come first: closing ALE under a step
        in progress kills the process.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._env is not None:
            self._env.close()
            self._env = None

    def _run(self) -> None:
        """
        Thread body: play until stopped, posting every episode that ends. The env and policy are
        built on this thread and kept across restarts.
        """
        try:
            if self._env is None:
                self._env = gym.make_vec(
                    ale.ENV_ID, num_envs=EVAL_NUM_ENVS, **ale.EVAL_PROTOCOL
                )
            if self._policy is None:
                self._policy = self._make_policy()
            env, policy = self._env, self._policy

            obs, _info = env.reset(seed=self._seed)
            returns = np.zeros(env.num_envs)
            started = np.full(env.num_envs, time.perf_counter())

            while not self._stop.is_set():
                # submit parks a snapshot before starting the thread, so this loads before the
                # first act.
                with self._lock:
                    pending, self._pending = self._pending, None
                if pending is not None:
                    policy.load(pending.weights, seed=self._seed)
                    self._num_env_steps = pending.num_env_steps

                actions, _ = policy.act(
                    obs, num_env_steps=self._num_env_steps, evaluation=True
                )
                obs, rewards, terminated, truncated, _info = env.step(
                    jax.device_get(actions)
                )

                # Safe under next-step autoreset: the extra step a termination forces carries a
                # reward of 0, so it adds nothing to the episode that is about to start.
                returns += rewards
                now = time.perf_counter()
                for i in np.flatnonzero(np.logical_or(terminated, truncated)):
                    self._results.put(
                        EvalResult(
                            episode_return=float(returns[i]),
                            seconds=now - started[i],
                        )
                    )
                    returns[i] = 0.0
                    started[i] = now
        except Exception:  # noqa: BLE001 -- a thread body has nowhere to raise
            # The next submit starts the fleet again.
            print("EVAL FAILED")
            traceback.print_exc()
