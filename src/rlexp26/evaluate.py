"""Evaluation of an agent's policy snapshots, played beside the training loop."""

import queue
import threading
import time
import traceback
from collections.abc import Callable, Generator
from contextlib import closing
from typing import Any, NamedTuple

import jax
import jax.random as jr
import numpy as np
from gymnasium.vector import VectorEnv

from .agent import Policy


class EvalResult(NamedTuple):
    episode_return: float
    seconds: float


class Evaluator:
    """
    A fleet of environments playing the newest ``submit``ted policy on a background thread.
    The env's vectoriser and JAX dispatch both drop the GIL, so it overlaps training.

    The fleet plays episode after episode without a common reset and posts each as it ends.
    A newer policy takes over at the next step, on the state the older one left, so an
    episode may span several snapshots and scores the policy over that episode.

    The fleet takes at most ``steps_per_iteration`` steps per ``advance``, which the training
    loop calls once an iteration: its acts share the device's queue and the GIL with training.
    """

    def __init__(
        self,
        make_env: Callable[[], VectorEnv[Any, Any, Any]],
        *,
        seed: int,
        steps_per_iteration: int,
    ) -> None:
        self._make_env = make_env
        self._seed = seed
        self._steps_per_iteration = steps_per_iteration
        self._policy: Policy | None = None
        self._steps_allowed = 0
        self._stop = False
        self._changed = threading.Condition()
        self._results = queue.SimpleQueue[EvalResult]()
        self._thread: threading.Thread | None = None

    def submit(self, policy: Policy) -> None:
        """Hand the fleet a newer policy, starting the fleet if it is not running."""
        with self._changed:
            self._policy = policy
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run, name="evaluator", daemon=True
            )
            self._thread.start()

    def advance(self) -> None:
        with self._changed:
            self._steps_allowed = self._steps_per_iteration
            self._changed.notify()

    def drain(self) -> Generator[EvalResult]:
        """Whatever has finished since the last call, without blocking."""
        while not self._results.empty():
            yield self._results.get()

    def close(self) -> None:
        """Stop the fleet. Its env closes on its own thread, after its last step."""
        with self._changed:
            self._stop = True
            self._changed.notify()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        try:
            with closing(self._make_env()) as env:
                obs, _info = env.reset(seed=self._seed)
                state = None
                episode_start = np.ones(env.num_envs, bool)
                autoreset = np.zeros(env.num_envs, bool)
                returns = np.zeros(env.num_envs)
                started = np.full(env.num_envs, time.perf_counter())
                while True:
                    with self._changed:
                        self._changed.wait_for(
                            lambda: self._steps_allowed > 0 or self._stop
                        )
                        if self._stop:
                            return
                        self._steps_allowed -= 1
                        policy = self._policy
                    assert policy is not None
                    if state is None:
                        state = policy.initial_state(
                            env.num_envs, key=jr.key(self._seed)
                        )
                    actions, state = policy.act(obs, state, episode_start=episode_start)
                    obs, rewards, terminated, truncated, _info = env.step(
                        jax.device_get(actions)
                    )
                    episode_start, autoreset = autoreset, terminated | truncated

                    # The autoreset step's reward is 0, so it adds nothing to the next episode.
                    returns += rewards
                    now = time.perf_counter()
                    for i in np.flatnonzero(autoreset):
                        self._results.put(
                            EvalResult(float(returns[i]), float(now - started[i]))
                        )
                        returns[i] = 0.0
                        started[i] = now
        except Exception:  # noqa: BLE001 -- a thread body has nowhere to raise
            # The next submit starts the fleet again.
            print("EVAL FAILED")
            traceback.print_exc()
