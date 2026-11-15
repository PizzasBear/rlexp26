"""
A distributional Q-learner: n-step Munchausen targets from prioritised replay, a hard-copied
target net, and a soft or epsilon-greedy behaviour policy. The net, its optimiser, how its
output is read as return distributions and how the agent explores are handed in as parts;
``recipes`` assembles the ones this project runs.
"""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, NamedTuple, Protocol, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
import optax
import optax.tree_utils as otu
from jaxtyping import PyTree

from .agent import Agent, EnvStep, Policy
from .config import Common, Replay
from .modules import QNet
from .replay import Batch, Draw, ReplayMemory

# Gradient steps between the scalars learn_step returns.
TRAIN_LOG_FREQ = 100
# Gradient steps between _parameter_stats' scalars, a multiple of TRAIN_LOG_FREQ.
PARAMETER_LOG_FREQ = 1000
# Gradient steps between _score's scalars, a multiple of TRAIN_LOG_FREQ.
HELDOUT_LOG_FREQ = 1000
# Transitions per side of a _score, taken in batches of batch_size.
HELDOUT_SAMPLES = 1024
# The parts of the net the parameter scalars are taken over (QNet.parameter_groups).
PARTS = ("trunk", "head", "readout")


class ReturnDistribution(Protocol):
    """How a net's output is read as a return distribution per action, and fitted to a target."""

    def forward(
        self,
        net: Callable[..., tuple[jax.Array, eqx.nn.State]],
        obs: jax.Array,
        state: eqx.nn.State,
        *,
        key: jax.Array,
        taken: jax.Array | None = None,
    ) -> tuple[Any, eqx.nn.State]:
        """
        ``net``'s output on a batch, under one draw of ``key``, and its new state. ``taken`` is
        the actions a loss will read, where there is one.
        """
        ...

    def means(self, out: Any) -> jax.Array:
        """Each action's mean return, ``[batch, actions]``."""
        ...

    def loss(
        self,
        out: Any,
        actions: jax.Array,
        next_out: Any,
        next_policy: jax.Array,
        shift: jax.Array,
        scale: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        """
        Per transition: the loss of ``actions``' distributions in ``out`` against
        ``shift + scale * Z``, with Z the distributions in ``next_out`` mixed over the actions by
        ``next_policy``; the signed TD error; the magnitude a priority is made from; and the
        distribution's own diagnostics, keyed by the name they are logged under. Nothing is
        differentiated through the target.
        """
        ...


class Learner(NamedTuple):
    """
    What a gradient step reads and writes. ``params`` is a QNet's floating arrays, and
    ``target`` is the ``(params, state)`` the target net last copied.
    """

    params: PyTree[jax.Array]
    state: eqx.nn.State
    target: PyTree[jax.Array]
    opt_state: optax.OptState
    key: jax.Array
    num_updates: jax.Array


class QPolicy(Policy):
    """The argmax of a ``QLearner``'s action values at one moment, under ``epsilon``."""

    def __init__(
        self,
        act: Callable[..., tuple[jax.Array, jax.Array]],
        learner: tuple[jax.Array, ...],
        epsilon: float,
    ) -> None:
        self._act = act
        self._learner = learner
        self._epsilon = epsilon

    @override
    def initial_state(
        self, num_envs: int, *, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return key, jnp.zeros((), jnp.int32)

    @override
    def act(
        self,
        obs: npt.NDArray[Any],
        state: tuple[jax.Array, jax.Array],
        *,
        episode_start: npt.NDArray[np.bool_],
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        key, step = state
        actions, step = self._act(self._learner, obs, key, step, self._epsilon)
        return actions, (key, step)


class QLearner(Agent):
    """
    The parts:

    - ``net`` and ``state``, a QNet on stacked image observations, and ``optimiser`` for its
      floating arrays. The target and acting nets are ``net`` in inference mode.
    - ``returns`` reads the net in a gradient step and ``act_returns`` when acting, whose
      ``means`` need be right only up to a constant per sample.
    - ``temperature`` is the Munchausen one, and the behaviour policy's where ``soft``: actions
      are then drawn from softmax(Q / temperature), and otherwise are the argmax. Either is
      replaced by a uniform action with probability ``epsilon(num_env_steps)``.
      ``eval_epsilon`` is the same for the policy ``policy`` hands out, always an argmax.
    - ``augment(obs, key)`` is applied to a training batch's observations and next
      observations apart, and ``project(params)`` after every optimiser step.
    - ``hyperparameters`` is what the property of that name reports.

    ``_choose_actions`` and ``_train_step`` are traced once, so they may read only what
    ``__init__`` fixes.
    """

    def __init__(
        self,
        net: QNet,
        state: eqx.nn.State,
        optimiser: optax.GradientTransformation,
        returns: ReturnDistribution,
        act_returns: ReturnDistribution,
        *,
        common: Common,
        replay: Replay,
        temperature: float,
        soft: bool,
        epsilon: Callable[[int], float],
        eval_epsilon: Callable[[int], float],
        augment: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        project: Callable[[PyTree[jax.Array]], PyTree[jax.Array]] | None = None,
        hyperparameters: Mapping[str, bool | int | float | str],
        key: jax.Array,
    ) -> None:
        self._optimiser = optimiser
        self._returns = returns
        self._act_returns = act_returns
        self._common = common
        self._replay = replay
        self._temperature = temperature
        self._soft = soft
        self._epsilon = epsilon
        self._eval_epsilon = eval_epsilon
        self._augment = augment
        self._project = project
        self._hyperparameters = dict(hyperparameters)

        params, self._static = eqx.partition(net, eqx.is_inexact_array)
        # The target and acting nets read the running statistics and leave them alone.
        _, self._inference_static = eqx.partition(
            eqx.nn.inference_mode(net), eqx.is_inexact_array
        )
        learner_key, self._act_key, memory_key, self._heldout_key = jr.split(key, 4)
        learner = Learner(
            params=params,
            state=state,
            target=(params, state),
            opt_state=optimiser.init(params),
            key=learner_key,
            num_updates=jnp.zeros((), jnp.int32),
        )
        # Between dispatches the learner is its leaves: walking its modules in Python at
        # every call costs as much as launching the call.
        self._structure = jax.tree.structure(learner)
        self._learner = tuple(jax.tree.leaves(learner))
        self._labels = tuple(jax.tree.leaves(net.parameter_groups()))
        self._memory_key = memory_key
        self._heldout_seed = int(jr.bits(self._heldout_key, dtype=jnp.uint32))
        self._uploaded: tuple[Draw, Batch] | None = None
        self._memory: ReplayMemory | None = None
        self._num_heldout = 0
        self._max_prio = 1.0
        self._num_updates = 0
        self._num_env_steps = 0

        self._act = jax.jit(partial(self._choose_actions, soft=soft))
        self._greedy_act = jax.jit(partial(self._choose_actions, soft=False))

        def train_burst(
            leaves: tuple[jax.Array, ...], batch: Batch, epsilon: jax.Array
        ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, dict[str, jax.Array]]]:
            """One _train_step on each of a burst's steps in turn, their outputs stacked."""
            learner, out = jax.lax.scan(
                partial(self._train_step, epsilon=epsilon),
                jax.tree.unflatten(self._structure, leaves),
                batch,
            )
            return tuple(jax.tree.leaves(learner)), out

        self._train_burst = jax.jit(train_burst)
        self._parameter_scalars = jax.jit(self._parameter_stats)
        self._heldout_scalars = jax.jit(self._score)

    def _choose_actions(
        self,
        leaves: tuple[jax.Array, ...],
        obs: jax.Array,
        key: jax.Array,
        step: jax.Array,
        epsilon: jax.Array,
        *,
        soft: bool,
    ) -> tuple[jax.Array, jax.Array]:
        """
        One action per environment from the learner of ``leaves``, under a fresh draw of the
        net's noise, and ``step + 1``. The randomness is ``key`` folded with ``step``.
        """
        learner: Learner = jax.tree.unflatten(self._structure, leaves)
        forward_key, policy_key, epsilon_key, random_key = jr.split(
            jr.fold_in(key, step), 4
        )
        out, _ = self._act_returns.forward(
            eqx.combine(learner.params, self._inference_static),
            obs,
            learner.state,
            key=forward_key,
        )
        qs = self._act_returns.means(out)
        if soft:
            actions = jr.categorical(policy_key, qs / self._temperature)
        else:
            actions = jnp.argmax(qs, -1)
        actions = jnp.where(
            jr.uniform(epsilon_key, actions.shape) < epsilon,
            jr.randint(random_key, actions.shape, 0, qs.shape[-1]),
            actions,
        )
        return actions, step + 1

    def _loss(
        self,
        params: PyTree[jax.Array],
        learner: Learner,
        batch: Batch,
        keys: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        jax.Array,
        tuple[
            eqx.nn.State,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            dict[str, jax.Array],
        ],
    ]:
        """
        The loss of ``params`` on one batch, ``[batch, ...]``, against ``learner``'s target
        net, from ``learner``'s state, under PER's importance weights. Returns it and, as aux,
        the new state, every transition's action values and the taken action's, signed and
        absolute TD errors and the return distribution's diagnostics. ``keys`` are the online
        pass', the target pass' and the augmentation's.

        The target is the n-step Munchausen one. The taken action's scaled log-policy is added
        to the return once, off the online pass. The next state is valued under the target
        net's soft policy with its entropy bonus, so the target distribution is the
        policy's mixture of the next state's, scaled by the discount and shifted by the rest.
        """
        common, replay, returns = self._common, self._replay, self._returns
        temperature = self._temperature
        online_key, target_key, augment_key = keys
        prios, _, frames, actions, rewards, lengths, terminated = map(
            jnp.asarray, batch
        )
        stack = frames.shape[1] - common.n_step
        obs, next_obs = frames[:, :stack], frames[:, -stack:]
        if self._augment is not None:
            obs, next_obs = (
                self._augment(x, k)
                for x, k in zip((obs, next_obs), jr.split(augment_key))
            )
        steps = jnp.arange(common.n_step)
        rewards = jnp.sum(
            jnp.where(steps < lengths[:, None], rewards, 0) * common.discount**steps,
            -1,
        )
        next_discount = jnp.where(terminated, 0.0, common.discount**common.n_step)
        actions = actions.astype(jnp.int32)
        rows = jnp.arange(actions.size)
        # Normalising by the batch's maximum cancels the sum of all priorities.
        is_weights = prios**-replay.per_beta
        is_weights /= is_weights.max()
        target_params, target_state = learner.target
        out, state = returns.forward(
            eqx.combine(params, self._static),
            obs,
            learner.state,
            key=online_key,
            taken=actions,
        )
        qs = returns.means(out)
        log_policy = jax.nn.log_softmax(jax.lax.stop_gradient(qs) / temperature)
        next_out, _ = returns.forward(
            eqx.combine(target_params, self._inference_static),
            next_obs,
            target_state,
            key=target_key,
        )
        next_log_policy = jax.nn.log_softmax(returns.means(next_out) / temperature)
        next_policy = jnp.exp(next_log_policy)
        next_entropy = -jnp.sum(next_policy * next_log_policy, -1)
        munchausen = common.munchausen_scaling_term * jnp.clip(
            temperature * log_policy[rows, actions],
            common.munchausen_clipping_val,
            0,
        )
        losses, td_error, abs_td_error, returns_stats = returns.loss(
            out,
            actions,
            next_out,
            next_policy,
            rewards + munchausen + next_discount * temperature * next_entropy,
            next_discount,
        )
        return jnp.mean(is_weights * losses), (
            state,
            qs,
            qs[rows, actions],
            td_error,
            abs_td_error,
            returns_stats,
        )

    def _train_step(
        self, learner: Learner, batch: Batch, *, epsilon: jax.Array
    ) -> tuple[Learner, tuple[jax.Array, dict[str, jax.Array]]]:
        """
        One gradient step of ``_loss`` on one batch, ``[batch, ...]``. Returns the new learner
        and, as a pair, every transition's new priority and the scalars. ``epsilon`` is the
        behaviour policy's, for the regret.
        """
        common, replay, temperature = self._common, self._replay, self._temperature
        key, *loss_keys = jr.split(learner.key, 4)
        (loss, (state, qs, taken, td_error, abs_td_error, returns_stats)), grads = (
            jax.value_and_grad(self._loss, has_aux=True)(
                learner.params, learner, batch, tuple(loss_keys)
            )
        )
        updates, opt_state = self._optimiser.update(
            grads, learner.opt_state, learner.params
        )
        params = optax.apply_updates(learner.params, updates)
        if self._project is not None:
            params = self._project(params)
        num_updates = learner.num_updates + 1
        target = optax.periodic_update(
            (params, state),
            learner.target,
            num_updates,
            common.target_network_update_freq,
        )

        log_policy = jax.nn.log_softmax(qs / temperature)
        policy = jnp.exp(log_policy)
        top_two = jax.lax.top_k(qs, 2)[0]
        # What the behaviour policy gives up per decision against the greedy action.
        regret = epsilon * (top_two[:, 0] - qs.mean(-1))
        if self._soft:
            regret += (1 - epsilon) * (top_two[:, 0] - jnp.sum(policy * qs, -1))
        spread = {
            "policy/entropy": -jnp.sum(policy * log_policy, -1),
            "policy/action_gap": top_two[:, 0] - top_two[:, 1],
            "policy/regret": regret,
        } | returns_stats
        # Means estimate the memory's, not what the draw favours; percentiles are the batch's
        # as drawn.
        memory_weights = 1 / jnp.asarray(batch.weights)
        memory_weights /= memory_weights.sum()
        stats = {"loss/total": loss, "optim/grad_norm": optax.global_norm(grads)} | {
            name: memory_weights @ value
            for name, value in {
                "loss/td_error": td_error,
                "loss/abs_td_error": abs_td_error,
                "value/q": taken,
                **spread,
            }.items()
        }
        percentiles = (5, 50, 95)
        for name, values in spread.items():
            for percentile, value in zip(
                percentiles,
                jnp.percentile(values, jnp.asarray(percentiles, jnp.float32)),
                strict=True,
            ):
                stats[f"{name}_p{percentile}"] = value
        # Handed to _parameter_stats, which divides them by the weights' norms.
        for name, tree in (
            ("grad", grads),
            ("update", jax.tree.map(jnp.subtract, params, learner.params)),
        ):
            squares = [jnp.sum(jnp.square(x)) for x in jax.tree.leaves(tree)]
            for part in PARTS:
                stats[f"_{name}_norm/{part}"] = jnp.sqrt(
                    sum(
                        (
                            square
                            for square, label in zip(squares, self._labels, strict=True)
                            if label in (f"{part}/kernel", f"{part}/noisy_kernel")
                        ),
                        jnp.zeros(()),
                    )
                )
        new_prios = (abs_td_error + replay.per_epsilon) ** replay.per_alpha
        return Learner(params, state, target, opt_state, key, num_updates), (
            new_prios,
            stats,
        )

    def _score(
        self, leaves: tuple[jax.Array, ...], batches: Batch, key: jax.Array
    ) -> dict[str, jax.Array]:
        """
        ``_loss`` with no update on ``ReplayMemory.draw_heldout``'s batches, one at a time, as
        a training batch is: the mean loss, signed and absolute TD error and taken action's
        value of the training environments' under ``replay/``, of the held-out ones' under
        ``heldout/``, and ``heldout/abs_td_ratio``, the held-out absolute TD error over the
        training environments'.
        """
        learner: Learner = jax.tree.unflatten(self._structure, leaves)
        sides, num_batches = batches.actions.shape[:2]

        def score(
            batch_and_key: tuple[Batch, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            batch, key = batch_and_key
            loss, (_, _, taken, td_error, abs_td_error, _) = self._loss(
                learner.params, learner, batch, tuple(jr.split(key, 3))
            )
            return loss, td_error.mean(), abs_td_error.mean(), taken.mean()

        flat = jax.tree.map(
            lambda x: jnp.reshape(x, (sides * num_batches, *x.shape[2:])), batches
        )
        means = jax.lax.map(score, (flat, jr.split(key, sides * num_batches)))
        loss, td_error, abs_td_error, q = (
            x.reshape(sides, num_batches).mean(-1) for x in means
        )
        stats = {"heldout/abs_td_ratio": abs_td_error[1] / abs_td_error[0]}
        for side, prefix in enumerate(("replay", "heldout")):
            stats |= {
                f"{prefix}/loss": loss[side],
                f"{prefix}/td_error": td_error[side],
                f"{prefix}/abs_td_error": abs_td_error[side],
                f"{prefix}/q": q[side],
            }
        return stats

    @property
    @override
    def num_updates(self) -> int:
        return self._num_updates

    @property
    @override
    def hyperparameters(self) -> dict[str, bool | int | float | str]:
        return dict(self._hyperparameters)

    @override
    def init(self, obs: npt.NDArray[Any], *, num_heldout: int) -> None:
        self._num_heldout = num_heldout
        self._memory = ReplayMemory(
            self._replay,
            obs,
            n_step=self._common.n_step,
            # A resumed memory draws anew rather than as the run's start did.
            seed=int(
                jr.bits(
                    jr.fold_in(self._memory_key, self._num_updates), dtype=jnp.uint32
                )
            ),
            max_prio=self._max_prio,
            num_heldout=num_heldout,
        )

    @override
    def act(
        self,
        obs: npt.NDArray[Any],
        *,
        num_env_steps: int,
        episode_start: npt.NDArray[np.bool_],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        self._num_env_steps = num_env_steps
        actions, _ = self._act(
            self._learner,
            obs,
            self._act_key,
            num_env_steps,
            self._epsilon(num_env_steps),
        )
        return actions, {}

    @override
    def observe(self, step: EnvStep) -> None:
        assert self._memory is not None, "observe before init"
        self._memory.observe(step)

    @override
    def learn_step(self) -> tuple[int, dict[str, jax.Array]]:
        """
        Every whole burst of gradient steps the memory owes, each left in flight. Each trains on
        the batch drawn and uploaded a burst before it: an upload starts only once the device
        has run everything dispatched before it, so a burst whose batch went up just before it
        would wait out the copy. The scalars are those of the newest step landing on
        PARAMETER_LOG_FREQ, with _parameter_stats' added, or else on TRAIN_LOG_FREQ, and empty
        if none does. Passing a multiple of HELDOUT_LOG_FREQ adds _score's. Those two are
        dispatched behind the burst, so that like the rest they are still on the device.
        """
        assert self._memory is not None, "learn_step before init"
        start = self._num_updates
        logged = dict[str, jax.Array]()
        logged_step = -1

        def rank(step: int) -> tuple[bool, int]:
            return step % PARAMETER_LOG_FREQ == 0, step

        while (draw := self._memory.draw()) is not None:
            uploaded, self._uploaded = (
                self._uploaded,
                (draw, jax.device_put(draw.batch)),
            )
            if uploaded is None:
                continue
            draw, batch = uploaded
            self._learner, (prios, stats) = self._train_burst(
                self._learner, batch, self._epsilon(self._num_env_steps)
            )
            self._memory.update(draw, prios)
            first = self._num_updates
            self._num_updates += len(draw.handles)
            due = [
                step
                for freq in (PARAMETER_LOG_FREQ, TRAIN_LOG_FREQ)
                if (step := self._num_updates - self._num_updates % freq) > first
            ]
            if due and rank(last := max(due, key=rank)) > rank(logged_step):
                logged = {
                    name: value[last - first - 1] for name, value in stats.items()
                }
                logged_step = last
        norms = {name: logged.pop(name) for name in list(logged) if name[0] == "_"}
        if logged and logged_step % PARAMETER_LOG_FREQ == 0:
            logged |= self._parameter_scalars(self._learner, norms)
        heldout_step = self._num_updates - self._num_updates % HELDOUT_LOG_FREQ
        if self._num_heldout and heldout_step > start:
            batches = self._memory.draw_heldout(
                max(1, HELDOUT_SAMPLES // self._replay.batch_size),
                np.random.default_rng((self._heldout_seed, heldout_step)),
            )
            logged |= self._heldout_scalars(
                self._learner,
                batches,
                jr.fold_in(self._heldout_key, heldout_step),
            )
        return self._num_updates - start, logged

    def _parameter_stats(
        self, leaves: tuple[jax.Array, ...], norms: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """
        The scalars read off the parameters and the optimiser's state, per part of the net in
        PARTS, over its weight matrices and kernels only:

        - ``weights/rms``: their root mean square.
        - ``optim/grad_to_weight`` and ``optim/update_to_weight``: the norm of the gradient and of
          the change the optimiser and any projection made, from the step ``norms`` comes from,
          over the weights' norm.
        - ``optim/adam_snr``: Adam's mean ``|m| / sqrt(v)``, without bias correction, so exact
          once its count is in the thousands: 1 for a gradient that points the same way every
          step, about 0.18 for iid noise at b1 = 0.9.
        - ``noise/sigma_ratio`` and ``noise/sigma_neg_frac``, where the part has NoisyLinears:
          mean ``|sigma|`` over mean ``|mu|``, and the fraction of sigmas below 0, which starts
          at 0 and reads 0.5 once their signs have random-walked.

        And the net's own ``weight_stats``.
        """
        learner: Learner = jax.tree.unflatten(self._structure, leaves)
        params = jax.tree.leaves(learner.params)
        labels = self._labels
        snr_sums: list[jax.Array | None] = [None] * len(params)
        for _, adam in otu.tree_get_all_with_path(
            learner.opt_state, "ScaleByAdamState"
        ):
            # A multi_transform group's moments hold a MaskedNode for every parameter outside it.
            moments = (
                jax.tree.leaves(
                    moment, is_leaf=lambda x: isinstance(x, optax.MaskedNode)
                )
                for moment in (adam.mu, adam.nu)
            )
            for index, (mu, nu) in enumerate(zip(*moments, strict=True)):
                if not isinstance(mu, optax.MaskedNode):
                    snr_sums[index] = jnp.sum(
                        jnp.where(nu > 0, jnp.abs(mu) * jax.lax.rsqrt(nu), 0.0)
                    )

        def total(values: Any) -> jax.Array:
            return sum(values, jnp.zeros(()))

        stats = dict[str, jax.Array]()
        for part in PARTS:
            kernels = [
                index
                for index, label in enumerate(labels)
                if label in (f"{part}/kernel", f"{part}/noisy_kernel")
            ]
            norm = jnp.sqrt(total(jnp.sum(jnp.square(params[i])) for i in kernels))
            stats[f"weights/rms/{part}"] = (
                norm / sum(params[i].size for i in kernels) ** 0.5
            )
            stats[f"optim/grad_to_weight/{part}"] = norms[f"_grad_norm/{part}"] / norm
            stats[f"optim/update_to_weight/{part}"] = (
                norms[f"_update_norm/{part}"] / norm
            )
            if adam := [i for i in kernels if snr_sums[i] is not None]:
                stats[f"optim/adam_snr/{part}"] = total(
                    snr_sums[i] for i in adam
                ) / sum(params[i].size for i in adam)
            sigmas = [
                params[i] for i, label in enumerate(labels) if label == f"{part}/sigma"
            ]
            if sigmas:
                means = [
                    params[i]
                    for i, label in enumerate(labels)
                    if label == f"{part}/noisy_kernel"
                ]
                stats[f"noise/sigma_ratio/{part}"] = total(
                    jnp.sum(jnp.abs(sigma)) for sigma in sigmas
                ) / total(jnp.sum(jnp.abs(mean)) for mean in means)
                stats[f"noise/sigma_neg_frac/{part}"] = total(
                    jnp.sum(sigma < 0) for sigma in sigmas
                ) / sum(sigma.size for sigma in sigmas)
        net: QNet = eqx.combine(learner.params, self._static)
        return stats | net.weight_stats()

    @override
    def stats(self) -> dict[str, float]:
        stats = {"policy/epsilon": self._epsilon(self._num_env_steps)}
        if self._memory is not None:
            stats |= {
                "replay/size": float(len(self._memory)),
                "replay/max_prio": self._memory.max_prio,
            }
        return stats

    @override
    def policy(self, *, num_env_steps: int) -> Policy:
        return QPolicy(
            self._greedy_act, self._learner, self._eval_epsilon(num_env_steps)
        )

    @override
    def checkpointables(self) -> dict[str, Any]:
        """
        The memory is not saved: a resumed run refills it, at the priority ``max_prio`` had.
        """
        learner: Learner = jax.tree.unflatten(self._structure, self._learner)
        target_params, target_state = learner.target
        return {
            "params": jax.tree.leaves(learner.params),
            "target_params": jax.tree.leaves(target_params),
            "state": learner.state,
            "target_state": target_state,
            "opt_state": learner.opt_state,
            "keys": {
                "learner": jr.key_data(learner.key),
                "act": jr.key_data(self._act_key),
            },
            "counters": {
                "num_updates": self._num_updates,
                "max_prio": self._max_prio
                if self._memory is None
                else self._memory.max_prio,
            },
        }

    @override
    def restore(self, restored: Mapping[str, Any]) -> None:
        learner: Learner = jax.tree.unflatten(self._structure, self._learner)
        structure = jax.tree.structure(learner.params)
        self._num_updates = int(restored["counters"]["num_updates"])
        self._max_prio = float(restored["counters"]["max_prio"])
        self._act_key = jr.wrap_key_data(restored["keys"]["act"])
        learner = Learner(
            params=jax.tree.unflatten(structure, restored["params"]),
            state=restored["state"],
            target=(
                jax.tree.unflatten(structure, restored["target_params"]),
                restored["target_state"],
            ),
            opt_state=restored["opt_state"],
            key=jr.wrap_key_data(restored["keys"]["learner"]),
            num_updates=jnp.asarray(self._num_updates, jnp.int32),
        )
        self._learner = tuple(jax.tree.leaves(learner))
