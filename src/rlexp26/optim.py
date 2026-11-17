"""Optax schedules and gradient transformations."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from jax.typing import ArrayLike


def inverse_sqrt_schedule(init_value: float, decay_steps: float) -> optax.Schedule:
    """
    ``init_value / sqrt(1 + count / decay_steps)``: constant early, ``count ** -0.5`` late, and
    constant throughout at ``decay_steps`` inf.
    """

    def schedule(count: ArrayLike) -> jax.Array:
        return init_value / jnp.sqrt(1 + jnp.asarray(count) / decay_steps)

    return schedule


class ScaleByPooledAdamState(NamedTuple):
    steps: jax.Array
    mu: optax.Updates
    # One second moment per leaf.
    nu: optax.Updates


def scale_by_pooled_adam(
    b1: ArrayLike = 0.9,
    b2: ArrayLike = 0.999,
    eps: ArrayLike = 1e-8,
) -> optax.GradientTransformation:
    """
    ``optax.scale_by_adam`` with its second moment pooled over each leaf: the running mean of the
    squared gradient over the leaf's entries, Adam-mini's update with one block per leaf. Each entry keeps its own
    momentum, so a leaf's steps stay in proportion to its entries' gradients: an entry whose
    gradient is far below the leaf's root-mean-square barely moves, where Adam would step it by
    the full rate.
    """

    def init_fn(params: optax.Params) -> ScaleByPooledAdamState:
        return ScaleByPooledAdamState(
            steps=jnp.zeros([], jnp.int32),
            mu=otu.tree_zeros_like(params),
            nu=jax.tree.map(lambda param: jnp.zeros([], param.dtype), params),
        )

    def update_fn(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, ScaleByPooledAdamState]:
        del params
        if not isinstance(state, ScaleByPooledAdamState):
            raise TypeError(
                f"Expected 'state' to be an instance of ScaleByPooledAdamState,"
                f" got {type(state).__name__!r}"
            )

        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        nu = jax.tree.map(
            lambda g, v: b2 * v + (1 - b2) * jnp.square(g).mean(), updates, state.nu
        )
        steps = jnp.asarray(optax.safe_increment(state.steps))
        mu_hat = otu.tree_bias_correction(mu, b1, steps)
        nu_hat = otu.tree_bias_correction(nu, b2, steps)
        updates = jax.tree.map(lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
        return updates, ScaleByPooledAdamState(steps=steps, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


class AddInitPullState(NamedTuple):
    init_params: optax.Params


def add_init_pull(strength: ArrayLike) -> optax.GradientTransformation:
    """
    ``optax.add_decayed_weights`` pulling towards the parameters' values at ``init`` rather than
    zero: adds ``strength * (params - init_params)``. Chained where ``optax.adamw`` adds its decay,
    between the scaling and the learning rate, the pull is decoupled: every entry closes
    ``learning_rate * strength`` of its distance to its initial value per step, whatever its
    gradient.
    """

    def init_fn(params: optax.Params) -> AddInitPullState:
        # A copy, so that donating the parameters' buffers leaves the anchor intact.
        return AddInitPullState(init_params=jax.tree.map(jnp.copy, params))

    def update_fn(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, AddInitPullState]:
        if not isinstance(state, AddInitPullState):
            raise TypeError(
                f"Expected 'state' to be an instance of AddInitPullState,"
                f" got {type(state).__name__!r}"
            )
        assert params is not None
        updates = jax.tree.map(
            lambda g, p, p0: g + strength * (p - p0),
            updates,
            params,
            state.init_params,
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)
