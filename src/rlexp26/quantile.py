"""Returns as quantiles: the quantile Huber loss and how an IQN head is read and trained."""

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.typing import ArrayLike


def scaled_huber_loss(
    errors: ArrayLike, delta: ArrayLike, *, eps: float = 1e-8
) -> jax.Array:
    """
    ``optax.huber_loss`` divided by ``delta``, as the quantile-Huber uses it, but defined at
    ``delta = 0``, where it is ``|errors|`` and the quantile-Huber becomes the pinball loss.

    At ``delta = 0`` and ``errors = 0`` the derivative is 0.5, a valid subgradient of
    ``|errors|``, because ``jnp.minimum`` splits its tie.
    """
    abs_errors = jnp.abs(errors)
    quadratic = jnp.minimum(abs_errors, delta)
    return abs_errors - quadratic + jnp.square(quadratic) / (2 * delta + eps)


@dataclass(frozen=True)
class QuantileReturns:
    """
    The return distributions of a net that answers ``[batch, levels, actions]`` values at the
    quantile levels it is given, IQN's: ``num_samples`` levels drawn uniformly per sample and
    pass, fitted with the quantile Huber of knee ``huber_k``. A ``ReturnDistribution``.
    """

    num_samples: int
    huber_k: float

    def forward(
        self,
        net: Callable[..., tuple[jax.Array, eqx.nn.State]],
        obs: jax.Array,
        state: eqx.nn.State,
        *,
        key: jax.Array,
        taken: jax.Array | None = None,
    ) -> tuple[tuple[jax.Array, jax.Array], eqx.nn.State]:
        noise_key, tau_key = jr.split(key)
        taus = jr.uniform(tau_key, (obs.shape[0], self.num_samples))
        values, state = net(obs, state, key=noise_key, taus=taus)
        return (values, taus), state

    def means(self, out: tuple[jax.Array, jax.Array]) -> jax.Array:
        values, _ = out
        return values.mean(-2)

    def loss(
        self,
        out: tuple[jax.Array, jax.Array],
        actions: jax.Array,
        next_out: tuple[jax.Array, jax.Array],
        next_policy: jax.Array,
        shift: jax.Array,
        scale: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        values, taus = out
        next_values, _ = next_out
        action_values = values[jnp.arange(actions.size), :, actions]
        targets = shift[:, None] + scale[:, None] * jnp.sum(
            next_policy[:, None, :] * next_values, -1
        )
        # [batch, online level, target level]
        td = targets[:, None, :] - action_values[:, :, None]
        pinball = jnp.abs(taus[:, :, None] - (td < 0))
        losses = (pinball * scaled_huber_loss(td, self.huber_k)).mean(-1).sum(-1)
        # The magnitude is taken before the reduction: errors straddling the target
        # distribution would cancel.
        return (
            losses,
            (pinball * td).mean((-2, -1)),
            jnp.abs(pinball * td).mean((-2, -1)),
            {"value/return_std": action_values.std(-1)},
        )
