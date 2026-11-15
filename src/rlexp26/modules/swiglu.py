import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .batch_renorm import AxisName, BatchRenorm
from .noisy_linear import NoisyLinear


class NormalisedSwiGLUBlock(eqx.nn.StatefulLayer):
    """
    ``x + gain * linear1(norm1(silu(norm0(g)) * v))``, ``g, v`` the two halves of
    ``linear0(x)``, a gated unit from ``features`` to ``hidden`` and back. ``linear0`` is made
    scale-invariant per neuron by the batch renorms, neither with a weight: ``norm0``, with a
    bias, before the activation, which is not homogeneous, and ``norm1``, without, on the
    product, through which a fixed gate passes ``v``'s scale. ``linear1`` is weight-normalised,
    so ``gain``, one per feature and zero at init, is the branch's only magnitude. Neither
    linear has a bias. ``axis_name`` and ``mask`` are the batch renorms'.
    """

    linear0: NoisyLinear
    norm0: BatchRenorm
    norm1: BatchRenorm
    linear1: NoisyLinear
    gain: jax.Array

    def __init__(
        self, features: int, hidden: int, *, axis_name: AxisName, key: jax.Array
    ) -> None:
        key0, key1 = jr.split(key)
        self.linear0 = NoisyLinear(features, 2 * hidden, use_bias=False, key=key0)
        self.norm0 = BatchRenorm(hidden, axis_name, use_weight=False)
        self.norm1 = BatchRenorm(hidden, axis_name, use_weight=False, use_bias=False)
        self.linear1 = NoisyLinear(
            hidden, features, use_bias=False, weight_norm=True, key=key1
        )
        self.gain = jnp.zeros(features)

    @jax.named_scope("NormalisedSwiGLUBlock")
    def __call__(
        self,
        x: jax.Array,
        state: eqx.nn.State,
        *,
        key: jax.Array,
        mask: jax.Array | bool = True,
    ) -> tuple[jax.Array, eqx.nn.State]:
        key0, key1 = jr.split(key)
        g, v = jnp.split(self.linear0(x, key=key0), 2)
        g, state = self.norm0(g, state, mask=mask)
        h, state = self.norm1(jax.nn.silu(g) * v, state, mask=mask)
        return x + self.gain * self.linear1(h, key=key1), state
