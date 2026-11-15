from collections.abc import Hashable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

type AxisName = Hashable | Sequence[Hashable]


class BatchRenorm(eqx.nn.StatefulLayer):
    """
    Batch renormalisation (Ioffe, 2017) per channel, the first axis, over the rest of the sample
    and the vmapped axes ``axis_name``. Training reads the batch's moments corrected towards the
    running ones,

        y = weight * ((x - mean_B) / std_B * r + d) + bias,
        r = sg(clip(std_B / std, 1 / r_max, r_max)),  d = sg(clip((mean_B - mean) / std, -d_max, d_max)),

    so that inside the clips the output is the running moments' normalisation and the gradient
    is batch norm's. The running moments are exponential averages from zero, debiased as Adam's
    are, and include the batch read against them; before the first update they read 0 and 1.
    ``inference`` normalises by the running moments and leaves them alone. ``mask``, a bool per
    sample, picks the samples the batch's moments are read from; every sample is normalised by
    them. It computes in its input's dtype; the moments are accumulated and kept in float32.
    """

    weight: jax.Array | None
    bias: jax.Array | None
    moments: eqx.nn.StateIndex[tuple[jax.Array, jax.Array, jax.Array]]
    axis_name: AxisName = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    r_max: float = eqx.field(static=True)
    d_max: float = eqx.field(static=True)
    inference: bool

    def __init__(
        self,
        num_features: int,
        axis_name: AxisName,
        *,
        use_weight: bool = True,
        use_bias: bool = True,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        r_max: float = 3.0,
        d_max: float = 5.0,
        inference: bool = False,
    ) -> None:
        self.weight = jnp.ones(num_features) if use_weight else None
        self.bias = jnp.zeros(num_features) if use_bias else None
        self.moments = eqx.nn.StateIndex(
            (jnp.zeros(num_features), jnp.zeros(num_features), jnp.zeros((), jnp.int32))
        )
        self.axis_name = axis_name
        self.momentum = momentum
        self.epsilon = epsilon
        self.r_max = r_max
        self.d_max = d_max
        self.inference = inference

    @jax.named_scope("BatchRenorm")
    def __call__(
        self, x: jax.Array, state: eqx.nn.State, *, mask: jax.Array | bool = True
    ) -> tuple[jax.Array, eqx.nn.State]:
        mean, var, updates = state.get(self.moments)

        if not self.inference:
            weight = jnp.asarray(mask, jnp.float32)

            def batch_moments(y: jax.Array) -> tuple[jax.Array, jax.Array]:
                y = y.astype(jnp.float32)
                count, total, squares = lax.psum(  # type: ignore[no-untyped-call]
                    (weight, weight * y.mean(), weight * jnp.square(y).mean()),
                    self.axis_name,
                )
                return total / count, squares / count - jnp.square(total / count)

            batch_mean, batch_var = jax.vmap(batch_moments)(x)
            batch_var = jnp.maximum(batch_var, 0)
            mean, var = (
                self.momentum * moment + (1 - self.momentum) * lax.stop_gradient(batch)
                for moment, batch in ((mean, batch_mean), (var, batch_var))
            )
            updates += 1
            state = state.set(self.moments, (mean, var, updates))

        debias = 1 - self.momentum**updates
        mean = jnp.where(updates > 0, mean / debias, 0.0)
        std = jnp.sqrt(jnp.where(updates > 0, var / debias, 1.0) + self.epsilon)

        if self.inference:
            centre, scale, shift = mean, 1 / std, jnp.zeros_like(mean)
        else:
            batch_std = jnp.sqrt(batch_var + self.epsilon)
            # r and d are constants to the gradient; 1 / batch_std is not.
            r = jnp.clip(batch_std / std, 1 / self.r_max, self.r_max)
            d = jnp.clip((batch_mean - mean) / std, -self.d_max, self.d_max)
            centre = batch_mean
            scale = lax.stop_gradient(r) / batch_std
            shift = lax.stop_gradient(d)
        if self.weight is not None:
            scale, shift = scale * self.weight, shift * self.weight
        if self.bias is not None:
            shift = shift + self.bias
        y = jax.vmap(lambda y, c, a, b: (y - c) * a + b)(
            x, *(v.astype(x.dtype) for v in (centre, scale, shift))
        )
        return y, state
