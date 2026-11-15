import equinox as eqx
import jax
import jax.numpy as jnp


class IQNCosineEmbedding(eqx.Module):
    """``relu(linear(cos(pi i tau)))`` for i in 1..num_cosines, of one quantile level ``tau``."""

    linear: eqx.nn.Linear

    def __init__(self, out_features: int, *, num_cosines: int, key: jax.Array) -> None:
        self.linear = eqx.nn.Linear(num_cosines, out_features, key=key)

    def __call__(self, tau: jax.Array) -> jax.Array:
        i = jnp.arange(1, self.linear.weight.shape[1] + 1)
        return jax.nn.relu(self.linear(jnp.cos(jnp.pi * i * tau)))
