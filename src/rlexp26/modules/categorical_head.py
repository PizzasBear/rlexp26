import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .noisy_linear import NoisyLinear


class CategoricalHead(eqx.Module):
    """
    MXQ1's head: NoisyLinear, LayerNorm, ReLU and a noiseless linear readout, from one sample's
    features to log-probabilities ``[num_actions, num_atoms]``. The readout's kernel starts at
    zero and its bias at ``init_logits`` for every action, so every state starts at that one
    distribution: random or noisy logits would start with mass on the outermost atoms, which
    bootstrapping removes only slowly.
    """

    linear0: NoisyLinear
    norm: eqx.nn.LayerNorm
    linear1: eqx.nn.Linear
    num_actions: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        num_actions: int,
        init_logits: jax.Array,
        *,
        hidden: int = 512,
        key: jax.Array,
    ) -> None:
        key0, key1 = jr.split(key)
        self.linear0 = NoisyLinear(in_features, hidden, key=key0)
        self.norm = eqx.nn.LayerNorm(hidden, eps=1e-6)
        linear1 = eqx.nn.Linear(hidden, num_actions * init_logits.size, key=key1)
        self.linear1 = eqx.tree_at(
            lambda linear: (linear.weight, linear.bias),
            linear1,
            (jnp.zeros_like(linear1.weight), jnp.tile(init_logits, num_actions)),
        )
        self.num_actions = num_actions

    @jax.named_scope("CategoricalHead")
    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        x = jax.nn.relu(self.norm(self.linear0(x, key=key)))
        return jax.nn.log_softmax(self.linear1(x).reshape(self.num_actions, -1))
