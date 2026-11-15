import equinox as eqx
import jax
import jax.random as jr

from .iqn import IQNCosineEmbedding
from .noisy_linear import NoisyLinear


class DuelingIQNHead(eqx.Module):
    """
    BTR's head: the features times each quantile level's cosine embedding, then a value and an
    advantage stream of two NoisyLinears each, ``value + advantage - mean(advantage)``. Reads
    one sample's features and ``taus``, ``[num_levels]``, and returns
    ``[num_levels, num_actions]``, with one draw of noise for all the levels.
    ``advantage_only`` skips the value stream: each level's values are then off by one
    constant across the actions, which an argmax or a softmax over them does not read.
    """

    embedding: IQNCosineEmbedding
    value0: NoisyLinear
    value1: NoisyLinear
    advantage0: NoisyLinear
    advantage1: NoisyLinear

    def __init__(
        self,
        in_features: int,
        num_actions: int,
        *,
        num_cosines: int,
        hidden: int = 512,
        key: jax.Array,
    ) -> None:
        keys = jr.split(key, 5)
        self.embedding = IQNCosineEmbedding(
            in_features, num_cosines=num_cosines, key=keys[0]
        )
        self.value0 = NoisyLinear(in_features, hidden, key=keys[1])
        self.value1 = NoisyLinear(hidden, 1, key=keys[2])
        self.advantage0 = NoisyLinear(in_features, hidden, key=keys[3])
        self.advantage1 = NoisyLinear(hidden, num_actions, key=keys[4])

    @jax.named_scope("DuelingIQNHead")
    def __call__(
        self,
        x: jax.Array,
        taus: jax.Array,
        *,
        key: jax.Array,
        advantage_only: bool = False,
    ) -> jax.Array:
        keys = jr.split(key, 4)

        def level(tau: jax.Array) -> jax.Array:
            h = x * self.embedding(tau)
            a = self.advantage1(
                jax.nn.relu(self.advantage0(h, key=keys[2])), key=keys[3]
            )
            if advantage_only:
                return a - a.mean()
            v = self.value1(jax.nn.relu(self.value0(h, key=keys[0])), key=keys[1])
            return v + a - a.mean()

        return jax.vmap(level)(taus)
