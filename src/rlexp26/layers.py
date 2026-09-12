"""
The network components this project builds for itself, kept free of BTR's hyperparameters so
that the algorithm owns its numbers and this file owns only the shapes. Anything here should
make sense to a different agent with different constants; ``btr.QNet`` is where they are chosen.

``internals.SpectralNorm`` is vendored Flax rather than ours -- see that module's header.
"""

import math
from collections.abc import Collection

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

from .internals import SpectralNorm


def adaptive_max_pool(x: jax.Array, output_size: Collection[int]) -> jax.Array:
    """
    Applies adaptive max pooling to map arbitrary spatial dimensions to a fixed output_size.
    Calculates fixed stride, kernel size, and padding values that satisfy standard
    pooling output formulas: O = floor((H + pad - kernel) / stride) + 1
    """

    strides = list[int]()
    window_shape = list[int]()
    padding = list[tuple[int, int]]()

    for h, o in zip(x.shape[-len(output_size) - 1 : -1], output_size):
        if h < o:
            raise ValueError(
                "Adaptive max pool output size must not be larger than the input size"
            )

        s1, r = divmod(h, o)

        if 0 < r and o - r <= s1:
            s = k = s1 + 1
            p = o - r
        else:
            s = s1
            k = s1 + r
            p = 0

        strides.append(s)
        window_shape.append(k)
        padding.append(((p + 1) // 2, p // 2))

    return nnx.max_pool(  # type: ignore[no-untyped-call, no-any-return]
        x,
        window_shape=tuple(window_shape),
        strides=tuple(strides),
        padding=padding,  # pyright: ignore
    )


class NoisyLinear(nnx.Module):
    """
    Factorised NoisyNets linear layer.

    One deliberate departure from BTR: each ``__call__`` redraws its noise pair, where BTR draws
    once per gradient step, so the Munchausen term and the next-state quantiles see different
    target noise -- which is what NoisyNets' Algorithm 1 argues for.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        sigma0: float = 0.5,
        deterministic: bool = False,
    ):
        init_range = 1 / math.sqrt(in_features)

        self.kernel_mean = nnx.Param(
            rngs.params.uniform(
                (in_features, out_features), minval=-init_range, maxval=init_range
            )
        )
        self.kernel_sigma = nnx.Param(
            jnp.full((in_features, out_features), sigma0 * init_range)
        )
        self.bias_mean = nnx.Param(
            rngs.params.uniform((out_features,), minval=-init_range, maxval=init_range),
        )
        self.bias_sigma = nnx.Param(jnp.full((out_features,), sigma0 * init_range))

        self.in_features = in_features
        self.out_features = out_features

        self.deterministic = deterministic

    @staticmethod
    def _f(x: ArrayLike) -> jax.Array:
        return jnp.sign(x) * jnp.sqrt(jnp.abs(x))

    def __call__(
        self,
        inputs: ArrayLike,
        *,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        if deterministic is None:
            deterministic = self.deterministic

        inputs = jnp.asarray(inputs)

        y: jax.Array = inputs @ self.kernel_mean[...] + self.bias_mean
        if deterministic:
            return y

        if rngs is None:
            raise ValueError(
                "rngs argument is required unless deterministic behaviour is specified"
            )

        eps_in = self._f(rngs.noise.normal((self.in_features,)))
        eps_out = self._f(rngs.noise.normal((self.out_features,)))

        noise = eps_out * ((eps_in * inputs) @ self.kernel_sigma + self.bias_sigma)

        return y + noise


class ImpalaResSubBlock(nnx.Module):
    """
    Impala Residual Sub-Block.

    Both convs are spectrally normalised and the enclosing block's stem conv is not, per BTR.
    ``internals.SpectralNorm`` projects rather than mutates -- the ``# fix:`` at the end of its
    ``__call__`` puts the raw weight back, which is what makes it the same algorithm as the
    ``torch.nn.utils.spectral_norm`` BTR uses.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.conv0 = SpectralNorm(
            nnx.Conv(in_features, out_features, kernel_size=(3, 3), rngs=rngs),
            rngs=rngs,
        )
        self.conv1 = SpectralNorm(
            nnx.Conv(out_features, out_features, kernel_size=(3, 3), rngs=rngs),
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> jax.Array:
        y: jax.Array = jnp.asarray(x)
        y = nnx.relu(y)
        y = self.conv0(y)
        y = nnx.relu(y)
        y = self.conv1(y)
        return y + x


class ImpalaBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.conv0 = nnx.Conv(in_features, out_features, kernel_size=(3, 3), rngs=rngs)
        self.res1 = ImpalaResSubBlock(out_features, out_features, rngs=rngs)
        self.res2 = ImpalaResSubBlock(out_features, out_features, rngs=rngs)

    def __call__(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        x = self.conv0(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")  # type: ignore[no-untyped-call]
        x = self.res1(x)
        x = self.res2(x)
        return x


POOLED_SIZE = (6, 6)  # what adaptive_max_pool maps the trunk's spatial dims to


class ImpalaCNNLarge(nnx.Module):
    def __init__(
        self,
        in_features: int,
        *,
        rngs: nnx.Rngs,
        size_factor: int,
    ) -> None:
        self.size_factor = size_factor

        n = self.size_factor
        self.block0 = ImpalaBlock(in_features, n * 16, rngs=rngs)
        self.block1 = ImpalaBlock(n * 16, n * 32, rngs=rngs)
        self.block2 = ImpalaBlock(n * 32, n * 32, rngs=rngs)

    @property
    def out_channels(self) -> int:
        """
        Feature maps in the trunk's output, before the flatten below folds them together with
        the pooled positions. A diagnostic that counts convolutional neurons counts these.
        """
        return self.size_factor * 32

    @property
    def out_features(self) -> int:
        return self.out_channels * POOLED_SIZE[0] * POOLED_SIZE[1]

    def __call__(self, x: ArrayLike) -> jax.Array:
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        # Impala ends its trunk on a ReLU. Order against the pool is irrelevant:
        # relu is monotonic, so max(relu(x)) == relu(max(x)).
        x = nnx.relu(x)
        x = adaptive_max_pool(x, POOLED_SIZE)
        return x.reshape(*x.shape[:-3], -1)


class IQNCosineEmbedding(nnx.Module):
    def __init__(
        self,
        out_features: int,
        *,
        num_cosines: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_cosines = num_cosines
        self.linear = nnx.Linear(self.num_cosines, out_features, rngs=rngs)

    def __call__(self, x: ArrayLike) -> jax.Array:
        pi_factors = jnp.pi * jnp.arange(1, self.num_cosines + 1)
        x = pi_factors * jnp.expand_dims(x, -1)
        x = jnp.cos(x)
        x = self.linear(x)
        x = nnx.relu(x)
        return x
