import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.typing import DTypeLike

from ..utils import cast_floating
from .batch_renorm import AxisName, BatchRenorm


def filter_norms(conv: eqx.nn.Conv2d) -> jax.Array:
    """
    The norm of each output channel's weights, its bias one more weight, in float32 and shaped
    to divide the output. A convolution is linear in kernel and bias together, so dividing its
    output is dividing its weights.
    """
    weights = conv.weight.reshape(conv.out_channels, -1)
    if conv.bias is not None:
        weights = jnp.concat([weights, conv.bias.reshape(-1, 1)], axis=1)
    return jnp.sqrt(jnp.square(weights.astype(jnp.float32)).sum(1)).reshape(-1, 1, 1)


class NormalisedResidualBlock(eqx.nn.StatefulLayer):
    """
    ``x + gain * conv1(silu(norm1(conv0(silu(x + bias)))))``, a bias per channel at the
    entrance. ``norm1`` is a per-channel batch renorm with a bias and no weight. ``conv0`` is
    made scale-invariant per filter by ``norm1``; ``conv1`` is weight-normalised per filter, so
    ``gain``, one per channel and zero at init, is the branch's only magnitude. Neither conv has
    a bias.
    """

    bias: jax.Array
    conv0: eqx.nn.Conv2d
    norm1: BatchRenorm
    conv1: eqx.nn.Conv2d
    gain: jax.Array

    def __init__(self, channels: int, *, axis_name: AxisName, key: jax.Array) -> None:
        key0, key1 = jr.split(key)
        self.bias = jnp.zeros((channels, 1, 1))
        self.conv0, self.conv1 = (
            eqx.nn.Conv2d(channels, channels, 3, padding="SAME", use_bias=False, key=k)
            for k in (key0, key1)
        )
        self.norm1 = BatchRenorm(channels, axis_name, use_weight=False)
        self.gain = jnp.zeros((channels, 1, 1))

    @jax.named_scope("NormalisedResidualBlock")
    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> tuple[jax.Array, eqx.nn.State]:
        h, state = self.norm1(self.conv0(jax.nn.silu(x + self.bias)), state)
        h = self.conv1(jax.nn.silu(h)) / filter_norms(self.conv1).astype(h.dtype)
        return x + self.gain * h, state


class NormalisedImpalaBlock(eqx.nn.StatefulLayer):
    """
    One stage of NormalisedImpalaCNN: a weight-normalised stem convolution and ImpalaBlock's
    max-pool, then two NormalisedResidualBlocks. The first stage's stem reads the image,
    ``rms_norm(max_pool(conv0(x)))`` with ``conv0``'s kernel and bias normalised jointly, so its
    stream starts at an RMS of 1. The others' end the previous stage's stream and start their
    own, ``max_pool(conv0(silu(rms_norm(x) + bias)))`` with ``conv0`` bias-free. Each RMSNorm is
    per sample over the whole map, without a weight.
    """

    conv0: eqx.nn.Conv2d
    pool: eqx.nn.MaxPool2d
    norm0: eqx.nn.RMSNorm
    bias0: jax.Array | None
    res1: NormalisedResidualBlock
    res2: NormalisedResidualBlock

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        spatial: tuple[int, int],
        reads_image: bool,
        axis_name: AxisName,
        key: jax.Array,
    ) -> None:
        conv_key, res1_key, res2_key = jr.split(key, 3)
        self.conv0 = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding="SAME",
            use_bias=reads_image,
            key=conv_key,
        )
        self.pool = eqx.nn.MaxPool2d(3, 2, padding=1)
        if reads_image:
            pooled = ((spatial[0] + 1) // 2, (spatial[1] + 1) // 2)
            norm_shape = (out_channels, *pooled)
            self.bias0 = None
        else:
            norm_shape = (in_channels, *spatial)
            self.bias0 = jnp.zeros((in_channels, 1, 1))
        self.norm0 = eqx.nn.RMSNorm(
            norm_shape, eps=1e-6, use_weight=False, use_bias=False
        )
        self.res1 = NormalisedResidualBlock(
            out_channels, axis_name=axis_name, key=res1_key
        )
        self.res2 = NormalisedResidualBlock(
            out_channels, axis_name=axis_name, key=res2_key
        )

    @jax.named_scope("NormalisedImpalaBlock")
    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> tuple[jax.Array, eqx.nn.State]:
        if self.bias0 is not None:
            x = jax.nn.silu(self.norm0(x) + self.bias0)
        x = self.pool(self.conv0(x) / filter_norms(self.conv0).astype(x.dtype))
        if self.bias0 is None:
            # Unfused, XLA computes the pool inside the norm's reduction and writes it in float32.
            x = self.norm0(lax.optimization_barrier(x))  # type: ignore[no-untyped-call]
        x, state = self.res1(x, state)
        return self.res2(x, state)


class NormalisedImpalaCNN(eqx.nn.StatefulLayer):
    """
    ImpalaCNNLarge's three stages, widths and pools, every stream read through a norm and
    written through a weight-normalised convolution (NormalisedImpalaBlock,
    NormalisedResidualBlock), with SiLU for every nonlinearity. ``dtype`` is as in
    ImpalaCNNLarge. The last stage's map is not pooled: it is RMS-normalised per sample in
    float32, given a bias per channel and passed through the SiLU, so the output is float32.
    ``axis_name`` is the residual blocks' batch renorms'.
    """

    blocks: tuple[NormalisedImpalaBlock, ...]
    out_norm: eqx.nn.RMSNorm
    out_bias: jax.Array
    dtype: DTypeLike = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        *,
        in_spatial: tuple[int, int],
        size_factor: int,
        axis_name: AxisName,
        dtype: DTypeLike,
        key: jax.Array,
    ) -> None:
        channels = (in_channels, 16 * size_factor, 32 * size_factor, 32 * size_factor)
        blocks = []
        spatial = in_spatial
        for i, k in enumerate(jr.split(key, 3)):
            blocks.append(
                NormalisedImpalaBlock(
                    channels[i],
                    channels[i + 1],
                    spatial=spatial,
                    reads_image=i == 0,
                    axis_name=axis_name,
                    key=k,
                )
            )
            spatial = ((spatial[0] + 1) // 2, (spatial[1] + 1) // 2)
        self.blocks = tuple(blocks)
        self.out_norm = eqx.nn.RMSNorm(
            (channels[-1], *spatial), eps=1e-6, use_weight=False, use_bias=False
        )
        self.out_bias = jnp.zeros((channels[-1], 1, 1))
        self.dtype = dtype
        self.out_features = channels[-1] * spatial[0] * spatial[1]

    def project_kernels(self) -> "NormalisedImpalaCNN":
        """
        Every filter rescaled to the square root of its fan-in, a weight RMS of 1 (EDM2's forced
        weight normalisation), the image stem's kernel and bias jointly. Every filter is
        normalised upon use, by weight normalisation or by the batch renorm it feeds, so this
        leaves the function unchanged, the batch renorms' running moments aside.
        """

        def convs(cnn: NormalisedImpalaCNN) -> list[eqx.nn.Conv2d]:
            leaves = jax.tree.leaves(
                cnn, is_leaf=lambda x: isinstance(x, eqx.nn.Conv2d)
            )
            return [x for x in leaves if isinstance(x, eqx.nn.Conv2d)]

        def project(conv: eqx.nn.Conv2d) -> eqx.nn.Conv2d:
            fan_in = conv.weight[0].size + (conv.bias is not None)
            scale = fan_in**0.5 / filter_norms(conv)
            return jax.tree.map(  # type: ignore[no-any-return]
                lambda w: w * scale.reshape(-1, *(1,) * (w.ndim - 1)), conv
            )

        return eqx.tree_at(  # type: ignore[no-any-return]
            convs, self, [project(conv) for conv in convs(self)]
        )

    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> tuple[jax.Array, eqx.nn.State]:
        x = x.astype(self.dtype)
        for block in cast_floating(self.blocks, self.dtype):
            x, state = block(x, state)
        x = jax.nn.silu(self.out_norm(x.astype(jnp.float32)) + self.out_bias)
        # Channel-major: each group of a grouped linear reading this gets a run of channels.
        return x.reshape(-1), state
