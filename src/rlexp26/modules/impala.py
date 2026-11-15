import equinox as eqx
import jax
import jax.random as jr
from jax.typing import DTypeLike

from ..utils import cast_floating
from .vendored.spectral_norm import SpectralNorm


class ImpalaResSubBlock(eqx.nn.StatefulLayer):
    """``x + conv1(relu(conv0(relu(x))))``, both convs spectrally normalised, per BTR."""

    conv0: SpectralNorm[eqx.nn.Conv2d]
    conv1: SpectralNorm[eqx.nn.Conv2d]

    def __init__(self, channels: int, *, key: jax.Array) -> None:
        conv0_key, sn0_key, conv1_key, sn1_key = jr.split(key, 4)
        self.conv0, self.conv1 = (
            SpectralNorm(
                eqx.nn.Conv2d(channels, channels, 3, padding="SAME", key=conv_key),
                "weight",
                key=sn_key,
            )
            for conv_key, sn_key in ((conv0_key, sn0_key), (conv1_key, sn1_key))
        )

    @jax.named_scope("ImpalaResSubBlock")
    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> tuple[jax.Array, eqx.nn.State]:
        h, state = self.conv0(jax.nn.relu(x), state)
        h, state = self.conv1(jax.nn.relu(h), state)
        return x + h, state


class ImpalaBlock(eqx.nn.StatefulLayer):
    """
    A stem conv, unnormalised per BTR, a 3x3/2 max-pool padded by 1 on every side, as BTR's
    code pads it, and two ImpalaResSubBlocks.
    """

    conv0: eqx.nn.Conv2d
    pool: eqx.nn.MaxPool2d
    res1: ImpalaResSubBlock
    res2: ImpalaResSubBlock

    def __init__(self, in_channels: int, out_channels: int, *, key: jax.Array) -> None:
        conv_key, res1_key, res2_key = jr.split(key, 3)
        self.conv0 = eqx.nn.Conv2d(
            in_channels, out_channels, 3, padding="SAME", key=conv_key
        )
        self.pool = eqx.nn.MaxPool2d(3, 2, padding=1)
        self.res1 = ImpalaResSubBlock(out_channels, key=res1_key)
        self.res2 = ImpalaResSubBlock(out_channels, key=res2_key)

    @jax.named_scope("ImpalaBlock")
    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> tuple[jax.Array, eqx.nn.State]:
        x = self.pool(self.conv0(x))
        x, state = self.res1(x, state)
        return self.res2(x, state)


class ImpalaCNNLarge(eqx.nn.StatefulLayer):
    """
    Impala's three stages, ``size_factor`` times as wide, then an adaptive max-pool to
    ``pooled_size``, a ReLU and a flatten. Equinox's adaptive pool partitions each axis into
    cells without overlap, as BTR's paper describes; ``torch.nn.AdaptiveMaxPool2d``, which BTR's
    code runs, overlaps them and pools less.

    ``dtype`` is the convolutions' compute dtype. What it does and does not reach, for a 16-bit
    one:

    - Parameters stay float32: the blocks are cast down for each call, and their gradients come
      back up through that cast.
    - The forward activations are ``dtype``, and so is the trunk's output: casting is the
      caller's.
    - The backward pass is ``dtype`` too: cuDNN accumulates in float32 but writes the gradients
      out in ``dtype``, so a float32 parameter's gradient carries ``dtype``'s precision.
    - SpectralNorm's ``u`` and ``v`` stay float32, so its power iteration runs in float32 on the
      rounded kernel the convolution reads.
    """

    blocks: tuple[ImpalaBlock, ...]
    pool: eqx.nn.AdaptiveMaxPool2d
    dtype: DTypeLike = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        *,
        size_factor: int,
        pooled_size: tuple[int, int],
        dtype: DTypeLike,
        key: jax.Array,
    ) -> None:
        channels = (in_channels, 16 * size_factor, 32 * size_factor, 32 * size_factor)
        self.blocks = tuple(
            ImpalaBlock(c_in, c_out, key=k)
            for c_in, c_out, k in zip(channels, channels[1:], jr.split(key, 3))
        )
        self.pool = eqx.nn.AdaptiveMaxPool2d(pooled_size)
        self.dtype = dtype
        self.out_features = channels[-1] * pooled_size[0] * pooled_size[1]

    @jax.named_scope("ImpalaCNNLarge")
    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> tuple[jax.Array, eqx.nn.State]:
        x = x.astype(self.dtype)
        for block in cast_floating(self.blocks, self.dtype):
            x, state = block(x, state)
        # relu is monotonic, so it commutes with the max-pool.
        return jax.nn.relu(self.pool(x)).reshape(-1), state
