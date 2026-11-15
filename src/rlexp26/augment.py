"""Image augmentations for a batch ``[batch, ..., H, W]``, each drawn per sample."""

import jax
import jax.numpy as jnp
import jax.random as jr


def random_shift(x: jax.Array, key: jax.Array, pad: int) -> jax.Array:
    """DrQ's random shift: padded by ``pad`` pixels repeating the edge, cropped back at a random offset."""
    padded = jnp.pad(x, [(0, 0)] * (x.ndim - 2) + [(pad, pad)] * 2, mode="edge")
    offsets = jr.randint(key, (x.shape[0], 2), 0, 2 * pad + 1)
    return jax.vmap(
        lambda image, offset: jax.lax.dynamic_slice(
            image, (*[0] * (image.ndim - 2), *offset), x.shape[1:]
        )
    )(padded, offsets)


def random_intensity(x: jax.Array, key: jax.Array, scale: float) -> jax.Array:
    """BBF's brightness factor: times ``1 + scale * clip(N(0, 1), -2, 2)``."""
    noise = jr.normal(key, (x.shape[0], *[1] * (x.ndim - 1)))
    return x * (1 + scale * jnp.clip(noise, -2, 2))
