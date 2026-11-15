"""Generic helpers with no narrower home."""

import equinox as eqx
import jax
from jax.typing import DTypeLike


def cast_floating[T](tree: T, dtype: DTypeLike) -> T:
    return jax.tree.map(  # type: ignore[no-any-return]
        lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, tree
    )
