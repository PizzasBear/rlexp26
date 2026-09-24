"""Losses and diagnostics over any ``nnx.Module``; nothing here knows about BTR."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from flax import nnx
from jax.typing import ArrayLike

from .internals import SpectralNorm
from .layers import NoisyLinear


def scaled_huber_loss(
    errors: ArrayLike, delta: ArrayLike, *, eps: float = 1e-8
) -> jax.Array:
    """
    ``optax.huber_loss`` divided by ``delta``, as the quantile-Huber uses it, but defined at
    ``delta = 0``, where it is ``|errors|`` and the quantile-Huber becomes the pinball loss.

    At ``delta = 0`` and ``errors = 0`` the derivative is 0.5, a valid subgradient of
    ``|errors|``, because ``jnp.minimum`` splits its tie.
    """
    abs_errors = jnp.abs(errors)
    quadratic = jnp.minimum(abs_errors, delta)
    return abs_errors - quadratic + jnp.square(quadratic) / (2 * delta + eps)


def unnormalised_param_mask(module: nnx.Module) -> nnx.State[Any, Any]:
    """
    ``module``'s ``nnx.Param`` tree with a boolean at every leaf, false for a normalisation
    layer's scale or bias: the shape ``optax.adamw``'s ``mask`` takes.

    Weight decay skips them: after normalisation the gain alone carries the layer's scale, so
    decay moves that scale downstream instead of shrinking it. Global norms skip them too: a
    gain sits near one and says nothing about growth, but its L2 grows with its size. Other
    biases are kept, since nothing else bounds them.
    """
    norm_paths = {
        path
        for path, node in nnx.iter_graph(module)
        if isinstance(node, nnx.LayerNorm | nnx.RMSNorm | nnx.BatchNorm | nnx.GroupNorm)
    }
    mask: nnx.State[Any, Any] = jax.tree_util.tree_map_with_path(
        lambda path, _: (
            tuple(key.key for key in path if isinstance(key, jax.tree_util.DictKey))[
                :-1
            ]
            not in norm_paths
        ),
        # as_pure: the mask has to match the unwrapped tree nnx.Optimizer hands optax.
        nnx.as_pure(nnx.state(module, nnx.Param)),
    )
    return mask


def masked_leaves(mask: Any, tree: Any) -> list[jax.Array]:
    """
    The leaves of ``tree`` whose matching leaf in ``mask`` is true, flat. One mask serves the
    parameters, their gradient and an Adam moment alike.
    """
    return [
        leaf
        for leaf, keep in zip(jax.tree.leaves(tree), jax.tree.leaves(mask), strict=True)
        if keep
    ]


def diagnostic_scales(module: nnx.Module) -> dict[str, jax.Array]:
    """
    The scale diagnostics read off the module alone, keyed by layer path. They need no batch,
    so a caller can take them outside its gradient step and on its own cadence.

    ``noisy_sigma`` is mean |sigma| per NoisyLinear, and should fall as the policy sharpens.
    Sigma's sign is a symmetry of the parameterisation, so the series beside it tell learning
    from drift:

    - ``noisy_sigma_signed``: the signed mean. Flat while |sigma| climbs is a random walk.
    - ``noisy_sigma_neg_frac``: the fraction below zero, from 0 at init; near 0.5 is drift.
    - ``noisy_sigma_ratio``: mean |sigma| over mean |mu|. Flat while both grow is rescaling,
      not added noise.

    ``spectral_sigma`` is each SpectralNorm's estimate of its weight's largest singular value.

    ``weight_rms`` is each parameter's RMS; ``weights/global_norm`` is the L2 over the
    parameters ``unnormalised_param_mask`` keeps. Nothing but a decay term bounds their growth
    under Adam.
    """
    scales = dict[str, jax.Array]()
    for path, node in nnx.iter_graph(module):
        name = "/".join(map(str, path))
        if isinstance(node, NoisyLinear):
            sigma = node.kernel_sigma[...]
            abs_sigma = jnp.mean(jnp.abs(sigma))
            scales[f"noisy_sigma/{name}"] = abs_sigma
            scales[f"noisy_sigma_signed/{name}"] = jnp.mean(sigma)
            scales[f"noisy_sigma_neg_frac/{name}"] = jnp.mean(sigma < 0)
            scales[f"noisy_sigma_ratio/{name}"] = abs_sigma / jnp.mean(
                jnp.abs(node.kernel_mean[...])
            )
        elif isinstance(node, SpectralNorm):
            for key, stat in node.batch_stats.items():
                if key[-1] == "sigma":
                    weight = "/".join(map(str, key[:-1]))
                    scales[f"spectral_sigma/{name}/{weight}"] = jnp.asarray(stat[...])

    params = nnx.state(module, nnx.Param)
    for path, param in nnx.to_flat_state(params):
        name = "/".join(map(str, path))
        scales[f"weight_rms/{name}"] = jnp.sqrt((param[...] ** 2).mean())
    scales["weights/global_norm"] = optax.global_norm(
        masked_leaves(unnormalised_param_mask(module), params)
    )
    return scales


DORMANT_THRESHOLD = 0.025  # Sokar et al.'s tau, on the normalised activation score


def feature_scales(features: jax.Array, *, num_channels: int) -> dict[str, jax.Array]:
    """
    The plasticity diagnostics read off activations: the trunk features of train_step's online
    pass. A convolutional neuron is a whole feature map, so the flattened ``features`` are
    folded back into ``num_channels`` maps before anything is counted.

    - ``dormant_frac``: Sokar et al.'s dormant-neuron ratio. A neuron's score is its mean
      |activation| over batch and positions, over the layer's mean of that; dormant is a score
      at or below DORMANT_THRESHOLD.
    - ``dead_frac``: neurons exactly zero at every position across the batch.
    - ``feature_rms``: the representation's scale.
    - ``effective_rank``: the participation ratio ``(tr C)^2 / tr(C^2)`` of the centred feature
      covariance, in [1, min(batch - 1, units)], taken on the flattened features the head sees
      and computed through the [batch, batch] Gram matrix. Uncentred, the post-ReLU mean would
      swamp it. It depends on the batch as much as on the trunk, so compare it only within a run
      (docs/plasticity.md).
    """
    x = features.reshape(-1, features.shape[-1])
    # [batch, position, channel]: the trunk flattens position-major, channel-minor.
    maps = jnp.abs(x).reshape(x.shape[0], -1, num_channels)
    mean_abs = maps.mean((0, 1))
    score = mean_abs / (jnp.mean(mean_abs) + 1e-12)

    centred = x - x.mean(0)
    gram = centred @ centred.T
    trace = jnp.trace(gram)

    return {
        "plasticity/dormant_frac": jnp.mean(score <= DORMANT_THRESHOLD),
        "plasticity/dead_frac": jnp.mean(maps.max((0, 1)) == 0),
        "plasticity/feature_rms": jnp.sqrt(jnp.mean(jnp.square(x))),
        "plasticity/effective_rank": jnp.square(trace)
        / (jnp.sum(jnp.square(gram)) + 1e-12),
    }


def adam_optim_scales(
    opt: nnx.Optimizer[Any],
    module: nnx.Module,
) -> dict[str, jax.Array]:
    """
    Adam's view of the gradient per weight matrix, read off the optimizer state as it is when
    called. Biases are skipped.

    ``adam_snr`` is mean ``|m| / (sqrt(v) + eps)``: 1 for a gradient that points the same way
    every step, and ``sqrt((1 - b1) / (1 + b1)) * sqrt(2 / pi)``, about 0.18 at b1 = 0.9, for
    pure iid noise. A large eps also pulls it below that floor, on layers whose sqrt(v) sits
    under eps (docs/plasticity.md), so compare it against the floor only where the gradient
    clears eps, and never across layers.

    ``adam_step_rel`` is the learning rate times that, over the parameter's RMS: the fraction of
    its own scale a weight moves per step.

    Both are bias-corrected. Every number is read from the optimizer state by name -- moments
    from adam's state, settings from ``optax.inject_hyperparams`` -- so an optimizer that is not
    adam, or whose settings are not state, reports nothing.
    """
    adam = otu.tree_get(opt.opt_state, "ScaleByAdamState")
    b1 = otu.tree_get(opt.opt_state, "b1")
    b2 = otu.tree_get(opt.opt_state, "b2")
    eps = otu.tree_get(opt.opt_state, "eps")
    learning_rate = otu.tree_get(opt.opt_state, "learning_rate")
    if None in (adam, b1, b2, eps, learning_rate):
        return {}

    # At count == 0 both moments are zero; dividing by one there reads 0 rather than nan.
    steps = jnp.asarray(adam.count)
    bias_correction1 = jnp.where(steps == 0, 1.0, 1 - b1**steps)
    bias_correction2 = jnp.where(steps == 0, 1.0, 1 - b2**steps)

    nus = dict(nnx.to_flat_state(adam.nu))
    params = dict(nnx.to_flat_state(nnx.state(module, nnx.Param)))

    scales = dict[str, jax.Array]()
    for path, mu_leaf in nnx.to_flat_state(adam.mu):
        first_moment = mu_leaf[...]
        if first_moment.ndim < 2:
            continue
        second_moment = nus[path][...]
        snr = jnp.mean(
            jnp.abs(first_moment / bias_correction1)
            / (jnp.sqrt(second_moment / bias_correction2) + eps)
        )
        param_rms = jnp.sqrt((params[path][...] ** 2).mean())

        name = "/".join(map(str, path))
        scales[f"adam_snr/{name}"] = snr
        scales[f"adam_step_rel/{name}"] = learning_rate * snr / (param_rms + 1e-12)
    return scales
