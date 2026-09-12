"""
Diagnostics that read a module, an optimizer or a batch of activations and return TensorBoard
scalars. Nothing here knows about BTR: the hyperparameters they need are arguments, and the
networks they walk are ``nnx.Module``, so ``btr`` imports this and not the other way round.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from flax import nnx

from .internals import SpectralNorm
from .layers import NoisyLinear


def diagnostic_scales(module: nnx.Module) -> dict[str, jax.Array]:
    """
    The scale diagnostics that have to be read off the module tree, keyed by layer path. All of
    them are reductions over parameters already on the device, so they cost nothing against
    train_step's ~598 GFLOP and are computed every step; the loop drains them every
    TRAIN_LOG_FREQ.

    ``noisy_sigma`` is mean |sigma| per NoisyLinear, the standard check that NoisyNets is alive.
    It is expected to *fall* as the policy sharpens, and the three series beside it exist to say
    what it means when it does not:

    - ``noisy_sigma_signed``, the mean with its sign. Sigma is initialised all-positive and its
      sign is a pure symmetry of the parameterisation -- the noise it scales is symmetric, so
      nothing anchors it -- which means an unbiased random walk driven by Adam raises mean
      |sigma| while leaving this one near its starting value. Signed tracking absolute is real
      pressure towards noisier weights; signed flat while absolute climbs is drift.
    - ``noisy_sigma_neg_frac``, the fraction of sigma entries below zero. Starts at exactly 0.
      Anything approaching 0.5 is the same drift story, told without the cancellation.
    - ``noisy_sigma_ratio``, mean |sigma| over mean |mu| for the same kernel: the perturbation
      relative to the weight it perturbs. Flat while both inflate means the layer is being
      rescaled, not made noisier, and it is the series to read against ``weight_rms``.

    ``spectral_sigma`` is each SpectralNorm's estimate of its weight's largest singular value --
    the layer's Lipschitz constant before normalisation -- read from the batch stats that
    train_step's forward pass just updated.

    ``weight_rms`` is per-parameter RMS and ``weights/global_norm`` the L2 over all of them.
    Nothing in this run bounds parameter growth -- no weight decay, and Adam's update size is
    scale-free -- so these are the plasticity-loss proxy the backlog's reset experiments would
    be measured against, and the place to see whether inflation is the trunk or the head.
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
    scales["weights/global_norm"] = optax.global_norm(params)
    return scales


DORMANT_THRESHOLD = 0.025  # Sokar et al.'s tau, on the normalised activation score


def feature_scales(features: jax.Array, *, num_channels: int) -> dict[str, jax.Array]:
    """
    The plasticity diagnostics that need activations rather than parameters, read off the trunk
    features of train_step's online pass -- one batch of replay states through the network that
    is actually learning.

    A convolutional neuron is a whole feature map, so ``features`` arrives flattened -- the trunk
    folds ``num_channels`` maps together with the positions it pooled them to -- and is folded
    back before anything is counted. Scoring the (position, channel) pairs instead counts every
    position a channel happens not to fire at, which on Breakout is most of them for any channel
    watching one part of the screen: it reads ~2.4x high at initialisation and is comparable to
    no published figure.

    - ``dormant_frac``: Sokar et al.'s dormant-neuron ratio. A neuron's score is its mean
      |activation| over the batch and over its positions, divided by the layer's mean of that, so
      it is scale-free; dormant means a score at or below DORMANT_THRESHOLD, i.e. a neuron
      carrying ~2.5% of the average neuron's signal. This is the headline plasticity number and
      the one the backlog has been missing.
    - ``dead_frac``: the harder version, neurons whose activation is *exactly* zero at every
      position across the whole batch. A ReLU that has died rather than merely gone quiet.
    - ``feature_rms``: representation scale, the thing dormancy is measured relative to.
    - ``effective_rank``: the participation ratio ``(tr C)^2 / tr(C^2)`` of the feature second
      moment, a soft rank in [1, min(batch, units)] -- how many directions the representation
      actually uses. Measured on the flattened features rather than the folded ones, since this
      is the representation the head is handed. Computed through the [batch, batch] Gram matrix,
      which has the same non-zero eigenvalues as the [units, units] covariance and is ~300 MFLOP
      against train_step's ~598 GFLOP, so no SVD and no measurable cost. Collapsing rank at flat
      dormancy is capacity loss that dead-unit counting alone would miss.
    """
    x = features.reshape(-1, features.shape[-1])
    # [batch, position, channel]: the trunk flattens position-major, channel-minor.
    maps = jnp.abs(x).reshape(x.shape[0], -1, num_channels)
    mean_abs = maps.mean((0, 1))
    score = mean_abs / (jnp.mean(mean_abs) + 1e-12)

    gram = x @ x.T
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
    Adam's own view of the gradient, per weight matrix, read off the optimizer state that is
    already there -- nothing is stored for this and the checkpoint is unaffected.

    ``adam_snr`` is mean ``|m| / (sqrt(v) + eps)``, how consistent the gradient has been in
    sign and size. It has a known floor: for gradients that are pure iid noise the EMA gives
    ``E|m| / sqrt(v) = sqrt((1 - b1) / (1 + b1)) * sqrt(2 / pi)``, about **0.18** at b1 = 0.9,
    while a gradient pointing the same way every step drives it to 1. So a series sitting near
    0.18 is a parameter Adam is random-walking, and one near 1 is a parameter being driven.
    That is the direct test of whether a rising ``noisy_sigma`` is signal or drift, and it is
    worth reading per layer: a strong SNR on one NoisyLinear and a floor-level one on another
    says the two are doing different things, not that the diagnostic is broken.

    ``adam_step_rel`` multiplies that by the learning rate and divides by the parameter's own
    RMS: the fraction of its own scale a weight moves per step. Decaying towards zero is the
    plasticity failure proper -- a network that can no longer move regardless of gradient.

    Both undo Adam's bias correction, so they read true from the first step rather than opening
    at ~3 and decaying while the moments warm up. Biases are skipped -- the interesting dynamics
    are in the matrices, and one series per tensor is enough already.

    Every number comes out of the optimizer by name: the moments from adam's own state, the
    settings from the hyperparameters ``optax.inject_hyperparams`` keeps beside them. So no
    position in the optax chain is named, nothing restates a constant the optimizer already
    holds, and an optimizer that is not adam -- or one whose settings are closure constants
    rather than state -- reports nothing instead of reporting the wrong thing.
    """
    adam = otu.tree_get(opt.opt_state, "ScaleByAdamState")
    b1 = otu.tree_get(opt.opt_state, "b1")
    b2 = otu.tree_get(opt.opt_state, "b2")
    eps = otu.tree_get(opt.opt_state, "eps")
    learning_rate = otu.tree_get(opt.opt_state, "learning_rate")
    if None in (adam, b1, b2, eps, learning_rate):
        return {}

    # Adam's own bias correction. Without it both moments read low, but nu far more so at
    # b2 = 0.999, and the ratio opens at ~3 and decays -- warm-up masquerading as signal.
    # The state is read before opt.update, so the first call sees count == 0, where both
    # corrections are zero and both moments are exactly zero with them. Divide by one there and
    # the series opens at a clean 0 rather than a nan.
    steps = jnp.asarray(adam.count)
    bias_correction1 = jnp.where(steps == 0, 1.0, 1 - b1**steps)
    bias_correction2 = jnp.where(steps == 0, 1.0, 1 - b2**steps)

    nus = dict(nnx.to_flat_state(adam.nu))
    params = dict(nnx.to_flat_state(nnx.state(module, nnx.Param)))

    scales = dict[str, jax.Array]()
    for path, mu_leaf in nnx.to_flat_state(adam.mu):
        first_moment = mu_leaf[...]
        if first_moment.ndim < 2:  # biases: one series per matrix is enough
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
