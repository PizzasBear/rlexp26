"""
Losses and diagnostics: arithmetic that reads a module, an optimizer, a batch of activations or
a batch of errors. Nothing here knows about BTR -- the hyperparameters they need are arguments,
and the networks they walk are ``nnx.Module`` -- so ``btr`` imports this and not the other way
round.
"""

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
    ``optax.huber_loss`` divided by ``delta``, which is the form the quantile-Huber uses, and
    unlike the builtin it is defined at ``delta = 0``, where it is ``|errors|`` and the
    quantile-Huber becomes the pinball loss. The builtin cannot reach that: it is identically
    zero there, so dividing its scaling back off is 0 / 0.

    At ``delta = 0`` and ``errors = 0`` this reports a derivative of 0.5 rather than 0, since
    ``jnp.abs`` reads 1 at zero and ``jnp.minimum`` splits its tie. Both are subgradients of
    ``|errors|`` there and neither is wrong; on the knee itself, where the tie also splits, the
    two halves cancel and the slope is 1 as it should be.
    """
    abs_errors = jnp.abs(errors)
    quadratic = jnp.minimum(abs_errors, delta)
    return abs_errors - quadratic + jnp.square(quadratic) / (2 * delta + eps)


def unnormalised_param_mask(module: nnx.Module) -> nnx.State[Any, Any]:
    """
    ``module``'s ``nnx.Param`` tree with a boolean at every leaf: true for the parameters that
    are not a normalisation layer's scale or bias, false for the ones that are. It mirrors the
    tree's structure rather than listing paths, which is what ``optax.adamw``'s ``mask`` takes.

    Two things want that split. A decay term is one: after normalisation the gain is the only
    thing carrying a layer's output scale, so decaying it does not shrink the layer so much as
    move its scale downstream, where under Adam's scale-free step nothing pulls back -- the
    growth the decay was added to stop. Biases are a different case and stay decayed: nothing
    else bounds them, and they grow by one to two orders of magnitude over a full run here.

    Reading a global norm is the other. A gain starts at one and its layer's invariance holds it
    near there, so it says nothing about growth while contributing an L2 that goes with the root
    of its element count -- normalised per position, an Impala trunk's gains are a near-constant
    pedestal ten times the L2 of every other parameter put together. Drift in the gains is
    per-layer in ``weight_rms``, which is where it is legible.
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
        # as_pure, because nnx.Optimizer unwraps the Variables before this reaches optax and a
        # mask has to match the tree optax is handed, not the one the module holds.
        nnx.as_pure(nnx.state(module, nnx.Param)),
    )
    return mask


def masked_leaves(mask: Any, tree: Any) -> list[jax.Array]:
    """
    The leaves of ``tree`` whose matching leaf in ``mask`` is true, flat, to take a norm over.
    One mask serves a parameter tree, a gradient of one and an Adam moment alike, since all
    three have the structure it was built against.
    """
    return [
        leaf
        for leaf, keep in zip(jax.tree.leaves(tree), jax.tree.leaves(mask), strict=True)
        if keep
    ]


def diagnostic_scales(module: nnx.Module) -> dict[str, jax.Array]:
    """
    The scale diagnostics that have to be read off the module tree, keyed by layer path. They
    need no batch and no gradient, only the module, so a caller is free to take them outside its
    gradient step and on whatever cadence it likes; ``btr.BTR.learn_step`` does both. They are
    reductions over arrays already on the device, so what that cadence buys is host launches
    rather than device time.

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

    ``weight_rms`` is per-parameter RMS, over every parameter; ``weights/global_norm`` is the
    L2 over the ones ``unnormalised_param_mask`` keeps, which is all of them for a module with no
    normalisation layer and so the same series a run predating one can be compared against.
    Adam's update size is scale-free, so short of a decay term in the caller's optimizer nothing
    bounds parameter growth: these are the plasticity-loss proxy a decay or reset experiment is
    measured against, and the place to see whether inflation is the trunk or the head.
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
    - ``effective_rank``: the participation ratio ``(tr C)^2 / tr(C^2)`` of the feature
      covariance, a soft rank in [1, min(batch - 1, units)] -- how many directions the
      representation actually uses, one short of the batch because centring costs a degree of
      freedom. The batch mean comes off first, and has to: the features are post-ReLU and
      non-negative, so every row of the batch has a large positive projection onto the same
      direction, and an uncentred second moment has one eigenvalue that swamps the rest. That
      version reads ~1 on a freshly initialised network and saturates around 9 whatever the true
      rank is, so it measures the mean rather than the representation. Measured on the flattened
      features rather than the folded ones, since this is the representation the head is handed.
      Computed through the [batch, batch] Gram matrix, which has the same non-zero eigenvalues as
      the [units, units] covariance and is ~300 MFLOP against train_step's ~598 GFLOP, so no SVD
      and no measurable cost. Collapsing rank at flat dormancy is capacity loss that dead-unit
      counting alone would miss.

      Read it as a property of the trunk *and* the batch, never of the trunk alone, and compare
      only within a run. On 256 Breakout observations the batch is what binds: states from a
      random policy read ~5.5 whether the trunk is trained or freshly initialised, states the
      trained policy visits read 11.7 through a fresh trunk and 23.4 through the trained one,
      and only synthetic uniform noise gets near the 255 ceiling -- 191 fresh, 140 trained. So a
      low absolute number is the input manifold, a *fall* against a fixed batch is the capacity
      loss, and training raising the number on its own state distribution is the healthy case.
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

    Two things move a series *below* that floor, and only one of them is about the gradient.
    Adam's eps is in the denominator, and BTR's ``0.005 / batch`` is ~2e-5 -- four thousand times
    optax's default and large enough to be the denominator outright on a wide layer whose
    gradient is spread thin. It is: at the end of a full run 49% of ``advantage_linear0``'s
    elements have ``sqrt(v)`` below eps and 95% below 10 eps, which reads 0.067 where dropping
    eps reads 0.109; the trunk's convolutions have no element anywhere near it and read the
    same either way. So compare this series against the floor only for layers whose gradient
    scale clears eps, and across layers not at all. What is left after that is the real reading:
    a gradient whose sign alternates, or one intermittent enough that v's 1000-step memory holds
    spikes m's 10-step memory has already forgotten. Both say Adam has no persistent direction
    to follow, and neither is distinguishable from these series alone.

    ``adam_step_rel`` multiplies that by the learning rate and divides by the parameter's own
    RMS: the fraction of its own scale a weight moves per step. Decaying towards zero is the
    plasticity failure proper -- a network that can no longer move regardless of gradient.

    Both undo Adam's bias correction, so they read true from the first step rather than opening
    at ~3 and decaying while the moments warm up. Biases are skipped -- the interesting dynamics
    are in the matrices, and one series per tensor is enough already.

    Whether the state read is the one before or after the step's update is the caller's to
    choose; this reports whatever the optimizer holds when called.

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
    # An optimizer that has not stepped yet has count == 0, where both corrections are zero and
    # both moments are exactly zero with them. Divide by one there and the series reads a clean
    # 0 rather than a nan.
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
