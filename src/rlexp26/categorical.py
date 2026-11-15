"""Distributions over a grid of atoms: projections onto it and losses between them."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

# value/tail_mass is the probability on atoms beyond this magnitude.
TAIL_MASS_BEYOND = 1000.0


def sinh_atoms(num_atoms: int, asinh_bound: float) -> jax.Array:
    """
    ``num_atoms`` atoms at sinh(u), u uniform on [-``asinh_bound``, ``asinh_bound``], exactly
    mirror-symmetric: one half is the other negated, where ``jnp.linspace`` would round each
    side apart.
    """
    half = jnp.arange((num_atoms + 1) % 2, num_atoms, 2)
    half = jnp.sinh(asinh_bound * half / (num_atoms - 1))
    return jnp.concat((-half[num_atoms % 2 :][::-1], half))


def gaussian_log_probs(atoms: ArrayLike, mean: ArrayLike, std: ArrayLike) -> jax.Array:
    """
    Log-probabilities of N(``mean``, ``std``^2) on ``atoms``, ``mean`` and ``std`` being the
    leading axes: the density at each atom times the grid's spacing there, normalised. That is
    a quadrature, for a ``std`` no smaller than the spacing; on a mirror-symmetric grid it is
    exactly symmetric about a zero mean, where CDF differences are not.
    """
    atoms = jnp.asarray(atoms)
    spacing = jnp.gradient(atoms)
    assert isinstance(spacing, jax.Array)
    return jax.nn.log_softmax(
        jax.scipy.stats.norm.logpdf(
            atoms, jnp.asarray(mean)[..., None], jnp.asarray(std)[..., None]
        )
        + jnp.log(spacing),
        -1,
    )


def expectation(probs: ArrayLike, atoms: ArrayLike) -> jax.Array:
    """
    The mean of ``probs`` on ``atoms``, summed over pairs of atoms taken from the two ends
    inwards. On a mirror-symmetric grid a pair's atoms sum to exactly 0, so equal mass on both
    sides cancels exactly before it meets the outer atoms' magnitude.
    """
    probs = jnp.asarray(probs)
    atoms = jnp.asarray(atoms)
    half = atoms.size // 2
    low, high = atoms[:half], atoms[::-1][:half]
    low_probs, high_probs = probs[..., :half], probs[..., ::-1][..., :half]
    mean = (low_probs - high_probs) @ low + high_probs @ (low + high)
    if atoms.size % 2:
        mean += probs[..., half] * atoms[half]
    return mean


def project_cat(
    probs: ArrayLike, src_atoms: ArrayLike, dst_atoms: ArrayLike
) -> jax.Array:
    """
    ``probs`` on ``src_atoms`` moved onto ``dst_atoms``: each atom's mass is split between the
    two ``dst_atoms`` around it in proportion to its closeness to each, which keeps the mean.
    """
    dst_atoms = jnp.asarray(dst_atoms)
    assert dst_atoms.ndim == 1

    @partial(jnp.vectorize, signature="(n),(n)->(m)")
    def project(probs: jax.Array, src_atoms: jax.Array) -> jax.Array:
        src_atoms = jnp.clip(src_atoms, dst_atoms[0], dst_atoms[-1])
        upper = jnp.clip(
            jnp.searchsorted(
                dst_atoms, src_atoms, side="right", method="scan_unrolled"
            ),
            1,
            dst_atoms.size - 1,
        )
        w = (src_atoms - dst_atoms[upper - 1]) / (
            dst_atoms[upper] - dst_atoms[upper - 1]
        )
        return jnp.bincount(
            upper - 1, probs * (1.0 - w), length=dst_atoms.size
        ) + jnp.bincount(upper, probs * w, length=dst_atoms.size)

    return project(jnp.asarray(probs), jnp.asarray(src_atoms))  # type: ignore[no-any-return]


def two_hot(x: ArrayLike, atoms: ArrayLike) -> jax.Array:
    """``project_cat`` of a point mass at ``x``."""
    x = jnp.asarray(x)[..., None]
    return project_cat(jnp.ones_like(x), x, atoms)


def project_moment_matched(
    probs: ArrayLike,
    src_atoms: ArrayLike,
    dst_atoms: ArrayLike,
    *,
    std_scale: float = 1.0,
    steps: int = 2,
) -> jax.Array:
    """
    ``project_cat`` of ``src_atoms`` scaled about their mean by a factor c, so that the
    projection keeps their mean and ``std_scale`` times their standard deviation. A two-hot split
    adds (x - lower)(upper - x) to its point's variance; c solves c^2 Var + E[split] =
    std_scale^2 Var in ``steps`` fixed-point steps from ``std_scale``, and is 0 where nothing
    smaller than the splits' own variance is reachable.
    """
    probs = jnp.asarray(probs)
    src_atoms = jnp.asarray(src_atoms)
    dst_atoms = jnp.asarray(dst_atoms)
    mean = jnp.sum(probs * src_atoms, -1, keepdims=True)
    deviations = src_atoms - mean
    variance = jnp.sum(probs * jnp.square(deviations), -1, keepdims=True)
    scale = jnp.full_like(mean, std_scale)
    for _ in range(steps):
        points = jnp.clip(mean + scale * deviations, dst_atoms[0], dst_atoms[-1])
        upper = jnp.clip(
            jnp.searchsorted(dst_atoms, points, side="right", method="scan_unrolled"),
            1,
            dst_atoms.size - 1,
        )
        split = jnp.sum(
            probs * (points - dst_atoms[upper - 1]) * (dst_atoms[upper] - points),
            -1,
            keepdims=True,
        )
        scale = jnp.sqrt(
            jnp.maximum(std_scale**2 * variance - split, 0.0)
            / jnp.maximum(variance, np.finfo(np.float32).tiny)
        )
    return project_cat(probs, mean + scale * deviations, dst_atoms)


def floor_probs(probs: ArrayLike, floor: float) -> jax.Array:
    """``probs`` less ``floor``, clipped at 0 and renormalised over the last axis."""
    probs = jnp.maximum(jnp.asarray(probs) - floor, 0.0)
    return probs / jnp.sum(probs, -1, keepdims=True)


def cdf_bce(
    log_probs: ArrayLike, target_probs: ArrayLike, atoms: ArrayLike
) -> jax.Array:
    """
    CDF-BCE per entry: the binary cross-entropy between the target's and the prediction's CDFs
    at every threshold between neighbouring ``atoms``, weighted by the spacing there.
    Each CDF and its complement are summed from their own ends, so that the tails, where leaked
    mass sits, stay exact in float32.
    """
    log_probs = jnp.asarray(log_probs)
    target_probs = jnp.asarray(target_probs)
    atoms = jnp.asarray(atoms)
    last = log_probs.ndim - 1
    logit_cdf = (
        jax.lax.cumlogsumexp(log_probs, axis=last)[..., :-1]
        - jax.lax.cumlogsumexp(log_probs, axis=last, reverse=True)[..., 1:]
    )
    below = jax.lax.cumsum(target_probs, axis=last)[..., :-1]
    above = jax.lax.cumsum(target_probs, axis=last, reverse=True)[..., 1:]
    bce = -below * jax.nn.log_sigmoid(logit_cdf) - above * jax.nn.log_sigmoid(
        -logit_cdf
    )
    return bce @ jnp.diff(atoms)


def relative_cdf_bce(
    log_probs: ArrayLike, target_probs: ArrayLike, atoms: ArrayLike, floor: float
) -> jax.Array:
    """
    ``cdf_bce`` over the prediction's ``sg[E_p |a|]``, read off its probabilities floored at
    ``floor`` and itself floored at the finest spacing.
    """
    atoms = jnp.asarray(atoms)
    scale = floor_probs(jax.lax.stop_gradient(jnp.exp(log_probs)), floor) @ jnp.abs(
        atoms
    )
    return cdf_bce(log_probs, target_probs, atoms) / jnp.maximum(
        scale, jnp.diff(atoms).min()
    )


@dataclass(frozen=True)
class CategoricalReturns:
    """
    The return distributions of a net that answers log-probabilities ``[batch, actions,
    atoms]`` over mirror-symmetric ``atoms``. Every probability read off it, and every target,
    is floored at ``floor``. The target is the moment-matched projection of the backed-up
    mixture, keeping ``target_std_scale`` of its standard deviation, and the loss is the
    cross-entropy against it plus ``cdf_bce_weight`` times the relative CDF-BCE. A nonzero
    ``floor_distill_weight`` adds that factor times the mean, over the actions not taken, of
    the CDF-BCE against their own prediction floored at ``floor_distill_floor``, relative to
    that floored prediction's E|a|. A ``ReturnDistribution``.
    """

    atoms: jax.Array
    floor: float
    target_std_scale: float
    target_moment_steps: int
    cdf_bce_weight: float
    floor_distill_weight: float = 0.0
    floor_distill_floor: float = 0.0

    def forward(
        self,
        net: Callable[..., tuple[jax.Array, eqx.nn.State]],
        obs: jax.Array,
        state: eqx.nn.State,
        *,
        key: jax.Array,
        taken: jax.Array | None = None,
    ) -> tuple[jax.Array, eqx.nn.State]:
        return net(obs, state, key=key, taken=taken)

    def means(self, out: jax.Array) -> jax.Array:
        return expectation(floor_probs(jnp.exp(out), self.floor), self.atoms)

    def loss(
        self,
        out: jax.Array,
        actions: jax.Array,
        next_out: jax.Array,
        next_policy: jax.Array,
        shift: jax.Array,
        scale: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        rows = jnp.arange(actions.size)
        action_log_probs = out[rows, actions]
        mixture = jnp.sum(
            next_policy[..., None] * floor_probs(jnp.exp(next_out), self.floor), -2
        )
        # Floored before the projection too: sub-floor mass on the outer atoms, squared, would
        # set the variance.
        targets = floor_probs(
            project_moment_matched(
                floor_probs(mixture, self.floor),
                shift[:, None] + scale[:, None] * self.atoms,
                self.atoms,
                std_scale=self.target_std_scale,
                steps=self.target_moment_steps,
            ),
            self.floor,
        )
        cross_entropy = -jnp.sum(targets * action_log_probs, -1)
        target_cdf_bce = relative_cdf_bce(
            action_log_probs, targets, self.atoms, self.floor
        )
        losses = cross_entropy + self.cdf_bce_weight * target_cdf_bce
        action_probs = floor_probs(jnp.exp(action_log_probs), self.floor)
        action_means = expectation(action_probs, self.atoms)
        stats = {
            "loss/cross_entropy": cross_entropy,
            "loss/cdf_bce": target_cdf_bce,
            "value/return_std": jnp.sqrt(
                jnp.sum(
                    action_probs * jnp.square(self.atoms - action_means[:, None]), -1
                )
            ),
            # The worst action's, unfloored.
            "value/tail_mass": jnp.sum(
                jnp.exp(out) * (jnp.abs(self.atoms) > TAIL_MASS_BEYOND), -1
            ).max(-1),
        }
        if self.floor_distill_weight:
            distill_targets = jax.lax.stop_gradient(
                floor_probs(jnp.exp(out), self.floor_distill_floor)
            )
            distill = cdf_bce(out, distill_targets, self.atoms) / jnp.maximum(
                distill_targets @ jnp.abs(self.atoms), jnp.diff(self.atoms).min()
            )
            untaken = jnp.arange(out.shape[-2]) != actions[:, None]
            distill_loss = jnp.sum(distill * untaken, -1) / jnp.sum(untaken, -1)
            losses += self.floor_distill_weight * distill_loss
            stats["loss/floor_distill_cdf_bce"] = distill_loss
        td_error = expectation(targets, self.atoms) - action_means
        return losses, td_error, jnp.abs(td_error), stats
