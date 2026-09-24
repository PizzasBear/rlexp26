import math
from collections import deque
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any, NamedTuple, override

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from flax import nnx
from jax.typing import ArrayLike

from expreplay import ReplayBuffer

from .agent import Agent, EnvStep, Policy
from .layers import (
    ImpalaCNNLarge,
    IQNCosineEmbedding,
    NoisyLinear,
)
from .utils import (
    adam_optim_scales,
    diagnostic_scales,
    feature_scales,
    masked_leaves,
    scaled_huber_loss,
    unnormalised_param_mask,
)

# Hyperparameters
# Gradient steps per environment step, over all environments at once: 1/64 at 64 environments is
# one gradient step an iteration, BTR's rr = 1. A Fraction so the remainder learn_step carries
# stays exact.
REPLAY_RATIO = Fraction(1, 64)
BATCH_SIZE = 256
BUFFER_SIZE = 1 << 20
TRAIN_START_BUF_SIZE = 200_000

PER_ALPHA = 0.2
# Sampling frequency times IS weight goes as p ** (PER_ALPHA * (1 - PER_BETA)), so at 1.0
# prioritisation drops out of the expected gradient.
PER_BETA = 0.2
PER_EPSILON = 1e-6
# Batches whose priorities may stay in flight beyond the ones an iteration dispatches itself; see
# BTR.init and BTR.observe.
PRIO_QUEUE_SLACK = 2

LEARNING_RATE = 1e-4
ADAM_EPS = 0.005 / BATCH_SIZE
ADAM_B1 = 0.9
ADAM_B2 = 0.999
WEIGHT_DECAY = 0.0
DISCOUNT = 0.997
N_STEP = 3

IQN_TRAIN_SAMPLES = 8
IQN_ACT_SAMPLES = 32  # BTR: 8
IQN_NUM_COS = 64
# The Huber knee, in units of TD error. Below it the pinball-weighted loss is minimised by the
# tau-expectile, above it by the tau-quantile, so this picks which statistic the head fits.
IQN_HUBER_LOSS_K = 1.0
# Applied before Adam, so it bounds what reaches Adam's moments.
GRADIENT_CLIPPING_MAX_NORM = 10

INFERENCE_SYNC_FREQ = 3
TARGET_NETWORK_UPDATE_FREQ = 500
LAYER_NORM = False
IMPALA_SIZE_FACTOR = 2
# The trunk convolutions' compute dtype; see ImpalaCNNLarge. A string so that ``hyperparameters``
# records it.
IMPALA_DTYPE = "bfloat16"  # BTR: float32

MUNCHAUSEN_TEMPERATURE = 0.03
MUNCHAUSEN_SCALING_TERM = 0.9
MUNCHAUSEN_CLIPPING_VAL = -1.0

# Act by sampling softmax(Q / ACT_TEMPERATURE) instead of the argmax. ACT_TEMPERATURE is read by
# _act and train/regret only; the loss uses MUNCHAUSEN_TEMPERATURE. Munchausen's gap amplification
# cancels its own (1 - alpha), so softmax(q / MUNCHAUSEN_TEMPERATURE) is already the soft-optimal
# policy: dividing by (1 - alpha) * tau acts 10x sharper (docs/behaviour-policy.md).
ACT_USE_SOFT_POLICY = True  # BTR: False
ACT_TEMPERATURE = MUNCHAUSEN_TEMPERATURE

# Geometric decay, see epsilon_at, and off from EPS_GREEDY_OFF_FRAMES.
EPS_GREEDY_START = 0.0  # BTR: 1.0
EPS_GREEDY_END = 0.0  # BTR: 0.01
EPS_GREEDY_DECAY_FRAMES = 8000_000
EPS_GREEDY_OFF_FRAMES = 100_000_000
EVAL_EPS_GREEDY = 0.0  # BTR: 0.01
EVAL_EPS_GREEDY_OFF_FRAMES = 125_000_000

# Gradient steps between writes: _train_step's scalars are transferred at TRAIN_LOG_FREQ, and
# _scale_stats runs at SCALE_LOG_FREQ.
TRAIN_LOG_FREQ = 100
SCALE_LOG_FREQ = 2500

# Every constant above, and the architecture below, checked against BTR (arXiv:2411.03820,
# Table D6 and Appendices E/H) and its reference implementation (github.com/VIPTankz/BTR) on
# 2026-09-12; everything matches bar what is marked. Table D6 gives PER only an alpha, so
# PER_BETA and PER_EPSILON come from the reference's PER.py: its eps is 1e-6, and its IS weights
# are ``(capacity * prob) ** -self.alpha`` -- alpha where beta belongs, so an effective beta of
# 0.2. docs/reproduction.md has the score comparison.

# +--------------+
# | AI Generated |
# +--------------+

# === Correctness to settle ===
# TODO: NoisyNets' sigma drifts rather than being learned (docs/plasticity.md). Candidates:
#       weight decay on the sigma parameters, or BTR's one noise draw per gradient step in
#       place of NoisyLinear's per-call redraw.

# === Missing core pieces ===
# TODO: decide whether resuming onto an empty replay buffer is acceptable. A checkpoint does not
#       carry the buffer, so a resumed run collects TRAIN_START_BUF_SIZE transitions without
#       learning, all of them from an already-trained policy.

# === Performance ===
# TODO: donate_argnums on the jitted steps to avoid param copies, leaving out the nets
#       sync_qnet aliases.
# TODO: the head and the loss are still float32. They are ~1/6 of the trunk's FLOPs, and the
#       quantile axes make the loss where precision is least obviously free. Measure first.

# === Reproduction / tuning ===
# TODO: train/regret reads 0 for a greedy run, since the NoisyNet draw's own cost needs a
#       reference pass with the noise zeroed. Only the head carries noise, so only the head
#       would re-run.

# === Experimental / longer term ===
# TODO: LayerNorm per BTR's Appendix H. Does it make SpectralNorm redundant? Try LN-only,
#       SN-only, both.
# TODO: plasticity. The pathology is growing scale collapsing the effective learning rate, not
#       dormancy or rank (docs/plasticity.md). Weight decay and projection below act on it;
#       resets address what none of the diagnostics show.
# TODO: weight decay (WEIGHT_DECAY, wired at 0.0), or XQC's per-step projection of each weight
#       onto the unit sphere. Projection is legal everywhere but value_linear1 and
#       advantage_linear1, which carry the return's scale (docs/plasticity.md). Decaying
#       quantile_embedding annealed IQN's tau modulation in the LN + WD run.
# TODO: IQN_HUBER_LOSS_K. At 1.0 the head fits expectiles, biased upward on a skewed return, and
#       the fitted statistic drifts as |td| moves over a run (docs/plasticity.md). Sweep
#       {1, 0.3, 0.1, 0}, or set the knee from the batch's own |td|. kappa = 0 is QR-DQN-0;
#       expect more clipping, and read adam_snr after changing it.
# TODO: ablate GRADIENT_CLIPPING_MAX_NORM, reading adam_snr and adam_step_rel beside the score:
#       the clip engages often enough that it may be what holds adam_snr where it is.
# TODO: XQC's components after LayerNorm, in order. (1) BatchRenorm in place of LN, without
#       CrossQ's joined forward pass, which buys nothing here (docs/plasticity.md). (2) The
#       categorical CE loss, which means replacing IQN with C51; the quantile Huber already
#       bounds dL/dy_hat, so only XQC's Hessian argument carries over.
# TODO: resets, at a higher replay ratio, whose cost is linear (docs/performance.md). BBF's 40k
#       cadence is ~21 resets over a 50M-step run.
# TODO: exploration beyond NoisyNets, once a clean baseline reproduces. NGU needs an R2D2
#       backbone and BYOL-Explore an RNN world model; only NGU's episodic bonus stands alone.
# TODO: sanity-check the whole stack on LunarLander (discrete) first; Atari runs are too slow a
#       feedback loop for debugging.
# TODO: a tiny env + tiny net regression test that runs in seconds.


def create_rngs(seed: int) -> nnx.Rngs:
    """
    One RNG stream per concern, so that drawing more from one does not shift the others. No
    ``default``, so an unnamed stream raises instead of aliasing one of these.
    """
    return nnx.Rngs(
        params=seed, noise=seed + 1000, samples=seed + 2000, explore=seed + 3000
    )


class QNet(nnx.Module):
    def __init__(
        self,
        num_actions: int,
        *,
        obs_shape: tuple[int, int, int],
        layer_norm: bool,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_actions = num_actions

        obs_stack, height, width = obs_shape
        self.decoder = ImpalaCNNLarge(
            obs_stack,
            in_spatial=(height, width),
            size_factor=IMPALA_SIZE_FACTOR,
            dtype=IMPALA_DTYPE,
            layer_norm=layer_norm,
            rngs=rngs,
        )
        self.quantile_embedding = IQNCosineEmbedding(
            self.decoder.out_features, num_cosines=IQN_NUM_COS, rngs=rngs
        )
        # Normalised before the hidden activation only: the output layer carries the return's
        # scale.
        self.value_linear0 = NoisyLinear(self.decoder.out_features, 512, rngs=rngs)
        self.value_layer_norm0: nnx.LayerNorm | None = nnx.data(None)
        if layer_norm:
            self.value_layer_norm0 = nnx.LayerNorm(512, rngs=rngs)
        self.value_linear1 = NoisyLinear(512, 1, rngs=rngs)

        self.advantage_linear0 = NoisyLinear(self.decoder.out_features, 512, rngs=rngs)
        self.advantage_layer_norm0: nnx.LayerNorm | None = nnx.data(None)
        if layer_norm:
            self.advantage_layer_norm0 = nnx.LayerNorm(512, rngs=rngs)
        self.advantage_linear1 = NoisyLinear(512, num_actions, rngs=rngs)

    def _value(self, x: ArrayLike, *, rngs: nnx.Rngs) -> jax.Array:
        x = self.value_linear0(x, rngs=rngs)
        if self.value_layer_norm0 is not None:
            x = self.value_layer_norm0(x)
        x = nnx.relu(x)
        x = self.value_linear1(x, rngs=rngs)
        return x

    def _advantage(self, x: ArrayLike, *, rngs: nnx.Rngs) -> jax.Array:
        x = self.advantage_linear0(x, rngs=rngs)
        if self.advantage_layer_norm0 is not None:
            x = self.advantage_layer_norm0(x)
        x = nnx.relu(x)
        x = self.advantage_linear1(x, rngs=rngs)
        return x

    def __call__(
        self,
        obs: ArrayLike,
        *,
        samples: ArrayLike,
        rngs: nnx.Rngs,
        advantage_only: bool = False,
    ) -> jax.Array:
        x = self.decoder(obs).astype(jnp.float32)
        x = x[..., None, :] * self.quantile_embedding(samples)

        a = self._advantage(x, rngs=rngs)
        v = self._value(x, rngs=rngs) if not advantage_only else 0
        return v + a - a.mean(-1, keepdims=True)

    def random_n_samples_mean(
        self,
        obs: ArrayLike,
        num_samples: int,
        *,
        rngs: nnx.Rngs,
        advantage_only: bool = False,
    ) -> jax.Array:
        obs = jnp.asarray(obs)
        samples = rngs.samples.uniform((*obs.shape[:-3], num_samples))
        return self(
            obs, samples=samples, rngs=rngs, advantage_only=advantage_only
        ).mean(-2)


def norm_obs(obs: ArrayLike) -> jax.Array:
    obs = jnp.moveaxis(obs, -3, -1)
    return jnp.astype(obs, jnp.float32) / 255


def epsilon_at(num_frames: int) -> float:
    """
    Training epsilon after ``num_frames`` ALE frames: geometric decay onto EPS_GREEDY_END with
    EPS_GREEDY_DECAY_FRAMES as the 1/e time constant, the continuous limit of BTR's
    ``EpsilonGreedy.update_eps``. Table D6's "Decay: 8M Frames" is not an anneal length.
    """
    if num_frames >= EPS_GREEDY_OFF_FRAMES:
        return 0.0
    decayed = math.exp(-num_frames / EPS_GREEDY_DECAY_FRAMES)
    return EPS_GREEDY_END + decayed * (EPS_GREEDY_START - EPS_GREEDY_END)


# num_samples sizes the quantile draw, so it has to be static.
@nnx.jit(static_argnames=("num_samples", "soft_sampling"))
def _act(
    qnet: QNet,
    obs: ArrayLike,
    *,
    rngs: nnx.Rngs,
    epsilon: float = 0.0,
    num_samples: int = IQN_ACT_SAMPLES,
    soft_sampling: bool = ACT_USE_SOFT_POLICY,
    temperature: float = ACT_TEMPERATURE,
) -> tuple[jax.Array, jax.Array]:
    """
    One action per environment -- sampled from ``softmax(Q / temperature)`` if ``soft_sampling``,
    the argmax otherwise -- under NoisyNets and an epsilon-greedy override, and the taken action's
    log-probability under that behaviour policy.

    The log-probability is conditional on this call's NoisyNet draw, so it understates how
    stochastic acting is; so does train_step's ``policy_entropy``. Without soft sampling it
    depends on nothing but epsilon and the action count.
    """
    obs = norm_obs(obs)

    # Advantages suffice: softmax and argmax are both shift invariant.
    advantages = qnet.random_n_samples_mean(
        obs, num_samples, advantage_only=True, rngs=rngs
    )
    num_actions = advantages.shape[-1]

    if soft_sampling:
        policy_log_probs = nnx.log_softmax(advantages / temperature)
        policy_actions = rngs.explore.categorical(policy_log_probs)
    else:
        policy_actions = jnp.argmax(advantages, -1)
        policy_log_probs = jnp.where(
            policy_actions[..., None] == jnp.arange(num_actions), 0, -jnp.inf
        )

    choose_random_action = rngs.explore.uniform(policy_actions.shape) < epsilon
    random_actions = rngs.explore.randint(
        policy_actions.shape, 0, num_actions, policy_actions.dtype
    )
    actions = jnp.where(choose_random_action, random_actions, policy_actions)

    log_probs = jnp.logaddexp(
        jnp.log1p(-epsilon) + policy_log_probs, jnp.log(epsilon / num_actions)
    )

    log_probs = jnp.take_along_axis(log_probs, actions[..., None], -1).squeeze(-1)
    return actions, log_probs


@nnx.jit(static_argnames=("num_samples", "soft_sampling"))
def _train_step(
    qnet: QNet,
    target_qnet: QNet,
    opt: nnx.Optimizer[QNet],
    rngs: nnx.Rngs,
    *,
    sample_prios: ArrayLike,
    obs: ArrayLike,
    actions: ArrayLike,
    rewards: ArrayLike,
    dones: ArrayLike,
    next_obs: ArrayLike,
    discount: float = DISCOUNT,
    n_steps: int = N_STEP,
    num_samples: int = IQN_TRAIN_SAMPLES,
    huber_k: float = IQN_HUBER_LOSS_K,
    temperature: float = MUNCHAUSEN_TEMPERATURE,
    munchausen_scaling_term: float = MUNCHAUSEN_SCALING_TERM,
    munchausen_clipping_val: float = MUNCHAUSEN_CLIPPING_VAL,
    soft_sampling: bool = ACT_USE_SOFT_POLICY,
    act_temperature: float = ACT_TEMPERATURE,
    epsilon: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """
    One gradient step, returning the batch's new priorities and its diagnostic scalars, both left
    on the device. Only the scalars that need the batch, its gradient or the online pass's
    activations are here; the per-parameter scales are ``_scale_stats``.
    """
    obs = norm_obs(obs)
    actions = jnp.expand_dims(actions, -1)
    rewards = jnp.expand_dims(rewards, -1)
    dones = jnp.expand_dims(dones, -1)
    next_obs = norm_obs(next_obs)

    # Raw stored priorities: normalising by the batch maximum cancels the missing sum_prios.
    # Keep this [batch], not [batch, 1], or the product with transition_losses silently becomes
    # mean(w) * mean(loss).
    importance_sampling_weights = jnp.asarray(sample_prios) ** -PER_BETA
    importance_sampling_weights /= jnp.max(importance_sampling_weights)

    class RawStats(NamedTuple):
        td_error: jax.Array
        q_quants: jax.Array
        policy_entropy: jax.Array
        action_gap: jax.Array
        regret: jax.Array
        gap_bar: jax.Array
        sigma_rank: jax.Array
        features: jax.Array

    def loss_fn(
        qnet: QNet, target_qnet: QNet, rngs: nnx.Rngs
    ) -> tuple[jax.Array, RawStats]:
        samples = rngs.samples.uniform((*obs.shape[:-3], num_samples))
        q_quants, outputs = nnx.capture(
            qnet, nnx.Intermediate, method_outputs=nnx.Intermediate
        )(obs, samples=samples, rngs=rngs)
        features = jnp.asarray(outputs["decoder"]["__call__"][0], jnp.float32)
        action_q_quants = jnp.take_along_axis(q_quants, actions[..., None], -1)

        qs = q_quants.mean(-2)
        policy_log_probs = nnx.log_softmax(qs / temperature)
        policy_entropy = -jnp.sum(jnp.exp(policy_log_probs) * policy_log_probs, -1)

        top_two = jax.lax.top_k(qs, 2)[0]
        action_gap = top_two[..., 0] - top_two[..., 1]
        max_qs = top_two[..., 0]

        # docs/temperature.tex's two per-state scales: the mean gap, and the spread of the
        # action ordering across quantiles. Over IQN_TRAIN_SAMPLES quantiles, so their levels
        # differ from the offline probe's (docs/behaviour-policy.md).
        gap_bar = (max_qs[..., None] - qs).mean(-1)
        sigma_rank = (q_quants - q_quants.mean(-1, keepdims=True)).std(-2).mean(-1)

        # What the behaviour policy gives up per decision against the greedy action, in reward
        # units. Blind to the NoisyNet draw's own cost: a draw's argmax is this q's argmax.
        if soft_sampling:
            behaviour = nnx.softmax(qs / act_temperature)
            regret = max_qs - (behaviour * qs).sum(-1)
        else:
            regret = jnp.zeros_like(action_gap)
        regret = (1 - epsilon) * regret + epsilon * (max_qs - qs.mean(-1))

        # Off the online pass, as BTR's code does (``self.net.qvals(states)`` in Agent.py),
        # where Munchausen's Eq. 7 and BTR's Eq. E1 write the target net. It saves a trunk pass.
        action_log_probs = jax.lax.stop_gradient(
            jnp.take_along_axis(policy_log_probs, actions, -1)
        )

        # Added once and unscaled by n, with the clip inside the scaling, as Eq. E1 and BTR's
        # code both do.
        munchausen_rewards = rewards + munchausen_scaling_term * (
            temperature * action_log_probs
        ).clip(munchausen_clipping_val, 0)

        next_samples = rngs.samples.uniform((*next_obs.shape[:-3], num_samples))
        target_next_q_quants = target_qnet(next_obs, samples=next_samples, rngs=rngs)
        target_next_qs = target_next_q_quants.mean(-2, keepdims=True)

        target_next_logits = nnx.log_softmax(target_next_qs / temperature)
        target_next_probs = jnp.exp(target_next_logits)

        next_value_quants = jnp.sum(
            target_next_probs
            * (target_next_q_quants - temperature * target_next_logits),
            -1,
        )

        # The buffer redraws any rollout a time limit would cut short, so every transition
        # spans exactly n_steps or ends terminal.
        target_q_quants = (
            munchausen_rewards
            + jnp.where(dones, 0, discount**n_steps) * next_value_quants
        )

        unscaled_td_error = target_q_quants[..., None, :] - action_q_quants
        pinball_weights = jnp.abs(samples[..., None] - (unscaled_td_error < 0))
        td_error = pinball_weights * unscaled_td_error

        total_loss = pinball_weights * scaled_huber_loss(unscaled_td_error, huber_k)
        transition_losses = total_loss.mean(-1).sum(-1)
        total_loss = jnp.mean(importance_sampling_weights * transition_losses)

        return total_loss, RawStats(
            td_error,
            action_q_quants,
            policy_entropy,
            action_gap,
            regret,
            gap_bar,
            sigma_rank,
            features,
        )

    loss: jax.Array
    raw_stats: RawStats
    ((loss, raw_stats), grads) = nnx.value_and_grad(loss_fn, has_aux=True)(
        qnet, target_qnet, rngs
    )

    (
        td_error,
        q_quants,
        policy_entropy,
        action_gap,
        regret,
        gap_bar,
        sigma_rank,
        features,
    ) = raw_stats

    mean_td_error = td_error.mean((-2, -1))
    mean_abs_td_error = jnp.abs(td_error).mean((-2, -1))
    new_prios = (mean_abs_td_error + PER_EPSILON) ** PER_ALPHA

    # grad_norm is the whole gradient, as the clip reads it; grad_to_weight takes both norms
    # over the same masked parameters.
    grad_norm = optax.global_norm(grads)
    param_mask = unnormalised_param_mask(qnet)
    weight_norm = optax.global_norm(
        masked_leaves(param_mask, nnx.state(qnet, nnx.Param))
    )
    weight_grad_norm = optax.global_norm(masked_leaves(param_mask, grads))

    buffer_weights = 1 / jnp.asarray(sample_prios)
    buffer_weights /= buffer_weights.sum()
    stats = {
        "train/loss": loss,
        "train/abs_td_error": (buffer_weights * mean_abs_td_error).sum(),
        "train/td_error": (buffer_weights * mean_td_error).sum(),
        "train/q": (buffer_weights * q_quants.mean((-2, -1))).sum(),
        "train/policy_entropy": (buffer_weights * policy_entropy).sum(),
        "train/action_gap": (buffer_weights * action_gap).sum(),
        "train/regret": (buffer_weights * regret).sum(),
        "train/gap_bar": (buffer_weights * gap_bar).sum(),
        "train/sigma_rank": (buffer_weights * sigma_rank).sum(),
        "train/gap_over_sigma_rank": (buffer_weights * gap_bar / sigma_rank).sum(),
        "train/grad_norm": grad_norm,
        "train/grad_to_weight": weight_grad_norm / weight_norm,
    }
    # Over the batch as PER drew it, unweighted, where the means above are reweighted onto
    # uniform replay.
    percentiles = (5, 50, 95)
    for name, values in (
        ("policy_entropy", policy_entropy),
        ("action_gap", action_gap),
        ("regret", regret),
        ("sigma_rank", sigma_rank),
        ("gap_over_sigma_rank", gap_bar / sigma_rank),
    ):
        quantiles = jnp.percentile(values, jnp.asarray(percentiles, jnp.float32))
        for index, percentile in enumerate(percentiles):
            stats[f"train/{name}_p{percentile}"] = quantiles[index]
    stats |= feature_scales(features, num_channels=qnet.decoder.out_channels)

    opt.update(qnet, grads)

    return new_prios, stats


@nnx.jit(graph=False)
def _scale_stats(qnet: QNet, opt: nnx.Optimizer[QNet]) -> dict[str, jax.Array]:
    """
    The per-parameter scale diagnostics, left on the device. Apart from ``_train_step`` because
    they need no batch and so run on a coarser cadence. ``graph=False`` because this mutates
    nothing, and graph mode would write every parameter and moment back out on each call.
    """
    return diagnostic_scales(qnet) | adam_optim_scales(opt, qnet)


def sync_qnet(qnet: QNet, target_qnet: QNet) -> None:
    """
    Point ``target_qnet`` at the arrays ``qnet`` holds now. An alias, not a copy: JAX arrays are
    immutable, so a later ``opt.update`` cannot reach the target. Donating these buffers would
    break that.
    """
    nnx.update(target_qnet, nnx.state(qnet))


def make_inference_qnet(
    num_actions: int, obs_shape: tuple[int, int, int], rngs: nnx.Rngs
) -> QNet:
    """
    A network to load weights into and act with, never to train. ``use_running_average`` makes
    its SpectralNorm layers read the carried ``u`` without advancing it; ``eval()`` would set
    more flags than that one.
    """
    qnet = QNet(num_actions, obs_shape=obs_shape, layer_norm=LAYER_NORM, rngs=rngs)
    qnet.set_attributes(use_running_average=True, raise_if_not_found=False)
    return qnet


def eval_epsilon_at(num_frames: int) -> float:
    """
    Evaluation epsilon after ``num_frames`` ALE frames: EVAL_EPS_GREEDY, then 0 from
    EVAL_EPS_GREEDY_OFF_FRAMES, as in BTR's Table D6.
    """
    return EVAL_EPS_GREEDY if num_frames < EVAL_EPS_GREEDY_OFF_FRAMES else 0.0


def _behaviour(
    act: Callable[..., tuple[jax.Array, jax.Array]],
    obs: npt.NDArray[Any],
    rngs: nnx.Rngs,
    *,
    num_frames: int,
    evaluation: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """
    The acting half of ``BTR`` and ``QNetPolicy``: pick the epsilon for this many frames, then
    draw with ``act``, which is ``_act`` bound to the caller's net. Evaluation takes the argmax.
    The log-probability comes back as ``log_prob``, logged as ``run/action_log_prob``.
    """
    epsilon = eval_epsilon_at(num_frames) if evaluation else epsilon_at(num_frames)
    actions, log_probs = act(
        obs,
        rngs=rngs,
        soft_sampling=not evaluation and ACT_USE_SOFT_POLICY,
        epsilon=epsilon,
    )
    return actions, {"log_prob": log_probs}


class QNetPolicy(Policy):
    """
    A ``QNet`` that is only ever loaded into, for scoring and for watching. Its RNG streams are
    its own, so acting with it does not disturb a training run's draws.
    """

    def __init__(
        self, num_actions: int, obs_shape: tuple[int, int, int], *, frames_per_step: int
    ) -> None:
        self._frames_per_step = frames_per_step
        self._rngs = create_rngs(0)
        self.qnet = make_inference_qnet(num_actions, obs_shape, self._rngs)
        # Caches the graph walk, not the values: the binding shares the net's Variables, so
        # ``load`` shows through. Valid only while ``self.qnet`` is never replaced.
        self._act = nnx.cached_partial(_act, self.qnet)

    @override
    def checkpointables(self) -> dict[str, Any]:
        return {"qnet": nnx.state(self.qnet)}

    @override
    def load(self, weights: Mapping[str, Any], *, seed: int) -> None:
        # nnx.update writes variables only, so make_inference_qnet's flag survives it.
        nnx.update(self.qnet, weights["qnet"])
        self._rngs = create_rngs(seed)

    @override
    def act(
        self, obs: npt.NDArray[Any], *, num_env_steps: int, evaluation: bool = False
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        return _behaviour(
            self._act,
            obs,
            self._rngs,
            num_frames=num_env_steps * self._frames_per_step,
            evaluation=evaluation,
        )


class BTR(Agent):
    """
    Beyond the Rainbow: IQN quantile regression against a Munchausen target, NoisyNets for
    exploration, prioritised replay, a dueling head and an Impala trunk under spectral norm.

    ``frames_per_step`` is the task's frameskip, since every schedule above is quoted in ALE
    frames and the run counts env steps.
    """

    # The batches whose priorities are still on the device, oldest first: each is the indices
    # learn_step drew and the array it left in flight for them. _write_back_prios empties it.
    _prio_queue: deque[tuple[npt.NDArray[np.uint32], jax.Array]]
    # How many of them may be in flight at once, fixed in init where the environment count is.
    _prio_queue_len: int

    def __init__(
        self,
        num_actions: int,
        obs_shape: tuple[int, int, int],
        *,
        frames_per_step: int,
        seed: int,
    ) -> None:
        self._num_actions = num_actions
        self._obs_shape = obs_shape
        self._frames_per_step = frames_per_step
        self._seed = seed
        self._num_updates = 0
        self._num_frames = 0
        self._max_prio = 1.0
        self._buf: ReplayBuffer | None = None
        self._prio_queue = deque()
        self._grad_steps_owed = Fraction(0)

        self.rngs = create_rngs(seed)
        self.qnet = QNet(
            num_actions, obs_shape=obs_shape, layer_norm=LAYER_NORM, rngs=self.rngs
        )
        # Not checkpointed: sync_qnet rebuilds both from qnet. The flag stops their passes
        # advancing SpectralNorm's power iteration, and survives every sync.
        self.target_qnet = nnx.clone(self.qnet)
        self.inference_qnet = nnx.clone(self.qnet)
        for net in (self.target_qnet, self.inference_qnet):
            net.set_attributes(use_running_average=True, raise_if_not_found=False)
        # Bound as in QNetPolicy; sync_qnet and restore write through the shared Variables.
        self._act = nnx.cached_partial(_act, self.inference_qnet)

        self.opt = nnx.Optimizer(
            self.qnet,
            optax.chain(
                optax.clip_by_global_norm(GRADIENT_CLIPPING_MAX_NORM),
                optax.inject_hyperparams(optax.adamw)(
                    learning_rate=LEARNING_RATE,
                    b1=ADAM_B1,
                    b2=ADAM_B2,
                    eps=ADAM_EPS,
                    weight_decay=WEIGHT_DECAY,
                    # Gains excluded; see unnormalised_param_mask.
                    mask=unnormalised_param_mask(self.qnet),
                ),
            ),
            wrt=nnx.Param,
        )

        # Bound like _act, which holds because opt.update leaves the graph structure alone.
        # Last, since opt holds qnet.
        self._train_step = nnx.cached_partial(
            _train_step, self.qnet, self.target_qnet, self.opt, self.rngs
        )

    @property
    @override
    def num_updates(self) -> int:
        return self._num_updates

    @property
    @override
    def hyperparameters(self) -> dict[str, bool | int | float | str]:
        """Every upper-case scalar constant in this module; a Fraction as its nearest float."""
        return {
            f"btr/{name}": float(value) if isinstance(value, Fraction) else value
            for name, value in globals().items()
            if name.isupper() and isinstance(value, bool | int | float | str | Fraction)
        }

    @override
    def init(self, obs: npt.NDArray[Any]) -> None:
        num_envs = obs.shape[0]
        # An iteration's own dispatches plus the slack, so the write-back always reads a batch
        # dispatched an iteration earlier, whatever the ratio.
        self._prio_queue_len = math.ceil(REPLAY_RATIO * num_envs) + PRIO_QUEUE_SLACK

        # A draw needs obs_stack + N_STEP - 1 transitions in its environment, while
        # learn_step's gate counts the whole buffer.
        per_env_start = TRAIN_START_BUF_SIZE // num_envs
        samplable_at = self._obs_shape[0] + N_STEP - 1
        if per_env_start < samplable_at:
            raise ValueError(
                f"TRAIN_START_BUF_SIZE {TRAIN_START_BUF_SIZE} over {num_envs} environments is "
                f"{per_env_start} transitions each, short of the {samplable_at} that a "
                f"{self._obs_shape[0]}-frame stack and an {N_STEP}-step rollout need"
            )

        self._buf = ReplayBuffer(
            num_envs,
            BUFFER_SIZE // num_envs,
            obs_stack=self._obs_shape[0],
            obs_shape=obs.shape[2:],
            obs_dtype=obs.dtype,
            act_dtype=np.uint8,
            act_shape=(),
            use_prios=True,
            max_prio=self._max_prio,
            seed=self._seed,
        )
        self._buf.reset(obs)

    @override
    def act(
        self, obs: npt.NDArray[Any], *, num_env_steps: int, evaluation: bool = False
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """On inference_qnet: acting on qnet would advance its power iteration."""
        if not evaluation:
            self._num_frames = num_env_steps * self._frames_per_step
        return _behaviour(
            self._act,
            obs,
            self.rngs,
            num_frames=num_env_steps * self._frames_per_step,
            evaluation=evaluation,
        )

    @override
    def observe(self, step: EnvStep) -> None:
        """
        Write back the priorities of finished gradient steps, then store the step.

        Only batches at least an iteration old are written back, since reading the step just
        dispatched would wait on the device. That costs exactness in two ways. A slot recycled
        between a batch's draw and its write-back inherits the old priority: at most
        ``2 * PRIO_QUEUE_SLACK`` slots per environment, and never an unsamplable one, since the
        buffer rejects on slot type. And until its write-back lands, a batch's transitions are
        sampled at the priority they were drawn on.
        """
        assert self._buf is not None, "observe before init"

        self._write_back_prios()
        self._buf.save_step(
            step.actions.astype(np.uint8),
            step.rewards.astype(np.float32),
            step.terminated,
            step.truncated,
            step.next_obs,
        )
        self._grad_steps_owed += REPLAY_RATIO * self._buf.num_envs

    def _write_back_prios(self) -> None:
        """
        Write back the batches in flight longest, leaving room in the queue for one more. Called
        from ``observe``, and from ``learn_step`` between the steps of a burst.
        """
        assert self._buf is not None, "_write_back_prios before init"

        while self._prio_queue_len <= len(self._prio_queue):
            indices, prios = self._prio_queue.popleft()
            self._buf.update_prios(indices, jax.device_get(prios))

    @override
    def learn_step(self) -> tuple[int, dict[str, jax.Array]]:
        """
        Take every whole gradient step REPLAY_RATIO owes, once the buffer has filled, and leave
        them in flight. The scalars returned are those of the newest step landing on
        TRAIN_LOG_FREQ or SCALE_LOG_FREQ, and empty if none does.
        """
        assert self._buf is not None, "learn_step before init"
        if len(self._buf) < TRAIN_START_BUF_SIZE:
            # Dropped, not carried: they would come due in one burst on a single policy's data.
            self._grad_steps_owed = Fraction(0)
            return 0, {}

        num_steps, self._grad_steps_owed = divmod(self._grad_steps_owed, 1)

        stats: dict[str, jax.Array] = {}
        for _ in range(num_steps):
            self._write_back_prios()

            (
                batch_indices,
                batch_prios,
                batch_obs,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_obs,
            ) = self._buf.sample(BATCH_SIZE, n_steps=N_STEP, discount=DISCOUNT)

            new_prios, step_stats = self._train_step(
                sample_prios=batch_prios,
                obs=batch_obs,
                actions=batch_actions,
                rewards=batch_rewards,
                dones=batch_dones,
                next_obs=batch_next_obs,
                epsilon=epsilon_at(self._num_frames),
            )
            new_prios.copy_to_host_async()
            self._prio_queue.append((batch_indices, new_prios))

            self._num_updates += 1
            if self._num_updates % INFERENCE_SYNC_FREQ == 0:
                sync_qnet(self.qnet, self.inference_qnet)
            if self._num_updates % TARGET_NETWORK_UPDATE_FREQ == 0:
                sync_qnet(self.qnet, self.target_qnet)

            writes_scales = self._num_updates % SCALE_LOG_FREQ == 0
            if writes_scales:
                step_stats |= _scale_stats(self.qnet, self.opt)
            if writes_scales or self._num_updates % TRAIN_LOG_FREQ == 0:
                stats = step_stats

        return num_steps, stats

    @override
    def stats(self) -> dict[str, float]:
        """
        run/buffer_transitions includes the unsamplable slots around each write head. It is the
        one series that shows a resumed run's refill.
        """
        if self._buf is None:
            return {}
        return {"run/buffer_transitions": len(self._buf)}

    @override
    def policy_weights(self) -> dict[str, Any]:
        """
        Cloned because ``nnx.state`` aliases the live Params, which ``opt.update`` writes to. The
        clone shares the device arrays, so it costs nothing. Keyed as
        ``QNetPolicy.checkpointables``.
        """
        return {"qnet": nnx.clone(nnx.state(self.qnet))}

    @override
    def make_policy(self) -> Policy:
        return QNetPolicy(
            self._num_actions,
            self._obs_shape,
            frames_per_step=self._frames_per_step,
        )

    @override
    def checkpointables(self) -> dict[str, Any]:
        """
        ``max_prio`` is saved because the buffer is not: a resumed run refills at the priority
        scale training reached, not at 1.0.
        """
        max_prio = self._max_prio if self._buf is None else self._buf.max_prio
        # TODO: we should probably store most if not all hyper-parameters here.
        #       Maybe create a TypedDict or dataclass to hold them.
        return {
            "qnet": nnx.state(self.qnet),
            "target_qnet": nnx.state(self.target_qnet),
            "opt": nnx.state(self.opt),
            "rngs": nnx.state(self.rngs),
            "counters": {"num_updates": self._num_updates},
            "replay": {"max_prio": max_prio},
        }

    @override
    def restore(self, restored: Mapping[str, Any]) -> None:
        nnx.update(self.qnet, restored["qnet"])
        nnx.update(self.target_qnet, restored["target_qnet"])
        nnx.update(self.opt, restored["opt"])
        nnx.update(self.rngs, restored["rngs"])
        sync_qnet(self.qnet, self.inference_qnet)

        self._num_updates = restored["counters"]["num_updates"]
        # Read by init; the buffer holds it read-only.
        self._max_prio = restored["replay"]["max_prio"]
