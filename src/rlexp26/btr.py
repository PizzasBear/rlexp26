from collections.abc import Mapping
from typing import Any, override

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
from .utils import adam_optim_scales, diagnostic_scales, feature_scales

# Hyperparameters
BATCH_SIZE = 256
BUFFER_SIZE = 1 << 20  # 2 MebiTrans
TRAIN_START_BUF_SIZE = 200_000

PER_ALPHA = 0.2
PER_BETA = 1.0  # BTR: 0.2
PER_EPSILON = 1e-6

LEARNING_RATE = 1e-4
ADAM_EPS = 0.005 / BATCH_SIZE
ADAM_B1 = 0.9
ADAM_B2 = 0.999
DISCOUNT = 0.997
N_STEP = 3

IQN_TRAIN_SAMPLES = 8
IQN_ACT_SAMPLES = 32  # BTR: 8
IQN_NUM_COS = 64
IQN_HUBER_LOSS_K = 1.0
GRADIENT_CLIPPING_MAX_NORM = 10

INFERENCE_SYNC_FREQ = 3
TARGET_NETWORK_UPDATE_FREQ = 500
IMPALA_SIZE_FACTOR = 2

MUNCHAUSEN_TEMPERATURE = 0.03
MUNCHAUSEN_SCALING_TERM = 0.9
MUNCHAUSEN_CLIPPING_VAL = -1.0

# Acting draws from the soft policy softmax(Q / tau) that Munchausen's target already assumes,
# rather than BTR's argmax, so the behaviour policy is the one the loss is written around.
# ACT_TEMPERATURE is an alias, not a coincidence: decouple it only to deliberately sharpen or
# flatten exploration relative to the learning target.
ACT_USE_SOFT_POLICY = True  # BTR: False (argmax + epsilon)
ACT_TEMPERATURE = MUNCHAUSEN_TEMPERATURE

# Off in favour of the softmax categorical sampling
EPS_GREEDY_START = 0.0  # BTR: 1.0
EPS_GREEDY_END = 0.0  # BTR: 0.01
EPS_GREEDY_DECAY_FRAMES = 8000_000
EPS_GREEDY_OFF_FRAMES = 100_000_000
EVAL_EPS_GREEDY = 0.0  # BTR: 0.01
EVAL_EPS_GREEDY_OFF_FRAMES = 125_000_000

# Every constant above, and the architecture below, checked against BTR (arXiv:2411.03820,
# Table D6 and Appendices E/H) and its reference implementation (github.com/VIPTankz/BTR)
# on 2026-09-12. All of it matches bar what is explicitly marked otherwise. Table D6 gives PER
# an alpha and nothing else, so PER_BETA and PER_EPSILON come from PER.py in the reference
# implementation: its eps is 1e-6, and its beta is the accident the backlog describes.

# +--------------+
# | AI Generated |
# +--------------+

# === Correctness to settle ===
# TODO: the Munchausen log-policy term is read off target_qnet, which is what the Munchausen
#       paper's Eq. 7 and BTR's Eq. E1 both write. BTR's *code* reads it off the online net
#       (`q_k_target = self.net.qvals(states)` in Agent.py), and that is the version their
#       published curve came from. Paper-vs-code, not ambiguity; the two differ by up to
#       TARGET_NETWORK_UPDATE_FREQ steps of staleness. Decide which to follow.
# TODO: PER_BETA is pinned at 1.0, which cancels PER_ALPHA exactly -- sampling frequency times
#       IS weight goes as p ** (PER_ALPHA * (1 - PER_BETA)) -- so prioritisation currently
#       affects only which transitions arrive together, not their expected contribution. BTR's
#       effective beta is 0.2 (its per_beta is vestigial: no live code reads it, inserts reset
#       it, and its anneal is switched off), leaving an effective exponent of 0.16. Reproducing
#       that is PER_BETA = 0.2. Whatever PER is worth in BTR's ablation, this run is not
#       collecting it.
# TODO: mean |sigma| per NoisyLinear *rose* over the live run, where the standard NoisyNets
#       diagnostic expects it to fall as the policy sharpens. Sigma is receiving gradient, then;
#       the question is whether it is signal. diagnostic_scales now logs the three series that
#       separate the cases -- see its docstring -- and the reading is: signed mean flat while
#       |sigma| climbs, with neg_frac leaving 0, is an Adam random walk on a near-zero-SNR
#       gradient, not exploration being learned. This run is more exposed to that than BTR is,
#       since NoisyLinear redraws per __call__ rather than once per gradient step, so the three
#       forward passes in a train_step disagree about eps. If it is drift, the fix is on the
#       optimiser side (weight decay on sigma, or BTR's draw-once), not the schedule.
#       A one-off grad probe outside train_step trips Flax's trace-level guard on the shared
#       Rngs counter, so this stays a matter of reading the curves.

# === Missing core pieces ===
# Logging and checkpointing are both done: train_step's scalars, the run rates, and the
# background evaluation's unclipped score all reach TensorBoard, and --resume picks up the
# newest step-numbered checkpoint with qnet, target_qnet, the optimizer, rngs, the counters and
# the buffer's max_prio. Dormancy, dead units and effective rank are in utils.feature_scales,
# fed by the trunk activations QNet.__call__ sows and train_step captures.
# TODO: decide whether resuming onto an empty replay buffer is acceptable. The buffer is the
#       one thing a checkpoint does not carry (tens of gigabytes), so the first
#       TRAIN_START_BUF_SIZE transitions after a resume are collected with no learning at all,
#       and the buffer then holds nothing but data from a policy that is already trained.

# === Performance ===
# Measured 2026-09-06, 3080 + 24-thread host, 64 envs, batch 256, 1:1 act/train, steady state:
# ~910 env steps/s (~3.6k ALE frames/s) => ~15h for 50M env steps / 200M frames.
# GPU 80% util at 324W/370W and 84C; host CPU ~1.4 cores of 24. The GPU is the wall, and it is
# already at its power/thermal limit, so wins have to come from doing less work on it.
# Since that measurement act() stopped evaluating the value stream, so the numbers above are
# stale on the act side and want re-taking. It still draws IQN_ACT_SAMPLES quantiles, which is
# 32 rather than BTR's 8.
# TODO: bf16 for the conv trunk, fp32 for the head and loss. The trunk is ~440 MFLOP/sample
#       and runs three times per train step (online obs, target obs, target next_obs) plus
#       once per act, so it is ~6x the head; ~8.5 TFLOP/s achieved against 29.8 fp32 / 59.5
#       TF32 peak. Check which precision XLA is actually picking for the convs first.
# TODO: donate_argnums on the jitted steps to avoid param copies.
# TODO: overlap the host side further: sample batches on a background thread and queue them so
#       the transfer runs alongside env.step, and consider moving buf.save_step off the main
#       thread too. Measure first -- the host is at ~1.4 of 24 cores, so this only pays if the
#       main thread is actually blocking the device. A full actor/learner split, which would
#       make train_step independent of collection, is the larger version of the same idea.
# TODO: profile with jax.profiler; check whether the Impala trunk or the buffer dominates.

# === Reproduction / tuning ===
# TODO: verify against BTR's reported Breakout curve before adding anything new.

# === Experimental / longer term ===
# TODO: LayerNorm, per BTR's Appendix H, which says where to put it and reports a clear gain.
#       Highest-value experiment on this list. Does LN make SN redundant? Try LN-only, SN-only,
#       both.
# TODO: plasticity. The recipe is settled (Nikishin et al., D'Oro et al., BBF); whether this run
#       suffers the pathology is not. The measurements are now in place -- plasticity/* for
#       dormancy, death and effective rank, adam_step_rel/* for whether the weights can still
#       move, weight_rms/* and weights/global_norm for the inflation that has nothing bounding
#       it -- so this is a matter of reading a full run before choosing an intervention. For
#       scale: a 50M-step run is ~780k updates, which is ~19 resets at BBF's 40k cadence.
# TODO: weight decay, as the cheap half of the above. BTR's AdamW null result was at 1e-4, so it
#       does not cover BBF's 0.1, whose reported gains *grow* with replay ratio.
# TODO: try higher replay ratio + resets; see where BTR breaks on a 3080.
# TODO: exploration beyond NoisyNets -- only after a clean baseline reproduces, and neither
#       candidate is a patch on this loop: NGU needs an R2D2 backbone and BYOL-Explore an RNN
#       world model. Only NGU's episodic bonus is extractable on its own.
# TODO: sanity-check the whole stack on LunarLander (discrete) first; Atari runs
#       are too slow a feedback loop for debugging.
# TODO: consider a tiny env + tiny net regression test that runs in seconds, so
#       late-night edits get caught by something other than the next reading.


def create_rngs(seed: int) -> nnx.Rngs:
    """
    One RNG stream per concern: ``nnx.Rngs(seed)`` would put them all on one counter, where
    drawing one more quantile per step shifts the NoisyNet noise for the rest of the run. No
    ``default``, so an unnamed stream raises instead of quietly aliasing the others.
    """
    return nnx.Rngs(
        params=seed, noise=seed + 1000, samples=seed + 2000, explore=seed + 3000
    )


class QNet(nnx.Module):
    def __init__(
        self,
        num_actions: int,
        *,
        obs_stack: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_actions = num_actions

        self.decoder = ImpalaCNNLarge(
            obs_stack, size_factor=IMPALA_SIZE_FACTOR, rngs=rngs
        )
        self.quantile_embedding = IQNCosineEmbedding(
            self.decoder.out_features, num_cosines=IQN_NUM_COS, rngs=rngs
        )
        self.value_linear0 = NoisyLinear(self.decoder.out_features, 512, rngs=rngs)
        self.value_linear1 = NoisyLinear(512, 1, rngs=rngs)

        self.advantage_linear0 = NoisyLinear(self.decoder.out_features, 512, rngs=rngs)
        self.advantage_linear1 = NoisyLinear(512, num_actions, rngs=rngs)

    def _value(self, x: ArrayLike, *, rngs: nnx.Rngs) -> jax.Array:
        x = self.value_linear0(x, rngs=rngs)
        x = nnx.relu(x)
        x = self.value_linear1(x, rngs=rngs)
        return x

    def _advantage(self, x: ArrayLike, *, rngs: nnx.Rngs) -> jax.Array:
        x = self.advantage_linear0(x, rngs=rngs)
        x = nnx.relu(x)
        x = self.advantage_linear1(x, rngs=rngs)
        return x

    def __call__(
        self,
        obs: ArrayLike,
        *,
        samples: ArrayLike,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        x = self.decoder(obs)
        x = x[..., None, :] * self.quantile_embedding(samples)

        a: jax.Array = self._advantage(x, rngs=rngs)
        v: jax.Array = self._value(x, rngs=rngs)
        return v + a - a.mean(-1, keepdims=True)

    def random_n_samples_mean(
        self,
        obs: ArrayLike,
        num_samples: int,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        obs = jnp.asarray(obs)
        samples = rngs.samples.uniform((*obs.shape[:-3], num_samples))
        return self(obs, samples=samples, rngs=rngs).mean(-2)

    def random_n_samples_advantage_mean(
        self,
        obs: ArrayLike,
        num_samples: int,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        obs = jnp.asarray(obs)
        samples = rngs.samples.uniform((*obs.shape[:-3], num_samples))
        a = self.decoder(obs)
        a = a[..., None, :] * self.quantile_embedding(samples)
        a = self._advantage(a, rngs=rngs)
        return (a - a.mean(-1, keepdims=True)).mean(-2)


def norm_obs(obs: ArrayLike) -> jax.Array:
    obs = jnp.moveaxis(obs, -3, -1)
    return jnp.astype(obs, jnp.float32) / 255


def epsilon_at(num_frames: int) -> float:
    """
    Training epsilon after ``num_frames`` ALE frames. Linear rather than exponential, which is
    what BTR's "start / decay / end" triple means.
    """
    if num_frames >= EPS_GREEDY_OFF_FRAMES:
        return 0.0
    decayed = min(num_frames / EPS_GREEDY_DECAY_FRAMES, 1.0)
    return EPS_GREEDY_START + decayed * (EPS_GREEDY_END - EPS_GREEDY_START)


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
    One action per environment -- drawn from ``softmax(Q / temperature)`` under
    ACT_USE_SOFT_POLICY, the argmax without it -- with NoisyNets and an optional epsilon-greedy
    override alongside, and that action's log-probability under the behaviour policy the
    override defines.

    That log-probability is conditional on this call's NoisyNet draw, so it is not the marginal
    behaviour policy's: read it as ``E_xi[log pi_xi(a | s)]`` with ``a ~ pi_xi``, a conditional
    entropy that understates how stochastic acting really is. Same caveat on train_step's
    ``policy_entropy``. It bounds the collapse argument below rather than measuring it.

    Soft sampling is self-correcting where epsilon-greedy is merely immune: a frozen state pays 0
    for every action forever, so Q(s, .) goes flat and the softmax goes uniform exactly where the
    policy is stuck. It also anneals itself, a fixed tau meeting an action gap that grows with Q
    -- 1.5 tau at init, a genuinely stochastic draw, against all but an argmax at the Q of 12.73
    the greedy run reached. The epsilon path is kept, per-env and independent of the weights, as
    the one escape a confidently wrong Q cannot fool; turn it back on if a freeze recurs.
    """
    obs = norm_obs(obs)

    # It's okay to use advantages here because softmax is shift invariant.
    advantages = qnet.random_n_samples_advantage_mean(obs, num_samples, rngs=rngs)
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


@nnx.jit(static_argnames=("num_samples",))
def _train_step(
    *,
    qnet: QNet,
    target_qnet: QNet,
    opt: nnx.Optimizer[QNet],
    sample_prios: ArrayLike,
    obs: ArrayLike,
    actions: ArrayLike,
    rewards: ArrayLike,
    dones: ArrayLike,
    next_obs: ArrayLike,
    rngs: nnx.Rngs,
    discount: float = DISCOUNT,
    n_steps: int = N_STEP,
    num_samples: int = IQN_TRAIN_SAMPLES,
    huber_k: float = IQN_HUBER_LOSS_K,
    temperature: float = MUNCHAUSEN_TEMPERATURE,
    munchausen_scaling_term: float = MUNCHAUSEN_SCALING_TERM,
    munchausen_clipping_val: float = MUNCHAUSEN_CLIPPING_VAL,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """
    One gradient step, returning the batch's new priorities and a dict of diagnostic scalars.
    Both stay on the device for the caller to drain an iteration later; see ``__init__.main``.
    """
    obs = norm_obs(obs)
    actions = jnp.expand_dims(actions, -1)
    rewards = jnp.expand_dims(rewards, -1)
    dones = jnp.expand_dims(dones, -1)
    next_obs = norm_obs(next_obs)

    # The buffer hands back raw stored priorities, not p / sum_prios. That is fine here:
    # normalising by the batch maximum cancels any constant factor, so neither sum_prios
    # nor the buffer's length is needed. Keep this [batch], not [batch, 1] -- broadcasting
    # it against transition_losses would silently reduce to mean(w) * mean(loss).
    importance_sampling_weights = jnp.asarray(sample_prios) ** -PER_BETA
    importance_sampling_weights /= jnp.max(importance_sampling_weights)

    def loss_fn(
        qnet: QNet, target_qnet: QNet, rngs: nnx.Rngs
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        target_qnet_qs = target_qnet.random_n_samples_mean(obs, num_samples, rngs=rngs)
        target_logits = nnx.log_softmax(target_qnet_qs / temperature)
        target_action_log_probs = jnp.take_along_axis(target_logits, actions, -1)

        # Added once, unscaled by n, with the clip inside the scaling -- Eq. E1 and BTR's code
        # agree, so the term stays a one-step correction bolted onto an n-step return rather
        # than being scaled by (gamma^n - 1)/(gamma - 1).
        munchausen_rewards = rewards + munchausen_scaling_term * (
            temperature * target_action_log_probs
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

        # One discount ** n_steps for the whole batch is right: the buffer accumulates the
        # n-step return itself and redraws any rollout a time limit would have cut short,
        # so every drawn transition spans exactly n_steps or ends terminal.
        target_q_quants = (
            munchausen_rewards
            + jnp.where(dones, 0, discount**n_steps) * next_value_quants
        )

        samples = rngs.samples.uniform((*obs.shape[:-3], num_samples))
        # The only pass over the online net on states it is training on, so the only one whose
        # trunk features are the right thing to measure plasticity on.
        # method_outputs records what each method returned, so the trunk features come out of
        # the pass the loss already needs. An uncaptured call records nothing, which is what
        # leaves the target net -- whose state is checkpointed -- untouched.
        q_quants, outputs = nnx.capture(
            qnet, nnx.Intermediate, method_outputs=nnx.Intermediate
        )(obs, samples=samples, rngs=rngs)
        features = jnp.asarray(outputs["decoder"]["__call__"][0])
        action_q_quants = jnp.take_along_axis(q_quants, actions[..., None], -1)

        policy_log_probs = nnx.log_softmax(q_quants.mean(-2) / temperature)
        policy_entropy = -jnp.sum(jnp.exp(policy_log_probs) * policy_log_probs, -1)

        unscaled_td_error = target_q_quants[..., None, :] - action_q_quants
        pinball_weights = jnp.abs(samples[..., None] - (unscaled_td_error < 0))
        td_error = pinball_weights * unscaled_td_error

        total_loss = pinball_weights * (
            optax.huber_loss(unscaled_td_error, delta=huber_k) / huber_k
        )
        transition_losses = total_loss.mean(-1).sum(-1)
        total_loss = jnp.mean(importance_sampling_weights * transition_losses)

        return total_loss, (td_error, action_q_quants, policy_entropy, features)

    loss: jax.Array
    td_error: jax.Array
    q_quants: jax.Array
    policy_entropy: jax.Array
    features: jax.Array
    (loss, (td_error, q_quants, policy_entropy, features)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True
    )(qnet, target_qnet, rngs)

    mean_td_error = td_error.mean((-2, -1))
    mean_abs_td_error = jnp.abs(td_error).mean((-2, -1))
    new_prios = (mean_abs_td_error + PER_EPSILON) ** PER_ALPHA

    scales = diagnostic_scales(qnet)
    scales |= adam_optim_scales(opt, qnet)
    scales |= feature_scales(features, num_channels=qnet.decoder.out_channels)
    grad_norm = optax.global_norm(grads)

    buffer_weights = 1 / jnp.asarray(sample_prios)
    buffer_weights /= buffer_weights.sum()
    stats = {
        "train/loss": loss,
        "train/abs_td_error": (buffer_weights * mean_abs_td_error).sum(),
        "train/td_error": (buffer_weights * mean_td_error).sum(),
        "train/q": (buffer_weights * q_quants.mean((-2, -1))).sum(),
        "train/policy_entropy": (buffer_weights * policy_entropy).sum(),
        "train/grad_norm": grad_norm,
        "train/grad_to_weight": grad_norm / scales["weights/global_norm"],
        **scales,
    }

    opt.update(qnet, grads)

    return new_prios, stats


# Once every TARGET_NETWORK_UPDATE_FREQ steps
@nnx.jit
def sync_qnet(qnet: QNet, target_qnet: QNet) -> None:
    # set_attributes below flips a static attribute, so this compiles once for the
    # use_running_average=False graphdef (the first call after nnx.clone) and once for the
    # True one, then hits cache. Verified; not a per-call retrace.

    # Copy parameters
    nnx.update(target_qnet, nnx.state(qnet))

    # Disable update_stats on SpectralNorm layers
    target_qnet.set_attributes(use_running_average=True, raise_if_not_found=False)


def make_inference_qnet(num_actions: int, obs_stack: int, rngs: nnx.Rngs) -> QNet:
    """
    A network built to act with and never to train, for the paths that load weights into one
    rather than cloning a live net through ``sync_qnet``.

    ``use_running_average`` is the flag sync sets, and only that flag: ``eval()`` sets more than
    this one and would change what ``act`` does. Its SpectralNorm layers then read the carried
    ``u`` without advancing it.
    """
    qnet = QNet(num_actions, obs_stack=obs_stack, rngs=rngs)
    qnet.set_attributes(use_running_average=True, raise_if_not_found=False)
    return qnet


def eval_epsilon_at(num_frames: int) -> float:
    """
    Evaluation epsilon after ``num_frames`` ALE frames.

    BTR scores with a little noise until 125M frames -- 25M past the point training stops
    exploring -- and only reports the fully greedy policy after that (Table D6). Kept whole, and
    reading 0 throughout, while the epsilon constants above are zeroed.
    """
    return EVAL_EPS_GREEDY if num_frames < EVAL_EPS_GREEDY_OFF_FRAMES else 0.0


def _behaviour(
    qnet: QNet,
    obs: npt.NDArray[Any],
    rngs: nnx.Rngs,
    *,
    num_frames: int,
    evaluation: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """
    The acting half of both ``BTR`` and ``QNetPolicy``: pick the epsilon this many frames in,
    then draw.

    The log-probability handed back, which the run logs as ``run/action_log_prob``, is the one
    of the action actually taken, so it is the behaviour policy's own surprise rather than a
    property of the learned q. It is conditional
    on this call's NoisyNet draw, so it reads more deterministic than the marginal policy is;
    see ``_act``. A ceiling on the soft policy's collapse toward argmax, not a measurement of it.
    """
    epsilon = eval_epsilon_at(num_frames) if evaluation else epsilon_at(num_frames)
    actions, log_probs = _act(qnet, obs, rngs=rngs, epsilon=epsilon)
    return actions, {"log_prob": log_probs}


class QNetPolicy(Policy):
    """
    A ``QNet`` that is only ever loaded into, for scoring and for watching.

    Its RNG streams are its own, so acting with it does not disturb the draws a training run is
    making on another thread.
    """

    def __init__(
        self, num_actions: int, obs_stack: int, *, frames_per_step: int
    ) -> None:
        self._frames_per_step = frames_per_step
        self._rngs = create_rngs(0)
        self.qnet = make_inference_qnet(num_actions, obs_stack, self._rngs)

    @override
    def checkpointables(self) -> dict[str, Any]:
        return {"qnet": nnx.state(self.qnet)}

    @override
    def load(self, weights: Mapping[str, Any], *, seed: int) -> None:
        # nnx.update writes variables and nothing else, so the flags make_inference_qnet set
        # survive it. The streams are replaced rather than carried on, since they are passed to
        # every call as arguments and nothing holds a reference to the old ones.
        nnx.update(self.qnet, weights["qnet"])
        self._rngs = create_rngs(seed)

    @override
    def act(
        self, obs: npt.NDArray[Any], *, num_env_steps: int, evaluation: bool = False
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        return _behaviour(
            self.qnet,
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

    # The priorities of a batch still on the device, which nnx would otherwise refuse on a
    # static attribute of a pytree class.
    _pending_prios: tuple[npt.NDArray[np.uint32], jax.Array] | None = None

    def __init__(
        self, num_actions: int, obs_stack: int, *, frames_per_step: int, seed: int
    ) -> None:
        self._num_actions = num_actions
        self._obs_stack = obs_stack
        self._frames_per_step = frames_per_step
        self._seed = seed
        self._num_updates = 0
        self._max_prio = 1.0
        self._buf: ReplayBuffer | None = None
        self._pending_prios = None

        self.rngs = create_rngs(seed)
        self.qnet = QNet(num_actions, obs_stack=obs_stack, rngs=self.rngs)
        self.target_qnet = nnx.clone(self.qnet)
        # Neither clone is checkpointed, since sync_qnet rebuilds both from qnet. The syncs are
        # not redundant: they are what set use_running_average on the SpectralNorm layers, and
        # acting on a net without it would advance the training net's power iteration outside
        # any gradient step.
        self.inference_qnet = nnx.clone(self.qnet)
        sync_qnet(self.qnet, self.target_qnet)
        sync_qnet(self.qnet, self.inference_qnet)

        self.opt = nnx.Optimizer(
            self.qnet,
            optax.chain(
                optax.clip_by_global_norm(GRADIENT_CLIPPING_MAX_NORM),
                # inject_hyperparams keeps the settings in the optimizer state, where
                # utils.adam_optim_scales reads them rather than being handed them again.
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=LEARNING_RATE,
                    b1=ADAM_B1,
                    b2=ADAM_B2,
                    eps=ADAM_EPS,
                ),
            ),
            wrt=nnx.Param,
        )

    @property
    @override
    def num_updates(self) -> int:
        return self._num_updates

    @property
    @override
    def hyperparameters(self) -> dict[str, bool | int | float | str]:
        """
        Every module-level constant here is a hyperparameter by construction, so they are swept
        up wholesale and stay correct as that block grows.
        """
        return {
            f"btr/{name}": value
            for name, value in globals().items()
            if name.isupper() and isinstance(value, bool | int | float | str)
        }

    @override
    def init(self, obs: npt.NDArray[Any]) -> None:
        num_envs = obs.shape[0]
        # An environment can only be drawn from once it holds enough slots for a draw's frame
        # stack to read backwards over and its rollout to read forwards over, which is
        # obs_stack + N_STEP - 1 transitions. learn_step's gate counts the whole buffer, so a
        # TRAIN_START_BUF_SIZE spread thinly enough over environments would open it on a buffer
        # no draw can serve; refused here, where both numbers are known, rather than surfacing
        # as a ValueError out of the first sample halfway into a run.
        per_env_start = TRAIN_START_BUF_SIZE // num_envs
        samplable_at = self._obs_stack + N_STEP - 1
        if per_env_start < samplable_at:
            raise ValueError(
                f"TRAIN_START_BUF_SIZE {TRAIN_START_BUF_SIZE} over {num_envs} environments is "
                f"{per_env_start} transitions each, short of the {samplable_at} that a "
                f"{self._obs_stack}-frame stack and an {N_STEP}-step rollout need"
            )

        self._buf = ReplayBuffer(
            num_envs,
            BUFFER_SIZE // num_envs,
            obs_stack=self._obs_stack,
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
        return _behaviour(
            self.inference_qnet,
            obs,
            self.rngs,
            num_frames=num_env_steps * self._frames_per_step,
            evaluation=evaluation,
        )

    @override
    def observe(self, step: EnvStep) -> None:
        """
        Settle up for the last gradient step, then store.

        That order is the whole reason the priority write is deferred rather than forced inside
        ``learn_step``: a batch is drawn at the end of one iteration and its priorities written
        at the start of the next, with no save_step in between, so no slot the batch names can
        have been recycled underneath it. Forcing the transfer here rather than at dispatch is
        also what lets ``learn_step`` overlap the environment -- the device has had the whole of
        ``env.step``.
        """
        assert self._buf is not None, "observe before init"

        # TODO: Maybe put a small queue of `EnvStep`s here and make pending prios a queue too?
        if self._pending_prios is not None:
            indices, prios = self._pending_prios
            self._buf.update_prios(indices, jax.device_get(prios))
            self._pending_prios = None

        self._buf.save_step(
            step.actions.astype(np.uint8),
            step.rewards.astype(np.float32),
            step.terminated,
            step.truncated,
            step.next_obs,
        )

    @override
    def learn_step(self) -> tuple[int, dict[str, jax.Array]]:
        """One gradient step once the buffer has filled, dispatched and left in flight."""
        assert self._buf is not None, "learn_step before init"
        if len(self._buf) < TRAIN_START_BUF_SIZE:
            return 0, {}

        (
            batch_indices,
            batch_prios,
            batch_obs,
            batch_actions,
            batch_rewards,
            batch_dones,
            batch_next_obs,
        ) = self._buf.sample(BATCH_SIZE, n_steps=N_STEP, discount=DISCOUNT)

        new_prios, stats = _train_step(
            qnet=self.qnet,
            target_qnet=self.target_qnet,
            opt=self.opt,
            sample_prios=batch_prios,
            obs=batch_obs,
            actions=batch_actions,
            rewards=batch_rewards,
            dones=batch_dones,
            next_obs=batch_next_obs,
            rngs=self.rngs,
        )
        new_prios.copy_to_host_async()

        if self._pending_prios is not None:
            raise RuntimeError(
                "two learn_steps with no observe between them: the batch already in flight\n"
                "would never be written back, leaving transitions the last gradient step\n"
                "already fitted at the priority they were drawn on"
            )
        self._pending_prios = (batch_indices, new_prios)

        self._num_updates += 1
        if self._num_updates % INFERENCE_SYNC_FREQ == 0:
            sync_qnet(self.qnet, self.inference_qnet)
        if self._num_updates % TARGET_NETWORK_UPDATE_FREQ == 0:
            sync_qnet(self.qnet, self.target_qnet)

        return 1, stats

    @override
    def stats(self) -> dict[str, float]:
        """
        run/buffer_transitions counts the unsamplable window around each write head too, so it
        reaches BUFFER_SIZE slightly before every slot is a drawable transition.

        A fresh run holds min(num_env_steps, BUFFER_SIZE), which is the step count every other
        curve is already drawn against. A resumed one refills from empty while that count
        carries on, so this is the only series that shows the refill -- which is the open
        question in the backlog above.
        """
        if self._buf is None:
            return {}
        return {"run/buffer_transitions": len(self._buf)}

    @override
    def policy_weights(self) -> dict[str, Any]:
        """
        The clone is what pins a snapshot to these weights: ``nnx.state`` aliases the live
        ``Param`` objects, so without it a reader on another thread would see whatever
        ``opt.update`` had written by the time it got there. It rebuilds the pytree around the
        same device arrays, so it costs nothing and does not wait on a step still computing them.

        Keyed as ``QNetPolicy.checkpointables``, which is the same key this agent checkpoints
        its acting weights under.
        """
        return {"qnet": nnx.clone(nnx.state(self.qnet))}

    @override
    def make_policy(self) -> Policy:
        return QNetPolicy(
            self._num_actions,
            self._obs_stack,
            frames_per_step=self._frames_per_step,
        )

    @override
    def checkpointables(self) -> dict[str, Any]:
        """
        ``max_prio`` rides along because the buffer itself does not: a resumed run would
        otherwise refill an empty buffer at a fresh one's 1.0 rather than the scale training had
        reached.
        """
        max_prio = self._max_prio if self._buf is None else self._buf.max_prio
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
        # Read at init, since the buffer holds it read-only once it exists.
        self._max_prio = restored["replay"]["max_prio"]
