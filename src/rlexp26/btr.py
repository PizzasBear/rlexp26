import math
from collections.abc import Collection
from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.typing import ArrayLike

from .internals import SpectralNorm

# Hyperparameters
BATCH_SIZE = 256

LEARNING_RATE = 1e-4
ADAM_EPS = 0.005 / BATCH_SIZE
DISCOUNT = 0.997
N_STEP = 3

IQN_TRAIN_SAMPLES = 8
IQN_NUM_COS = 64
IQN_HUBER_LOSS_K = 1.0
GRADIENT_CLIPPING_MAX_NORM = 10

TARGET_NETWORK_UPDATE_FREQ = 500
IMPALA_SIZE_FACTOR = 2

MUNCHAUSEN_TEMPERATURE = 0.03
MUNCHAUSEN_SCALING_TERM = 0.9
MUNCHAUSEN_CLIPPING_VAL = -1.0


# TODO: Implement the multi-step prioritised replay buffer
# TODO: Implement train step
# TODO: Check spectral normalisation for correctness (esp. in the target network).
#       Turns out NNX is again different from PyTorch in that they decided to have
#       SN mutate the weights instead of IDK... Maybe we'll need to implement it from scratch.
# TODO: Add improvements to deal with primacy bias and increase RR (e.g. layer / batch norm)
# TODO: Integrate some tracking method like TensorBoard

# +--------------+
# | AI Generated |
# +--------------+

# === Broken right now ===
# TODO: BATCH_SIZE is unused.

# === Correctness to settle ===
# TODO: decide the Munchausen n-step scale: add MUNCHAUSEN_N_STEP_SCALE, ablate
#       1.0 vs (γ^n-1)/(γ-1). Clip before scaling.
# TODO: NoisyLinear noise is shared across the whole batch. That matches the
#       original, but check whether per-sample noise helps gradient diversity.
# TODO: act() is effectively argmax already (q/τ has spread ~1e4). Switch to argmax
#       and drop rngs.actions; NoisyNets is the exploration mechanism.
# TODO: verify NoisyLinear sigma params are actually receiving gradient and their
#       magnitude decays over training (standard NoisyNets diagnostic).

# === Missing core pieces ===
# TODO: multi-step prioritised replay buffer.
#       - n-step windows must not cross episode boundaries
#       - Gymnasium >=1.0 vector envs use next-step autoreset: the step AFTER a
#         termination returns the reset obs with reward 0 and done False and
#         ignores the action. That transition must never enter the buffer.
#       - store obs as uint8, index frames rather than storing stacks (4x memory)
#       - decide: numpy sum-tree on host vs. jax-side sampling
# TODO: decide priority definition: per-transition loss vs |mean TD error|.
# TODO: real train step, decoupled from the env loop; warmup before first update.
# TODO: separate inference net for act(), synced from the training net.
# TODO: evaluation loop with unclipped rewards and no sticky-action mismatch.
# TODO: logging (TensorBoard or wandb): return, loss, Q magnitude, TD error,
#       grad norm, σ per SN layer, NoisyNet σ, dead-unit fraction, fps.
# TODO: checkpointing (orbax) — 12h runs will crash at hour 11.
# TODO: seeding: one Rngs per concern, reproducible across restarts.

# === Performance ===
# TODO: overlap learner and actor — dispatch opt_step (async) BEFORE env.step,
#       block only when the result is actually needed.
# TODO: donate_argnums on the jitted steps to avoid param copies.
# TODO: check replay sampling isn't the bottleneck; prefetch batches on a thread.
# TODO: try bf16 for the conv trunk, fp32 for the head and loss.
# TODO: measure whether act() should use fewer than K=32 samples.
# TODO: profile with jax.profiler; check whether the Impala trunk or the buffer dominates.

# === Reproduction / tuning ===
# TODO: replay ratio: with 64 envs and a 1:1 act/train schedule and batch 256,
#       each collected transition is replayed ~4x. Check that against the paper.
# TODO: target update every 500 gradient steps = 32k env steps here. Verify.
# TODO: verify against BTR's reported Breakout curve before adding anything new.

# === Experimental / longer term ===
# TODO: LayerNorm (BTR footnote) vs BatchNorm in the trunk / head. Does LN make
#       SN redundant? Try LN-only, SN-only, both.
# TODO: plasticity: periodic head resets (BBF-style), shrink-and-perturb; measure
#       dead units and effective rank before deciding it's needed.
# TODO: soft/EMA target update vs hard copy every 500 — cheaper to tune, and it
#       interacts with the SN projection differently.
# TODO: try swapping the IQN head for the leaky-clip / bounded matching-loss head
#       as a controlled ablation on the same trunk.
# TODO: try higher replay ratio + resets; see where BTR breaks on a 3080.
# TODO: exploration beyond NoisyNets (NGU / BYOL-Explore) — only after a clean
#       baseline reproduces.
# TODO: sanity-check the whole stack on LunarLander (discrete) first; Atari runs
#       are too slow a feedback loop for debugging.
# TODO: consider a tiny env + tiny net regression test that runs in seconds, so
#       late-night edits get caught by something other than the next reading.


def adaptive_max_pool(x: jax.Array, output_size: Collection[int]) -> jax.Array:
    """
    Applies adaptive max pooling to map arbitrary spatial dimensions to a fixed output_size.
    Calculates fixed stride, kernel size, and padding values that satisfy standard
    pooling output formulas: O = floor((H + pad - kernel) / stride) + 1
    """

    strides = list[int]()
    window_shape = list[int]()
    padding = list[tuple[int, int]]()

    for h, o in zip(x.shape[-len(output_size) - 1 : -1], output_size):
        if h < o:
            raise ValueError(
                "Adaptive max pool output size must not be larger than the input size"
            )

        s1, r = divmod(h, o)

        if 0 < r and o - r <= s1:
            s = k = s1 + 1
            p = o - r
        else:
            s = s1
            k = s1 + r
            p = 0

        strides.append(s)
        window_shape.append(k)
        padding.append(((p + 1) // 2, p // 2))

    return nnx.max_pool(  # type: ignore
        x,
        window_shape=tuple(window_shape),
        strides=tuple(strides),
        padding=padding,  # type: ignore
    )


class NoisyLinear(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        sigma0: float = 0.5,
        deterministic: bool = False,
    ):
        init_range = 1 / math.sqrt(in_features)

        self.kernel_mean = nnx.Param(
            rngs.params.uniform(
                (in_features, out_features), minval=-init_range, maxval=init_range
            )
        )
        self.kernel_sigma = nnx.Param(
            jnp.full((in_features, out_features), sigma0 * init_range)
        )
        self.bias_mean = nnx.Param(
            rngs.params.uniform((out_features,), minval=-init_range, maxval=init_range),
        )
        self.bias_sigma = nnx.Param(jnp.full((out_features,), sigma0 * init_range))

        self.in_features = in_features
        self.out_features = out_features

        self.deterministic = deterministic

    @staticmethod
    def _f(x: ArrayLike) -> jax.Array:
        return jnp.sign(x) * jnp.sqrt(jnp.abs(x))

    def __call__(
        self,
        inputs: ArrayLike,
        *,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        inputs = jnp.asarray(inputs)

        y = inputs @ self.kernel_mean[...] + self.bias_mean
        if deterministic if deterministic is not None else self.deterministic:
            return y

        if rngs is None:
            raise ValueError(
                "rngs argument is required unless deterministic behaviour is specified"
            )

        eps_in = self._f(rngs.noise.normal((*inputs.shape[:-1], self.in_features)))
        eps_out = self._f(rngs.noise.normal((*inputs.shape[:-1], self.out_features)))

        noise = eps_out * ((eps_in * inputs) @ self.kernel_sigma + self.bias_sigma)

        return y + noise


class ImpalaResSubBlock(nnx.Module):
    """Impala Residual Sub-Block"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.conv0 = SpectralNorm(
            nnx.Conv(in_features, out_features, kernel_size=(3, 3), rngs=rngs),
            rngs=rngs,
        )
        self.conv1 = SpectralNorm(
            nnx.Conv(out_features, out_features, kernel_size=(3, 3), rngs=rngs),
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> jax.Array:
        y: jax.Array = jnp.asarray(x)
        y = nnx.relu(y)
        y = self.conv0(y)
        y = nnx.relu(y)
        y = self.conv1(y)
        return y + x


class ImpalaBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.conv0 = nnx.Conv(in_features, out_features, kernel_size=(3, 3), rngs=rngs)
        self.res1 = ImpalaResSubBlock(out_features, out_features, rngs=rngs)
        self.res2 = ImpalaResSubBlock(out_features, out_features, rngs=rngs)

    def __call__(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        x = self.conv0(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNNLarge(nnx.Module):
    def __init__(
        self,
        in_features: int,
        *,
        rngs: nnx.Rngs,
        size_factor: int = IMPALA_SIZE_FACTOR,
    ) -> None:
        self.size_factor = size_factor

        n = self.size_factor
        self.block0 = ImpalaBlock(in_features, n * 16, rngs=rngs)
        self.block1 = ImpalaBlock(n * 16, n * 32, rngs=rngs)
        self.block2 = ImpalaBlock(n * 32, n * 32, rngs=rngs)

    @property
    def out_features(self) -> int:
        return self.size_factor * 32 * 6 * 6

    def __call__(self, x: ArrayLike) -> jax.Array:
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = nnx.relu(x)  # TODO: Should this ReLU be here?
        x = adaptive_max_pool(cast(jax.Array, x), (6, 6))
        return x.reshape(*x.shape[:-3], -1)


class IQNCosineEmbedding(nnx.Module):
    def __init__(
        self,
        out_features: int,
        *,
        num_cosines: int = IQN_NUM_COS,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_cosines = num_cosines
        self.linear = nnx.Linear(self.num_cosines, out_features, rngs=rngs)

    def __call__(self, x: ArrayLike) -> jax.Array:
        pi_factors = jnp.pi * jnp.arange(1, self.num_cosines + 1)
        x = pi_factors * jnp.expand_dims(x, -1)
        x = jnp.cos(x)
        x = self.linear(x)
        x = nnx.relu(x)
        return x


class QNet(nnx.Module):
    def __init__(self, num_actions: int, *, rngs: nnx.Rngs) -> None:
        self.num_actions = num_actions

        self.decoder = ImpalaCNNLarge(4, rngs=rngs)
        self.quantile_embedding = IQNCosineEmbedding(
            self.decoder.out_features, rngs=rngs
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


def norm_obs(obs: ArrayLike) -> jax.Array:
    return jnp.astype(obs, jnp.float32) / 255


@nnx.jit
def act(
    qnet: QNet,
    obs: ArrayLike,
    *,
    rngs: nnx.Rngs,
    num_samples: int = 32,
    temperature: float = MUNCHAUSEN_TEMPERATURE,
) -> jax.Array:
    obs = norm_obs(obs)

    # TODO: Seems like BTR isn't really stochastic here, rather it does
    #       Noisy-Nets and epsilon-greedy instead. Maybe I should switch
    #       this to argmax instead.
    q = qnet.random_n_samples_mean(obs, num_samples, rngs=rngs)
    logits = nnx.log_softmax(q / temperature)
    return rngs.actions.categorical(logits)


@nnx.jit
def opt_step(
    *,
    qnet: QNet,
    target_qnet: QNet,
    opt: nnx.Optimizer,
    importance_sampling_weights: ArrayLike,
    obs: ArrayLike,
    actions: ArrayLike,
    rewards: ArrayLike,
    dones: ArrayLike,
    next_obs: ArrayLike,
    rngs: nnx.Rngs,
    discount: float = DISCOUNT,
    n_steps: int = N_STEP,
    num_samples: int = IQN_TRAIN_SAMPLES,
    huber_k=IQN_HUBER_LOSS_K,
    temperature: float = MUNCHAUSEN_TEMPERATURE,
    munchausen_scaling_term: float = MUNCHAUSEN_SCALING_TERM,
    munchausen_clipping_val: float = MUNCHAUSEN_CLIPPING_VAL,
) -> jax.Array:
    obs = norm_obs(obs)
    actions = jnp.expand_dims(actions, -1)
    rewards = jnp.expand_dims(rewards, -1)
    dones = jnp.expand_dims(dones, -1)
    next_obs = norm_obs(next_obs)

    # TODO: check this correct
    def loss_fn(qnet: QNet, target_qnet: QNet, rngs: nnx.Rngs):
        target_qnet_qs = target_qnet.random_n_samples_mean(obs, num_samples, rngs=rngs)
        target_logits = nnx.log_softmax(target_qnet_qs / temperature)
        target_action_log_probs = jnp.take_along_axis(target_logits, actions, -1)

        # TODO: munchausen rewards and multi-step work together? how should we handle them?

        munchausen_rewards = rewards + munchausen_scaling_term * (
            temperature * target_action_log_probs
        ).clip(munchausen_clipping_val, 0)

        next_samples = rngs.samples.uniform((*next_obs.shape[:-3], num_samples))
        target_next_q_quants = target_qnet(next_obs, samples=next_samples, rngs=rngs)
        target_next_qs = target_next_q_quants.mean(-2, keepdims=True)

        # TODO: Should I compute softmax twice instead?
        target_next_logits = nnx.log_softmax(target_next_qs / temperature)
        target_next_probs = jnp.exp(target_next_logits)

        next_value_quants = jnp.sum(
            target_next_probs
            * (target_next_q_quants - temperature * target_next_logits),
            -1,
        )

        # TODO: discount to the power of multi-step n?
        target_q_quants = (
            munchausen_rewards
            + jnp.where(dones, 0, discount**n_steps) * next_value_quants
        )

        samples = rngs.samples.uniform((*obs.shape[:-3], num_samples))
        q_quants = jnp.take_along_axis(
            qnet(obs, samples=samples, rngs=rngs), actions[..., None], -1
        )

        td_error = target_q_quants[..., None, :] - q_quants
        loss = (
            jnp.abs(samples[..., None] - (td_error < 0))
            * optax.huber_loss(td_error, delta=huber_k)
            / huber_k
        )
        transition_losses = loss.mean(-1).sum(-1)

        return jnp.mean(
            importance_sampling_weights * transition_losses
        ), transition_losses

    (_loss, transition_losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        qnet, target_qnet, rngs
    )
    opt.update(qnet, grads)

    return transition_losses


# Once every TARGET_NETWORK_UPDATE_FREQ steps
@nnx.jit
def update_target(qnet: QNet, target_qnet: QNet):
    nnx.update(target_qnet, nnx.state(qnet))
