import math
from collections import deque
from collections.abc import Callable, Mapping
from fractions import Fraction
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
from .utils import (
    adam_optim_scales,
    diagnostic_scales,
    feature_scales,
    masked_leaves,
    scaled_huber_loss,
    unnormalised_param_mask,
)

# Hyperparameters
# Gradient steps per environment step, counted over every environment at once: at the run's 64
# environments 1/64 is one gradient step per loop iteration, which is BTR's rr = 1 -- their
# replay period is their environment count, so their ratio is this one times the environments.
# A Fraction rather than a float because learn_step spends it: the whole steps it owes come due
# and the remainder is carried, and only exact arithmetic keeps a ratio like 1/3 from drifting
# over a run's millions of iterations.
REPLAY_RATIO = Fraction(1, 64)
BATCH_SIZE = 256
BUFFER_SIZE = 1 << 20  # 2 MebiTrans
TRAIN_START_BUF_SIZE = 200_000

PER_ALPHA = 0.2
# At 1.0 this cancels PER_ALPHA exactly -- sampling frequency times IS weight goes as
# p ** (PER_ALPHA * (1 - PER_BETA)) -- and prioritisation reaches only which transitions arrive
# together. At BTR's 0.2 the exponent is 0.16 and it reaches their expected contribution too,
# worth +50 mean evaluation score per matched 2M-env-step window (17 of 21 windows) over the
# pair of full runs in the provenance note. It is why train/grad_norm's median is 4.4 rather
# than 2.7, and why the batch exceeds GRADIENT_CLIPPING_MAX_NORM on 8.7% of steps rather than
# 0.9%; BTR clips at the same 10 with the same effective beta.
PER_BETA = 0.2
PER_EPSILON = 1e-6
# Batches whose priorities may be in flight past the ones an iteration dispatches for itself,
# which is the depth left over once a burst has filled the queue with its own steps and so the
# part that a rising REPLAY_RATIO does not eat. ``init`` sizes the queue from it and the
# environment count; at one gradient step an iteration that is a queue of three. See BTR.observe.
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
# The Huber knee, in units of the TD error. Inside it the pinball-weighted loss is an asymmetric
# *square*, whose minimiser is the tau-expectile, not the tau-quantile; outside it the loss is
# the pinball loss proper and the minimiser is the quantile. So this constant chooses which
# statistic the network fits, and at 1.0 against a raw |td| that spans 0.02-0.45 over a run it
# chooses the expectile nearly always. QR-DQN's footnote says where the 1.0 came from: DQN
# clipped the squared error to [-1, 1], which is identically a Huber with kappa = 1, and the
# value has been inherited ever since. It is tied to reward clipping's [-1, 1], not to anything
# about the scale of Q.
IQN_HUBER_LOSS_K = 1.0
# Applied before Adam, so it is the gradient Adam's moments see. That ordering is the reason to
# keep it: a spike admitted into nu is held for 1/(1 - ADAM_B2) = 1000 steps while mu forgets it
# in ten, which suppresses every legitimate update in between. Engaged on 8.7% of steps over the
# last full run, so it is part of the update rule here rather than a rare guard.
GRADIENT_CLIPPING_MAX_NORM = 10

INFERENCE_SYNC_FREQ = 3
TARGET_NETWORK_UPDATE_FREQ = 500
LAYER_NORM = False
IMPALA_SIZE_FACTOR = 2
# The trunk's convolutions compute in this and nothing else does; ImpalaCNNLarge says what that
# means for the parameters. Worth 1237 -> 2152 env steps/s, the largest single win the profiling
# found, and it costs nothing measurable in score or in the diagnostics over a full budget --
# see the precision block below. float16 measures the same and is not used, since the residual
# stream has no loss scaling behind it and bfloat16 is the one that keeps float32's exponent
# range.
# Spelled as a string so that ``hyperparameters`` below sweeps it up with the rest: a dtype
# object is not one of the scalar types it keeps, and a run whose trunk precision were missing
# from the HParams table could not be told from a float32 one afterwards.
IMPALA_DTYPE = "bfloat16"  # BTR: float32

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

# Gradient steps between writes, per group. What _train_step returns -- train/* and plasticity/*
# -- is eleven series over arrays the step already holds, so only its transfer is worth thinning.
# The per-parameter scales are ~130 series that move over a whole run and say nothing new at a
# fine cadence, and learn_step computes them on this cadence rather than every step. At these
# numbers a 200M-frame run writes ~15 MB where one logging everything at 25 wrote 384.
TRAIN_LOG_FREQ = 100
SCALE_LOG_FREQ = 2500

# Every constant above, and the architecture below, checked against BTR (arXiv:2411.03820,
# Table D6 and Appendices E/H) and its reference implementation (github.com/VIPTankz/BTR)
# on 2026-09-12. All of it matches bar what is explicitly marked otherwise. Table D6 gives PER
# an alpha and nothing else, so PER_BETA and PER_EPSILON come from PER.py in the reference
# implementation: its eps is 1e-6, and its effective beta is 0.2 because its IS weights are
# ``(capacity * prob) ** -self.alpha``, with alpha standing where beta belongs -- the comment
# beside that line calls it an accident kept for performing better. ``self.beta`` itself is
# vestigial: initialised to 0.4, reset to 0 by every insert, and read by no live code.
#
# Scores checked against BTR's own results.csv (200 Breakout evaluations, 100 episodes each),
# from the 212M-frame run in logs/breakout_2026-09-13_14-12-41: 557 against their 549 past 40M
# frames, and 584 against their 601 over the closing evaluations. What remains of the shortfall
# is all before 40M frames -- 253 against 383 -- where this run is about 5M frames later than
# BTR in taking off and then converges onto its curve by 15M. PER_BETA is not what that is; see
# the evaluation-protocol entry in the backlog for what is.

# +--------------+
# | AI Generated |
# +--------------+

# === Correctness to settle ===
# TODO: the Munchausen log-policy term is read off target_qnet, which is what the Munchausen
#       paper's Eq. 7 and BTR's Eq. E1 both write. BTR's *code* reads it off the online net
#       (`q_k_target = self.net.qvals(states)` in Agent.py), and that is the version their
#       published curve came from. Paper-vs-code, not ambiguity; the two differ by up to
#       TARGET_NETWORK_UPDATE_FREQ steps of staleness. Decide which to follow.
#       Their version is also the fast one, and by more than anything else on this list: the
#       online pass already computes q_quants.mean(-2) for policy_entropy, so the log-policy
#       comes off it for a stop_gradient and the third trunk pass goes. Measured at 22.6 ms
#       a gradient step against 26.2, -13.8% and the only double-digit saving the profiling
#       found. Do not take it for the speed -- but it is not a tie-break to leave out either.
# TODO: NoisyNets' sigma is drifting rather than being learned, and a remedy has to be picked.
#       Over two full runs, and unmoved by PER_BETA, every NoisyLinear's noisy_sigma_neg_frac
#       leaves 0 for 0.37-0.48 and its noisy_sigma_signed collapses to zero, while every sigma
#       tensor's adam_snr sits below the 0.18 iid-gradient floor and its adam_step_rel *rises*
#       -- x3 on value_linear1/kernel_sigma, since sigma's RMS shrinks underneath it.
#       noisy_sigma_ratio ends at 0.024 and 0.071 on the two output layers, so whatever the head
#       is exploring with by then, it is not this. Both candidates are optimiser-side: weight
#       decay on the sigma parameters, or BTR's one draw per gradient step in place of
#       NoisyLinear's per-__call__ redraw, which is what has the three forward passes in a
#       train_step disagreeing about eps and is the reason this run is more exposed than BTR is.
#       The schedule is not the problem.

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
# Measured 2026-09-13, 3080 + 24-thread host, 64 envs, batch 256, 1:1 act/train, steady state,
# each step of the chain over two or three runs:
#   1034 env steps/s   where the profiling started
#   1112              --xla_gpu_force_conv_nhwc, since removed: see IMPALA_DTYPE
#   1237-1261         the priority write-back queued, PRIO_QUEUE_SLACK
#   2152-2153         the trunk's convolutions in bfloat16, IMPALA_DTYPE
# That is 61.9 -> 29.7 ms an iteration, ~8.6k ALE frames/s, ~6.5h for 50M env steps / 200M frames.
# Held over a full run: 2089 env steps/s median across 6.9 h and 824k gradient steps, flat from
# the first decile to the last. The per-parameter scales moving to SCALE_LOG_FREQ is worth the
# last 5.9% of that -- they used to be reductions inside every gradient step -- and it also
# lifts the slow tail, p5 1906 against 1694 over windows of equal length. Compare rates only at
# equal LOG_FREQ: the window is the averaging interval, so a longer one hides brief dips.
#
# From a jax.profiler trace of gradient steps 20..80 (`--profile 20:60`), per iteration. The
# kernel durations are the device's own and stand as they are; every host span carries CUPTI's
# per-launch cost, which is now most of what the trace measures -- it reports a 54.6 ms iteration
# against the 29.7 the rate scalar sees -- so read a host span as its share of 54.6, never
# against the real wall:
#   device kernels  23.0 ms   1153 launches an iteration, median 1.3 us -- 77% of the real 29.7
#   learn_step      41.6 ms   host, of 54.6; almost all of it inside the executable call
#   env.step         2.9 ms   host, ALE; it launches nothing, so the inflation misses it
#   observe          0.3 ms   host: 0.22 ms of it in the buffer's save_step, 0.04 in update_prios
#   act transfer     0.1 ms   host; the overlap works, this one is free
#
# Re-measured 2026-09-18 on the same host with LAYER_NORM on, evaluation and checkpointing off,
# by perf/bench_loop.py: 2253 env steps/s an iteration of 28.4 ms. Each part timed alone, device
# under saturation and host after a barrier, so the two columns are what each really costs:
#   _train_step        25-26 ms device, 12-16 ms host
#   _act, 64 envs       0.9 ms device,  3.9 ms host
#   sync_qnet             0 device,     6.7 ms host, before it stopped being jitted
#   buf.sample                          1.0-2.0 ms host, flat in how full the buffer is
#   the batch's h2d                     1.2 ms host, obs and next_obs together
#   env.step, 64 envs                   1.6 ms host
#   save_step                           0.07 ms host; update_prios 0.006
# The device is 26.2 ms of the 28.4 and the host, at ~22, has slack -- so every host saving
# below measured zero on the wall, and only device work is worth cutting. _train_step's device
# time is its three trunk passes and nothing else: 4.4 ms for the Munchausen target pass on
# obs, 4.4 for the target on next_obs, 16.7 for the online forward and backward, summing to the
# whole within 0.5%. The trunk holds ~30 TFLOP/s of bfloat16, about half of what this card does
# dense, so there is no large kernel-level win left under it either.
#
# Gradient steps an iteration against env steps/s, everything else held (perf/sweep.sh):
#   1 -> 2253     2 -> 1150     4 -> 602     8 -> 294
# Linear to within 5%, and that 5% is the per-iteration cost amortising rather than slack being
# taken up: a gradient step costs its own 25 ms of device wherever it lands. A 200M-frame budget
# is 6.2 h at 1, 12.1 at 2, 23.1 at 4, 47.2 at 8.
#
# Evaluation runs nearly always: one 8-episode evaluation on Phoenix outlasted 6250 gradient
# steps, so both EVAL_FREQ windows inside a 280 s run were skipped, and it is the episode's
# length that does that -- binding _act below did not change it. What it *costs* the loop is
# _act's host time, Python holding the GIL against the loop's own dispatch rather than the two
# contending for the device, and binding the net took 60% of that back:
#   2133 -> 2206 env steps/s with evaluation on   (Mann-Whitney p = 5e-10 over ~70 windows)
#   2257 -> 2255 with it off                      (p = 0.72)
# A 5.5% tax down to 2.2%, and nothing either way in the training loop itself -- which is the
# whole shape of this block in one measurement: host savings show up only where a thread is
# host-bound, and the loop is not.
# Convolutions still carry the device, at ~18% of it, but no longer dominate it; what is left of
# the idle is the launch-bound tail, 81% of launches under 5 us for 6% of device time. Shrinking
# that means fewer kernels rather than faster ones, and nothing below has found a way to.
#
# Measured and rejected, so they do not get tried twice:
#   nnx.cached_partial          cuts dispatch Python calls 4x (1.5M -> 369k per 25 steps) and
#                               does not move wall time -- but not because the traversal is
#                               cheap, which is what this entry used to say. cProfile puts 81%
#                               of a _train_step dispatch in nnx's graph flatten/unflatten, and
#                               binding the same four modules into an equivalent kernel takes
#                               its host cost 10.4 -> 3.0 ms. The host is simply not what the
#                               loop waits on. Note for whoever tries it again: _train_step
#                               takes its modules keyword-only and cached_partial binds only
#                               positional ones, so it cannot be applied as the signature
#                               stands. The place it would pay is the evaluator's thread, where
#                               the same traversal is spent per env step against the GIL.
#   XLA command buffers         no change, at any --xla_gpu_enable_command_buffer setting.
#   TF32 matmul precision       JAX_DEFAULT_MATMUL_PRECISION moves dots, not cuDNN's convs,
#                               which pick their own math type; no change either way.
#   dropping the diagnostics    0.8 ms/step of 59, so utils' "costs nothing" claim holds.
#   dropping SpectralNorm       2.0 ms/step of 59. Not where the time goes.
#   a queue of 5 batches        1280 against 1237 and 1261 at a queue of 3, inside the
#                               run-to-run spread, so PRIO_QUEUE_SLACK keeps the shallower queue
#                               and its smaller stale window.
#   --xla_gpu_force_conv_nhwc   worth 8% while the trunk was float32, nothing once it is
#                               bfloat16: XLA already lays 16-bit convolutions out NHWC.
# TODO: donate_argnums on the jitted steps to avoid param copies.
# TODO: the head and the loss are still float32. They are ~1/6 of the trunk's FLOPs, so this is
#       worth far less than the trunk was, and the quantile axes make the loss the part of the
#       graph where precision is least obviously free. Measure before assuming it is.

# === Precision ===
# IMPALA_DTYPE is settled over a full budget. Two runs matched step for step --
# logs/breakout_2026-09-12_02-33-13 in float32 to 951,987 gradient steps and
# logs/breakout_2026-09-13_04-18-22 in bfloat16 to 857,335 -- put bfloat16 at 513 +/- 74 mean
# evaluation score over the 10M-55M env step overlap against float32's 480 +/- 68 (Welch
# p = 0.002; one seed each, so this reads as no worse rather than better), at 2020 against 1063
# env steps/s. spectral_sigma and noisy_sigma agree between the two to better than 3% at 52M env
# steps, and dormant_frac and dead_frac stay at zero in both.
# Per batch, against an identical float32 net: the trunk's features carry 0.7% relative L2 error
# and its weight gradients 0.9% median, both a small multiple of bfloat16's own 0.39% rounding
# unit. Nothing compounds by construction either -- the parameters and Adam's moments are
# float32, so no rounding accumulates in the weights, and bfloat16 keeps float32's exponent range
# so nothing underflows.

# === Reproduction / tuning ===
# TODO: evaluation is not scored the way BTR scores it, and the two deviations both push the
#       early curve down. Evaluator plays the training policy with NoisyLinear drawing live,
#       where BTR's prep_evaluation deep-copies the net and zeroes weight_epsilon/bias_epsilon,
#       so it scores the mean weights; and EVAL_NUM_ENVS gives 8 episodes against their 100,
#       which puts +/-26 on each point when the eval-to-eval sd is 73. The first of those is
#       worth the most early, exactly where the remaining shortfall is: noisy_sigma_ratio opens
#       near 1.0, so the scored policy is at its noisiest while BTR's is deterministic, and the
#       curves converge as the ratio falls to 0.02-0.37. Settle it by scoring one checkpoint
#       both ways before reading anything more into the early frames.
# TODO: EPS_GREEDY_START/END are 0.0 where BTR anneals 1.0 -> 0.01 over 2M env steps and holds a
#       floor until half the budget. The deviation is deliberate and marked, but it is the other
#       candidate for the early gap and is untested.
# TODO: the action gap this run trains to is ~26x below BTR's, and since ACT_TEMPERATURE is
#       calibrated against nothing else, both the soft draw and any argument about it rest on a
#       number never measured here. Theorem 2 of Munchausen (arXiv:2007.14430) puts M-VI's
#       converged gap at (1 + alpha) / (1 - alpha) = 19x that of the MDP its entropy coefficient
#       (1 - alpha) * tau regularises -- a property of the task and the state, so BTR and this
#       run, sharing Phoenix and both constants, are predicted the same gap. They do not have
#       it: BTR measures 0.282, or 9.4 tau, where softmax(q / tau) is an argmax to three digits,
#       and this run's policy_entropy and run/action_log_prob imply ~0.011, or 0.4 tau,
#       shrinking over a run rather than growing. Neither side of the theorem is observable
#       here, so the 26x is approximation error or a difference in which states get visited, and
#       one candidate for the first is that the alpha * tau * log pi(a|s) bonus separates actions
#       only to the extent log pi is spread across them: a flat policy and a flat gap hold each
#       other in place, and acting from that same flat policy is what this run adds to BTR.
#       Measure the gap directly before touching ACT_TEMPERATURE or ACT_USE_SOFT_POLICY: mean
#       top-two |advantage| over a few thousand states with the NoisyNet draw zeroed, which is
#       the quantity BTR's 0.282 is. Score the same checkpoint under soft tau, argmax, and
#       argmax + eps = 0.01 while it is loaded and the evaluation entry above is settled with it.
#       (1 - alpha) * tau is not the alternative it looks like -- it is the coefficient of the
#       *unamplified* reference MDP, so dividing this q by it double-counts Theorem 2's factor.

# === Experimental / longer term ===
# TODO: LayerNorm, per BTR's Appendix H, which says where to put it and reports a clear gain.
#       Highest-value experiment on this list. Does LN make SN redundant? Try LN-only, SN-only,
#       both. XQC rates LN the worst of its three normalisers, but it is comparing LN against BN
#       in a continuous-control critic, not against nothing in an Atari trunk, and Appendix H's
#       gain is measured on this benchmark -- so LN stays ahead of BN below in the queue.
# TODO: plasticity. Two full runs have now been read and the pathology is neither dormancy nor
#       rank: dead_frac is zero throughout, dormant_frac is non-zero only before 0.31M env
#       steps, and effective_rank turns out to be bounded by the batch rather than the trunk --
#       see utils.feature_scales for the reference points, which put a *trained* trunk above a
#       fresh one on the same observations. What is unbounded is scale, and it is the
#       denominator of the effective learning rate that eats it: weights/global_norm goes
#       61 -> 162 and is still linear at the end, value_linear1/kernel_mean's RMS goes x11 and
#       quantile_embedding/linear/bias x89, while adam_snr is flat, so the output layers'
#       adam_step_rel falls x6 (3.1e-4 -> 4.6e-5) with no change in the gradient at all. That is
#       the ELR collapse Lyle et al. and XQC (arXiv, ICLR 2026 submission) describe, and weight
#       decay below is the half of it this run can act on. Resets address what none of this
#       shows; for scale, a 50M-step run is ~824k updates, ~21 at BBF's 40k cadence.
# TODO: weight decay, as the cheap half of the above, and now unblocked -- the PER_BETA
#       comparison it was waiting on is done. WEIGHT_DECAY is wired and sits at 0.0; the comment
#       on it says what to turn it on at and why that is not the value BTR's AdamW null result
#       covers. XQC's own answer to the same growth is to project every dense weight to the unit
#       sphere each step, which is strictly better at holding ||theta|| fixed but is only legal
#       because its BN layers make the network scale-invariant; nothing here is, so decay is the
#       available version.
# TODO: IQN_HUBER_LOSS_K, which decides whether the head fits quantiles or expectiles -- see the
#       constant. Two things follow from being in the quadratic branch nearly always. The
#       estimand is wrong: IQN reads Q as the mean over uniform tau of the learned statistics,
#       which is E[Z] for quantiles by construction and is not for expectiles, and on a
#       return-shaped skewed distribution the kappa = 1 fit comes out biased *upward* -- the
#       direction a max-over-actions bootstrap compounds -- where kappa = 0.1 does not. And the
#       estimand drifts, because kappa is an absolute threshold against a raw |td| that moves
#       18.5x over a run and a Q that moves 200x, so the statistic being fitted is not the same
#       one at 1M steps as at 800k. Sweep kappa in {1, 0.3, 0.1}; the scale-free version is to
#       set the knee from the batch's own |td| rather than from a constant, which is the fix the
#       drift argues for and is a deviation from BTR either way.
#       kappa = 0 is the pinball loss proper and is a real configuration, not a failure mode:
#       QR-DQN reports it as QR-DQN-0 at 199% median / 881% mean human-normalised against
#       QR-DQN-1's 211% / 915% over 57 games. It costs ~6% and it is what the unbiased estimand
#       actually requires. Two things to expect from it: mean |dl/dprediction| at the optimum
#       rises x1.89 against kappa = 1, which at a median grad_norm of 4.4 puts most steps into
#       GRADIENT_CLIPPING_MAX_NORM; and the gradient no longer shrinks as the prediction
#       approaches the target, which is one of the two candidate mechanisms for the sub-floor
#       adam_snr in the plasticity entry above, so read that diagnostic after changing this.
#       The loss already reaches the huber_k = 0 limit, so the sweep is one constant.
# TODO: GRADIENT_CLIPPING_MAX_NORM is a fixed absolute bound with the same scale problem as the
#       Huber knee, and Adam already bounds each coordinate's step to ~lr regardless of gradient
#       size, so the clip looks redundant. It is not, for one reason: it sits before Adam, and
#       what it protects is nu. Removing it is still worth measuring, because the alternative
#       reading of 8.7% engagement is that the clip is what is keeping adam_snr where it is.
#       Measure it as an ablation with adam_snr and adam_step_rel read beside the score, not as
#       a cleanup.
# TODO: XQC's three components, as the next place to look after LayerNorm, and in this order.
#       (1) BN in place of LN: XQC's Hessian eigenspectra put BN an order of magnitude below LN
#       on condition number, which is the opposite of this field's usual default. It does not
#       port cheaply -- their BN needs CrossQ's joined forward pass over (s,a) and (s',a') to
#       keep its running statistics honest, and that trick exists because CrossQ has no target
#       network, where this has one at 500-step staleness. (2) The categorical CE loss. Half of
#       XQC's case for it does not apply here: their contrast is against an MSE with unbounded
#       dL/dy_hat, and the quantile Huber already bounds it by pinball_weight * huber_k <= 1.
#       What is left is the Hessian structure, and the cost is replacing IQN with C51 outright.
# TODO: resets, at a higher replay ratio. What the ratio costs is settled -- the performance
#       block has it, and it is linear -- so the open half is what it buys, and on a 3080 the
#       budget is what decides: 12 h a run at two gradient steps an iteration, 23 at four.
#       BBF's 40k cadence is ~21 resets over a 50M-step run.
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
        # Each stream is normalised on the way into its hidden activation and nowhere else:
        # the layer after it has to carry the scale of a return, so normalising there would
        # take away the one job it has. ``epsilon`` is Flax's default here as in the trunk --
        # see ImpalaBlock for what that is worth against torch's.
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
    ``policy_entropy``. Both are therefore a floor on how stochastic the draw is, which is the
    direction the paragraph below rests on.

    Soft sampling is self-correcting where epsilon-greedy is merely immune: a frozen state pays 0
    for every action forever, so Q(s, .) goes flat and the softmax goes uniform exactly where the
    policy is stuck. The epsilon path is kept, per-env and independent of the weights, as the one
    escape a confidently wrong Q cannot fool; turn it back on if a freeze recurs.

    How sharp the draw comes out depends on the top-two action gap over ``temperature`` and on
    nothing else. That gap is not the scale of Q and cannot be read off it: it is the difference
    of two discounted sums sharing a 1 / (1 - DISCOUNT) = 333-step horizon, so it is a small
    residual of a large common value, and the dueling head splits the two by construction. On
    Phoenix the gap implied by ``run/action_log_prob`` and ``policy_entropy`` runs about 0.4
    tau, putting p(chosen action) near 0.6 against the 0.125 of a uniform draw over its 8, and it
    shrinks over a run rather than growing. BTR reports 0.282 for the same quantity on the same
    game, which is 9.4 tau and would make this softmax an argmax to three digits. So the draw is
    genuinely stochastic at this temperature and nothing in the loop holds it anywhere in
    particular: read the two series above before assuming either limit.
    """
    obs = norm_obs(obs)

    # It's okay to use advantages here because softmax is shift invariant.
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

    The scalars here are the ones that need something the step holds and nothing else does -- the
    batch, its gradient, the trunk activations of the online pass. The per-parameter scales are
    read off the module and the optimizer instead, so ``BTR.learn_step`` takes those outside this
    function and pays for them only on the steps that write them.
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
        # They arrive in IMPALA_DTYPE, which is the trunk's and nothing else's: the head casts
        # them on its own way in, and float32 is what the scales below are read in.
        q_quants, outputs = nnx.capture(
            qnet, nnx.Intermediate, method_outputs=nnx.Intermediate
        )(obs, samples=samples, rngs=rngs)
        features = jnp.asarray(outputs["decoder"]["__call__"][0], jnp.float32)
        action_q_quants = jnp.take_along_axis(q_quants, actions[..., None], -1)

        policy_log_probs = nnx.log_softmax(q_quants.mean(-2) / temperature)
        policy_entropy = -jnp.sum(jnp.exp(policy_log_probs) * policy_log_probs, -1)

        unscaled_td_error = target_q_quants[..., None, :] - action_q_quants
        pinball_weights = jnp.abs(samples[..., None] - (unscaled_td_error < 0))
        td_error = pinball_weights * unscaled_td_error

        total_loss = pinball_weights * scaled_huber_loss(unscaled_td_error, huber_k)
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

    # grad_norm is the whole gradient, which is what the clip reads. The ratio below runs over
    # the parameters weights/global_norm keeps on both sides -- halves covering different
    # parameters are not a ratio -- and takes them here rather than from diagnostic_scales, which
    # learn_step runs after the update and on a coarser cadence.
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
        "train/grad_norm": grad_norm,
        "train/grad_to_weight": weight_grad_norm / weight_norm,
    }
    stats |= feature_scales(features, num_channels=qnet.decoder.out_channels)

    opt.update(qnet, grads)

    return new_prios, stats


# Once every SCALE_LOG_FREQ steps
@nnx.jit(graph=False)
def _scale_stats(qnet: QNet, opt: nnx.Optimizer[QNet]) -> dict[str, jax.Array]:
    """
    The per-parameter scale diagnostics, keyed by scalar name and left on the device for the
    caller to drain with the rest.

    A kernel of its own rather than part of ``_train_step``: it reads the network and the
    optimizer and needs neither a batch nor a gradient, so the cadence it runs on is free to be
    far coarser than the step's. It still has to be jitted -- dispatched one at a time, its ~130
    reductions over every parameter and every Adam moment cost 48 ms of host against 1.4 here.

    ``graph=False`` puts both arguments through as plain pytrees, which is all this needs: it
    returns scalars and mutates nothing, so none of graph mode's reference semantics apply, while
    graph mode writes the whole 21 MB of parameters and moments back out as fresh buffers on
    every call.
    """
    return diagnostic_scales(qnet) | adam_optim_scales(opt, qnet)


def sync_qnet(qnet: QNet, target_qnet: QNet) -> None:
    """
    Point ``target_qnet`` at the arrays ``qnet`` holds now.

    Not jitted, and not a copy: JAX arrays are immutable, so the target keeps what the source
    held at the call and a later ``opt.update`` cannot reach through the alias. Checked against
    the jitted copy this replaces -- parameters, the actions the two nets draw, the flags, and
    SpectralNorm's carried ``u`` under acting.

    It drops 6.7 ms of host a sync, all of it nnx graph traversal for a call that did nothing on
    the device, and that buys no wall time: 2253 against 2247 env steps/s. The loop is
    device-bound with host to spare, which is what every host figure in the performance block
    comes to.

    An alias is what a donated buffer would break, so the donate_argnums the backlog wants has
    to leave the two nets this hands out of it.
    """
    nnx.update(target_qnet, nnx.state(qnet))


def make_inference_qnet(
    num_actions: int, obs_shape: tuple[int, int, int], rngs: nnx.Rngs
) -> QNet:
    """
    A network built to act with and never to train, for the paths that load weights into one
    rather than cloning a live net through ``sync_qnet``.

    ``use_running_average`` is the flag sync sets, and only that flag: ``eval()`` sets more than
    this one and would change what ``act`` does. Its SpectralNorm layers then read the carried
    ``u`` without advancing it.
    """
    qnet = QNet(num_actions, obs_shape=obs_shape, layer_norm=LAYER_NORM, rngs=rngs)
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
    act: Callable[..., tuple[jax.Array, jax.Array]],
    obs: npt.NDArray[Any],
    rngs: nnx.Rngs,
    *,
    num_frames: int,
    evaluation: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """
    The acting half of both ``BTR`` and ``QNetPolicy``: pick the epsilon this many frames in,
    then draw.

    ``act`` is ``_act`` with its net already bound, not the net, because both callers act with
    one net for their whole life and binding it is what keeps nnx from walking its 159 nodes
    again on every env step -- see where they bind it.

    The log-probability handed back, which the run logs as ``run/action_log_prob``, is the one
    of the action actually taken, so it is the behaviour policy's own surprise rather than a
    property of the learned q. It is conditional
    on this call's NoisyNet draw, so it reads more deterministic than the marginal policy is;
    see ``_act``. A ceiling on the soft policy's collapse toward argmax, not a measurement of it.
    """
    epsilon = eval_epsilon_at(num_frames) if evaluation else epsilon_at(num_frames)
    actions, log_probs = act(
        obs,
        rngs=rngs,
        epsilon=epsilon,
    )
    return actions, {"log_prob": log_probs}


class QNetPolicy(Policy):
    """
    A ``QNet`` that is only ever loaded into, for scoring and for watching.

    Its RNG streams are its own, so acting with it does not disturb the draws a training run is
    making on another thread.
    """

    def __init__(
        self, num_actions: int, obs_shape: tuple[int, int, int], *, frames_per_step: int
    ) -> None:
        self._frames_per_step = frames_per_step
        self._rngs = create_rngs(0)
        self.qnet = make_inference_qnet(num_actions, obs_shape, self._rngs)
        # The binding caches the walk over this net's graph, not its values: the clone
        # cached_partial keeps holds the same Variable objects, so ``load`` writing into them
        # is seen here. It is ``self.qnet`` never being *replaced* that keeps this valid.
        # Worth 3.8 -> 1.6 ms of host an act, which on the evaluator's thread is that much
        # less GIL held against the training loop's own dispatch.
        self._act = nnx.cached_partial(_act, self.qnet)

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
        self._max_prio = 1.0
        self._buf: ReplayBuffer | None = None
        self._prio_queue = deque()
        self._grad_steps_owed = Fraction(0)

        self.rngs = create_rngs(seed)
        self.qnet = QNet(
            num_actions, obs_shape=obs_shape, layer_norm=LAYER_NORM, rngs=self.rngs
        )
        # Neither clone is checkpointed, since sync_qnet rebuilds both from qnet. The flag is
        # what stops their passes advancing the power iteration outside a gradient step, and
        # nnx.update writes variables and nothing else, so every later sync leaves it standing
        # and it is set here once rather than on each of them.
        self.target_qnet = nnx.clone(self.qnet)
        self.inference_qnet = nnx.clone(self.qnet)
        for net in (self.target_qnet, self.inference_qnet):
            net.set_attributes(use_running_average=True, raise_if_not_found=False)
        # As in QNetPolicy: the walk over the acting net's graph is cached here rather than
        # repeated every iteration, and the sync_qnet writing into that net is seen through it
        # because both hold the same Variable objects. Only _act can be bound this way --
        # _train_step's optimizer changes graph structure across opt.update, and it holds qnet,
        # so neither can be cached without the other. There is nothing to win there anyway;
        # the performance block says why.
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

    @property
    @override
    def num_updates(self) -> int:
        return self._num_updates

    @property
    @override
    def hyperparameters(self) -> dict[str, bool | int | float | str]:
        """
        Swept up wholesale from the module's constants, so the table stays correct as that block
        grows. TRAIN_LOG_FREQ and SCALE_LOG_FREQ come with them: they do not change what a run
        means, but they do set how densely its curves are drawn, which is worth having beside
        the curves being compared.

        A Fraction goes in as the float nearest it, since the table holds scalars and nothing
        reads these back: the exactness is for the arithmetic, not for the record of it.
        """
        return {
            f"btr/{name}": float(value) if isinstance(value, Fraction) else value
            for name, value in globals().items()
            if name.isupper() and isinstance(value, bool | int | float | str | Fraction)
        }

    @override
    def init(self, obs: npt.NDArray[Any]) -> None:
        num_envs = obs.shape[0]
        # A burst settles its own oldest batch, so the queue has to carry what an iteration
        # dispatches before PRIO_QUEUE_SLACK is worth anything: sized this way, the write-back
        # reads a batch the device has had a whole iteration to finish however high the ratio is,
        # where a fixed depth would put it on one dispatched moments earlier.
        self._prio_queue_len = math.ceil(REPLAY_RATIO * num_envs) + PRIO_QUEUE_SLACK

        # An environment can only be drawn from once it holds enough slots for a draw's frame
        # stack to read backwards over and its rollout to read forwards over, which is
        # obs_stack + N_STEP - 1 transitions. learn_step's gate counts the whole buffer, so a
        # TRAIN_START_BUF_SIZE spread thinly enough over environments would open it on a buffer
        # no draw can serve; refused here, where both numbers are known, rather than surfacing
        # as a ValueError out of the first sample halfway into a run.
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
        Settle up for the gradient steps that have finished, then store.

        Settling up is a device transfer, and reading back the step just dispatched costs the
        host the whole of that step -- 12.9 ms an iteration, against 0.22 ms in ``save_step``
        below. An iteration's own gradient steps and ``PRIO_QUEUE_SLACK`` more stay in flight
        instead, so the transfer reads a step the device finished an iteration ago and the
        sampling and upload in the following ``learn_step`` overlap the one still running.

        What that spends is the exactness of the write-back. A batch is drawn against one write
        head and written back against a head at most ``PRIO_QUEUE_SLACK`` save_steps further on
        -- fewer the higher ``REPLAY_RATIO`` is, since the queue then turns over inside an
        iteration rather than across several -- so a slot within that distance of wrapping has
        been recycled underneath it and its new occupant takes the old one's priority. A
        save_step advances a head once, twice across an episode boundary, which bounds it at
        ``2 * PRIO_QUEUE_SLACK`` slots of an environment's capacity -- four of 16384 here, so
        0.06 draws of a 256 batch. It cannot make an unsamplable slot drawable: the buffer
        rejects a proposal on slot type, not on priority.

        The other half of that is what the delayed batches are still worth to the sampler: until
        its write-back lands, a batch keeps the priority it was drawn on, so the transitions of
        the gradient steps still in flight are offered at the error that selected them rather
        than the one they now have, and are drawn again at it. ``PER_ALPHA`` is low enough that
        the draw is near uniform, which puts that at 256 * 256 / 1048576 = 0.06 of a batch per
        following draw, 0.13 over the two in flight here -- the same order as the recycling
        above, and growing with the queue the same way, ratio included.
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
        # What the next learn_step has to spend. Accrued here rather than counted there because
        # this is the call that sees how much experience arrived.
        self._grad_steps_owed += REPLAY_RATIO * self._buf.num_envs

    def _write_back_prios(self) -> None:
        """
        Settle the batches that have been in flight longest, leaving room in the queue for one
        more. Called once an iteration from ``observe``, which is where the transfer is oldest
        and so cheapest, and again from ``learn_step`` for the steps of a burst past the first,
        which have no ``observe`` between them to settle up for them.
        """
        assert self._buf is not None, "_write_back_prios before init"

        while self._prio_queue_len <= len(self._prio_queue):
            indices, prios = self._prio_queue.popleft()
            self._buf.update_prios(indices, jax.device_get(prios))

    @override
    def learn_step(self) -> tuple[int, dict[str, jax.Array]]:
        """
        Every whole gradient step REPLAY_RATIO owes for the experience observed so far, taken
        once the buffer has filled, dispatched and left in flight, and the scalars to log for
        them -- empty on the steps between writes.

        Which steps those are is decided here rather than by the run because it is a property of
        what the scalars cost to produce. What _train_step returns is computed either way, since
        it is reductions over arrays that step already holds; TRAIN_LOG_FREQ only thins the
        transfer. _scale_stats is a kernel of its own, so SCALE_LOG_FREQ decides whether its ~130
        reductions run at all. A call taking several steps reports the newest of them that lands
        on a write, so the cadences stay what those constants say whatever the ratio is.
        """
        assert self._buf is not None, "learn_step before init"
        if len(self._buf) < TRAIN_START_BUF_SIZE:
            # The fill's own steps are dropped rather than carried: they would come due in one
            # burst against a buffer holding a single policy's experience, which is the opposite
            # of what a replay ratio is for. A resumed run's refill is dropped the same way.
            self._grad_steps_owed = Fraction(0)
            return 0, {}

        # The remainder is carried, so a ratio that owes a fraction of a step an iteration
        # spends it on the iteration it completes it rather than rounding it away.
        num_steps, self._grad_steps_owed = divmod(self._grad_steps_owed, 1)

        stats: dict[str, jax.Array] = {}
        for _ in range(num_steps):
            # Once an iteration this is the no-op that observe already did; past that it is what
            # keeps the batches in flight bounded by the queue rather than by the ratio.
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

            new_prios, step_stats = _train_step(
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
            self._prio_queue.append((batch_indices, new_prios))

            self._num_updates += 1
            if self._num_updates % INFERENCE_SYNC_FREQ == 0:
                sync_qnet(self.qnet, self.inference_qnet)
            if self._num_updates % TARGET_NETWORK_UPDATE_FREQ == 0:
                sync_qnet(self.qnet, self.target_qnet)

            writes_scales = self._num_updates % SCALE_LOG_FREQ == 0
            if writes_scales:
                # Dispatched like the step above and drained with the rest, and reading the state
                # that step just wrote, so the host pays one launch for all ~130 of them.
                step_stats |= _scale_stats(self.qnet, self.opt)
            if writes_scales or self._num_updates % TRAIN_LOG_FREQ == 0:
                stats = step_stats

        return num_steps, stats

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
            self._obs_shape,
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
        # Read at init, since the buffer holds it read-only once it exists.
        self._max_prio = restored["replay"]["max_prio"]
