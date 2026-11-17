"""
The recipes and their knobs. A recipe is a cappa subcommand whose defaults are its definition.
Every knob is a flag and, under its own name in capitals, an environment variable.
"""

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Annotated

import cappa
from cappa import Arg, Env


class ReplaySchedule(StrEnum):
    FIXED = "fixed"
    GPER = "gper"


@dataclass(frozen=True)
class Replay:
    # FIXED's gradient steps per environment step, over all environments at once.
    replay_ratio: Annotated[Fraction, Arg(default=Env("REPLAY_RATIO"))] = Fraction(
        1, 64
    )
    # FIXED: replay_ratio, drawn by priority p. GPER: drawn by p / (1 + a / gper_half_age), a
    # the age in env steps when last drawn, at a ratio of gper_reviews_per_doubling
    # ln(1 + n / gper_half_age) / (batch_size ln 2), n the transitions stored.
    replay_schedule: Annotated[ReplaySchedule, Arg(default=Env("REPLAY_SCHEDULE"))] = (
        ReplaySchedule.FIXED
    )
    gper_half_age: Annotated[int, Arg(default=Env("GPER_HALF_AGE"))] = 5_000
    gper_reviews_per_doubling: Annotated[
        float, Arg(default=Env("GPER_REVIEWS_PER_DOUBLING"))
    ] = 12.0
    batch_size: Annotated[int, Arg(default=Env("BATCH_SIZE"))] = 256
    # Gradient steps per dispatch. None is taken until this many are owed.
    burst_steps: Annotated[int, Arg(default=Env("BURST_STEPS"))] = 1
    buffer_size: Annotated[int, Arg(default=Env("BUFFER_SIZE"))] = 1 << 20
    train_start_buf_size: Annotated[int, Arg(default=Env("TRAIN_START_BUF_SIZE"))] = (
        200_000
    )
    per_alpha: Annotated[float, Arg(default=Env("PER_ALPHA"))] = 0.2
    per_beta: Annotated[float, Arg(default=Env("PER_BETA"))] = 0.2
    per_epsilon: Annotated[float, Arg(default=Env("PER_EPSILON"))] = 1e-6


@dataclass(frozen=True)
class Common:
    discount: Annotated[float, Arg(default=Env("DISCOUNT"))] = 0.997
    n_step: Annotated[int, Arg(default=Env("N_STEP"))] = 3
    munchausen_scaling_term: Annotated[
        float, Arg(default=Env("MUNCHAUSEN_SCALING_TERM"))
    ] = 0.9
    munchausen_clipping_val: Annotated[
        float, Arg(default=Env("MUNCHAUSEN_CLIPPING_VAL"))
    ] = -1.0
    # Gradient steps between copies of the online net into the target net.
    target_network_update_freq: Annotated[
        int, Arg(default=Env("TARGET_NETWORK_UPDATE_FREQ"))
    ] = 500
    learning_rate: Annotated[float, Arg(default=Env("LEARNING_RATE"))] = 1e-4
    # Applied before Adam.
    gradient_clipping_max_norm: Annotated[
        float, Arg(default=Env("GRADIENT_CLIPPING_MAX_NORM"))
    ] = 10.0


@cappa.command(name="btr", default_long=True)
@dataclass(frozen=True)
class BTR:
    """Beyond the Rainbow (arXiv:2411.03820): IQN, dueling noisy heads, epsilon-greedy."""

    common: cappa.Destructured[Common] = Common()
    replay: cappa.Destructured[Replay] = Replay()

    adam_eps: Annotated[float, Arg(default=Env("ADAM_EPS"))] = 0.005 / 256
    munchausen_temperature: Annotated[
        float, Arg(default=Env("MUNCHAUSEN_TEMPERATURE"))
    ] = 0.03
    iqn_num_cos: Annotated[int, Arg(default=Env("IQN_NUM_COS"))] = 64
    iqn_train_samples: Annotated[int, Arg(default=Env("IQN_TRAIN_SAMPLES"))] = 8
    iqn_act_samples: Annotated[int, Arg(default=Env("IQN_ACT_SAMPLES"))] = 8
    iqn_huber_loss_k: Annotated[float, Arg(default=Env("IQN_HUBER_LOSS_K"))] = 1.0
    # Training epsilon decays geometrically from start onto end with decay_frames as its 1/e
    # time, and is 0 from off_frames. All in ALE frames.
    eps_greedy_start: Annotated[float, Arg(default=Env("EPS_GREEDY_START"))] = 1.0
    eps_greedy_end: Annotated[float, Arg(default=Env("EPS_GREEDY_END"))] = 0.01
    eps_greedy_decay_frames: Annotated[
        int, Arg(default=Env("EPS_GREEDY_DECAY_FRAMES"))
    ] = 8_000_000
    eps_greedy_off_frames: Annotated[int, Arg(default=Env("EPS_GREEDY_OFF_FRAMES"))] = (
        100_000_000
    )
    eval_eps_greedy: Annotated[float, Arg(default=Env("EVAL_EPS_GREEDY"))] = 0.01
    eval_eps_greedy_off_frames: Annotated[
        int, Arg(default=Env("EVAL_EPS_GREEDY_OFF_FRAMES"))
    ] = 125_000_000


@dataclass(frozen=True)
class Categorical:
    # The atoms: cat_num_bins of them at sinh(u), u uniform on [-cat_asinh_bound,
    # cat_asinh_bound].
    cat_num_bins: Annotated[int, Arg(default=Env("CAT_NUM_BINS"))] = 255
    cat_asinh_bound: Annotated[float, Arg(default=Env("CAT_ASINH_BOUND"))] = 20.0
    # The initial return distribution, the same for every state and action:
    # N(0, cat_init_std^2), its log-probability per atom floored at cat_init_log_prob_floor.
    cat_init_std: Annotated[float, Arg(default=Env("CAT_INIT_STD"))] = 1.0
    cat_init_log_prob_floor: Annotated[
        float, Arg(default=Env("CAT_INIT_LOG_PROB_FLOOR"))
    ] = -45.0
    # The factor on the relative CDF-BCE, beside a cross-entropy of weight 1.
    cdf_bce_weight: Annotated[float, Arg(default=Env("CDF_BCE_WEIGHT"))] = 0.3
    # Every probability read off the head, and every target, has this subtracted, is clipped
    # at 0 and is renormalised.
    cat_target_floor: Annotated[float, Arg(default=Env("CAT_TARGET_FLOOR"))] = 1e-6
    # The moment-matched projection's factor on the Bellman target's standard deviation, and
    # the fixed-point steps that solve for the shrink.
    cat_target_std_scale: Annotated[float, Arg(default=Env("CAT_TARGET_STD_SCALE"))] = (
        0.95
    )
    cat_target_moment_steps: Annotated[
        int, Arg(default=Env("CAT_TARGET_MOMENT_STEPS"))
    ] = 2


@cappa.command(name="mxq1", default_long=True)
@dataclass(frozen=True)
class MXQ1:
    """BTR with a categorical head on sinh atoms, CE + relative CDF-BCE, and a soft policy."""

    common: cappa.Destructured[Common] = Common()
    replay: cappa.Destructured[Replay] = Replay()
    categorical: cappa.Destructured[Categorical] = Categorical()

    adam_eps: Annotated[float, Arg(default=Env("ADAM_EPS"))] = 1e-8  # BTR: 0.005 / 256
    # The acting policy's too: actions are drawn from softmax(q / munchausen_temperature).
    munchausen_temperature: Annotated[
        float, Arg(default=Env("MUNCHAUSEN_TEMPERATURE"))
    ] = 0.006  # BTR: 0.03

    # The factor on the mean, over the actions not taken, of CDF-BCE against their own
    # stop-gradient prediction floored at cat_floor_distill_floor.
    cat_floor_distill_weight: Annotated[
        float, Arg(default=Env("CAT_FLOOR_DISTILL_WEIGHT"))
    ] = 0.3
    cat_floor_distill_floor: Annotated[
        float, Arg(default=Env("CAT_FLOOR_DISTILL_FLOOR"))
    ] = 1e-6


@cappa.command(name="mxq2-alpha9-gi0.3-rscale0", default_long=True)
@dataclass(frozen=True)
class MXQ2:
    """MXQ1 on a normalised trunk, with an action-gated SwiGLU head and a batch-renorm readout."""

    common: cappa.Destructured[Common] = Common()
    replay: cappa.Destructured[Replay] = Replay()
    categorical: cappa.Destructured[Categorical] = Categorical()

    adam_eps: Annotated[float, Arg(default=Env("ADAM_EPS"))] = 1e-8  # BTR: 0.005 / 256
    # The acting policy's too: actions are drawn from softmax(q / munchausen_temperature).
    munchausen_temperature: Annotated[
        float, Arg(default=Env("MUNCHAUSEN_TEMPERATURE"))
    ] = 0.006  # BTR: 0.03

    # The action gates at init: 1 + this * N(0, 1) per entry.
    cat_action_gate_init_std: Annotated[
        float, Arg(default=Env("CAT_ACTION_GATE_INIT_STD"))
    ] = 0.3
    # The readout batch renorm's scale at init.
    cat_readout_scale_init: Annotated[
        float, Arg(default=Env("CAT_READOUT_SCALE_INIT"))
    ] = 0.0
    # The training batch's observations are padded by augment_pad pixels and cropped back at
    # a random offset, then scaled by 1 + augment_intensity * clip(N(0, 1), -2, 2).
    augment_pad: Annotated[int, Arg(default=Env("AUGMENT_PAD"))] = 4
    augment_intensity: Annotated[float, Arg(default=Env("AUGMENT_INTENSITY"))] = 0.05
    # Each group has its own Adam. A rate with decay steps is rate / sqrt(1 + t / steps) at
    # gradient step t, constant at inf. learning_rate is every other parameter's.
    head_input_learning_rate: Annotated[
        float, Arg(default=Env("HEAD_INPUT_LEARNING_RATE"))
    ] = 5e-3
    head_learning_rate: Annotated[float, Arg(default=Env("HEAD_LEARNING_RATE"))] = 5e-3
    # Both head groups'.
    head_lr_decay_steps: Annotated[float, Arg(default=Env("HEAD_LR_DECAY_STEPS"))] = (
        20_000
    )
    trunk_learning_rate: Annotated[float, Arg(default=Env("TRUNK_LEARNING_RATE"))] = (
        2e-3
    )
    trunk_lr_decay_steps: Annotated[float, Arg(default=Env("TRUNK_LR_DECAY_STEPS"))] = (
        200_000
    )
    readout_kernel_learning_rate: Annotated[
        float, Arg(default=Env("READOUT_KERNEL_LEARNING_RATE"))
    ] = 1e-3
    readout_kernel_lr_decay_steps: Annotated[
        float, Arg(default=Env("READOUT_KERNEL_LR_DECAY_STEPS"))
    ] = 20_000
    # The readout batch renorm's, under pooled Adam.
    readout_scale_learning_rate: Annotated[
        float, Arg(default=Env("READOUT_SCALE_LEARNING_RATE"))
    ] = 3e-4
    readout_bias_learning_rate: Annotated[
        float, Arg(default=Env("READOUT_BIAS_LEARNING_RATE"))
    ] = 3e-3
    # The head and trunk groups' epsilon. adam_eps is every other group's.
    projected_adam_eps: Annotated[float, Arg(default=Env("PROJECTED_ADAM_EPS"))] = 1e-10


Recipe = BTR | MXQ1 | MXQ2
