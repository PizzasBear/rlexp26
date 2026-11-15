"""The recipes of ``config`` as agents: which parts each one hands the ``QLearner``."""

import dataclasses
import math
from collections.abc import Callable
from typing import Any, assert_never

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from gymnasium.spaces import Box, Discrete, Space
from jaxtyping import PyTree

from .augment import random_intensity, random_shift
from .categorical import CategoricalReturns, gaussian_log_probs, sinh_atoms
from .config import BTR, MXQ1, MXQ2, Recipe
from .modules import (
    BATCH_AXIS,
    ActionGatedHead,
    CategoricalHead,
    DuelingIQNHead,
    ImpalaCNNLarge,
    NormalisedImpalaCNN,
    QNet,
)
from .optim import inverse_sqrt_schedule, scale_by_pooled_adam
from .qlearner import QLearner, ReturnDistribution
from .quantile import QuantileReturns

TRUNK_SIZE_FACTOR = 2
TRUNK_POOLED_SIZE = (6, 6)
# The trunk convolutions' compute dtype; see ImpalaCNNLarge.
TRUNK_DTYPE = jnp.bfloat16  # BTR: float32


def make_agent(
    hp: Recipe,
    observation_space: Space[Any],
    action_space: Space[Any],
    *,
    frames_per_step: int,
    seed: int,
) -> QLearner:
    """
    ``hp``'s recipe on one environment's spaces: stacked uint8 images ``(stack, H, W)`` and
    discrete actions. ``frames_per_step`` is the task's frameskip: BTR's epsilon schedules are
    quoted in frames and the run counts environment steps.
    """
    match observation_space, action_space:
        case Box(shape=(stack, height, width), dtype=np.uint8), Discrete():
            num_actions = int(action_space.n)
        case _:
            raise TypeError(
                f"{type(hp).__name__} acts on stacked uint8 images with discrete actions, "
                f"not {observation_space} and {action_space}"
            )
    trunk_key, head_key, agent_key = jr.split(jr.key(seed), 3)

    trunk: ImpalaCNNLarge | NormalisedImpalaCNN
    match hp:
        case BTR() | MXQ1():
            trunk = ImpalaCNNLarge(
                stack,
                size_factor=TRUNK_SIZE_FACTOR,
                pooled_size=TRUNK_POOLED_SIZE,
                dtype=TRUNK_DTYPE,
                key=trunk_key,
            )
        case MXQ2():
            trunk = NormalisedImpalaCNN(
                stack,
                in_spatial=(height, width),
                size_factor=TRUNK_SIZE_FACTOR,
                axis_name=BATCH_AXIS,
                dtype=TRUNK_DTYPE,
                key=trunk_key,
            ).project_kernels()
        case _:
            assert_never(hp)

    head: DuelingIQNHead | CategoricalHead | ActionGatedHead
    returns: ReturnDistribution
    act_returns: ReturnDistribution
    match hp:
        case BTR():
            head = DuelingIQNHead(
                trunk.out_features,
                num_actions,
                num_cosines=hp.iqn_num_cos,
                key=head_key,
            )
            returns = QuantileReturns(hp.iqn_train_samples, hp.iqn_huber_loss_k)
            act_returns = QuantileReturns(hp.iqn_act_samples, hp.iqn_huber_loss_k)
        case MXQ1() | MXQ2():
            cat = hp.categorical
            atoms = sinh_atoms(cat.cat_num_bins, cat.cat_asinh_bound)
            init_logits = jnp.maximum(
                gaussian_log_probs(atoms, 0.0, cat.cat_init_std),
                cat.cat_init_log_prob_floor,
            )
            init_logits -= init_logits.mean()
            categorical_returns = CategoricalReturns(
                atoms,
                floor=cat.cat_target_floor,
                target_std_scale=cat.cat_target_std_scale,
                target_moment_steps=cat.cat_target_moment_steps,
                cdf_bce_weight=cat.cdf_bce_weight,
            )
            match hp:
                case MXQ1():
                    head = CategoricalHead(
                        trunk.out_features, num_actions, init_logits, key=head_key
                    )
                    categorical_returns = dataclasses.replace(
                        categorical_returns,
                        floor_distill_weight=hp.cat_floor_distill_weight,
                        floor_distill_floor=hp.cat_floor_distill_floor,
                    )
                case MXQ2():
                    head = ActionGatedHead(
                        trunk.out_features,
                        num_actions,
                        init_logits,
                        gate_init_std=hp.cat_action_gate_init_std,
                        readout_scale_init=hp.cat_readout_scale_init,
                        axis_name=BATCH_AXIS,
                        key=head_key,
                    ).project_kernels()
                case _:
                    assert_never(hp)
            returns = act_returns = categorical_returns
        case _:
            assert_never(hp)
    net, state = eqx.nn.make_with_state(QNet)(trunk, head)

    optimiser: optax.GradientTransformation
    augment: Callable[[jax.Array, jax.Array], jax.Array] | None = None
    project: Callable[[PyTree[jax.Array]], PyTree[jax.Array]] | None = None
    match hp:
        case BTR() | MXQ1():
            optimiser = optax.adam(hp.common.learning_rate, eps=hp.adam_eps)
        case MXQ2():
            # head_input, head and trunk are what the projection holds at a fixed norm.
            labels = jax.tree.map(
                lambda _: "other", eqx.filter(net, eqx.is_inexact_array)
            )
            for group, where in (
                ("head_input", lambda net: net.head.stem),
                (
                    "head",
                    lambda net: (
                        net.head.block.linear0,
                        net.head.block.linear1,
                        net.head.gates,
                    ),
                ),
                (
                    "trunk",
                    lambda net: [
                        node
                        for node in jax.tree.leaves(
                            net.trunk, is_leaf=lambda x: isinstance(x, eqx.nn.Conv2d)
                        )
                        if isinstance(node, eqx.nn.Conv2d)
                    ],
                ),
                ("readout_kernel", lambda net: net.head.readout.weight),
                ("readout_scale", lambda net: net.head.readout_norm.weight),
                ("readout_bias", lambda net: net.head.readout_norm.bias),
            ):
                labels = eqx.tree_at(
                    where,
                    labels,
                    replace_fn=lambda node, group=group: jax.tree.map(  # type: ignore[misc]
                        lambda _: group, node
                    ),
                )
            head_input_rate, head_rate, trunk_rate, readout_kernel_rate = (
                inverse_sqrt_schedule(rate, decay_steps)
                for rate, decay_steps in (
                    (hp.head_input_learning_rate, hp.head_lr_decay_steps),
                    (hp.head_learning_rate, hp.head_lr_decay_steps),
                    (hp.trunk_learning_rate, hp.trunk_lr_decay_steps),
                    (hp.readout_kernel_learning_rate, hp.readout_kernel_lr_decay_steps),
                )
            )
            optimiser = optax.multi_transform(
                {
                    "head_input": optax.adam(
                        head_input_rate, eps=hp.projected_adam_eps
                    ),
                    "head": optax.adam(head_rate, eps=hp.projected_adam_eps),
                    "trunk": optax.adam(trunk_rate, eps=hp.projected_adam_eps),
                    "readout_kernel": optax.adam(readout_kernel_rate, eps=hp.adam_eps),
                    "readout_scale": optax.chain(
                        scale_by_pooled_adam(eps=hp.adam_eps),
                        optax.scale_by_learning_rate(hp.readout_scale_learning_rate),
                    ),
                    "readout_bias": optax.chain(
                        scale_by_pooled_adam(eps=hp.adam_eps),
                        optax.scale_by_learning_rate(hp.readout_bias_learning_rate),
                    ),
                    "other": optax.adam(hp.common.learning_rate, eps=hp.adam_eps),
                },
                # A function, because the labels are a QNet, which is callable itself.
                lambda _: labels,
            )

            pad, intensity = hp.augment_pad, hp.augment_intensity

            def shift_and_scale(obs: jax.Array, key: jax.Array) -> jax.Array:
                shift_key, intensity_key = jr.split(key)
                return random_intensity(
                    random_shift(obs, shift_key, pad), intensity_key, intensity
                )

            def project_kernels(params: PyTree[jax.Array]) -> PyTree[jax.Array]:
                return eqx.tree_at(
                    lambda params: (params.trunk, params.head),
                    params,
                    replace_fn=lambda module: module.project_kernels(),
                )

            augment, project = shift_and_scale, project_kernels
        case _:
            assert_never(hp)
    optimiser = optax.chain(
        optax.clip_by_global_norm(hp.common.gradient_clipping_max_norm), optimiser
    )

    epsilon: Callable[[int], float]
    eval_epsilon: Callable[[int], float]
    match hp:
        case BTR():
            # Geometric decay onto the end value, the continuous limit of BTR's update.

            def decayed(num_env_steps: int) -> float:
                frames = num_env_steps * frames_per_step
                if frames >= hp.eps_greedy_off_frames:
                    return 0.0
                decayed = math.exp(-frames / hp.eps_greedy_decay_frames)
                return hp.eps_greedy_end + decayed * (
                    hp.eps_greedy_start - hp.eps_greedy_end
                )

            def held(num_env_steps: int) -> float:
                frames = num_env_steps * frames_per_step
                if frames >= hp.eval_eps_greedy_off_frames:
                    return 0.0
                return hp.eval_eps_greedy

            epsilon, eval_epsilon = decayed, held
        case MXQ1() | MXQ2():
            epsilon = eval_epsilon = lambda num_env_steps: 0.0
        case _:
            assert_never(hp)

    hyperparameters = dict[str, Any]()
    for name, value in dataclasses.asdict(hp).items():
        hyperparameters |= value if isinstance(value, dict) else {name: value}
    return QLearner(
        net,
        state,
        optimiser,
        returns,
        act_returns,
        common=hp.common,
        replay=hp.replay,
        temperature=hp.munchausen_temperature,
        soft=not isinstance(hp, BTR),
        epsilon=epsilon,
        eval_epsilon=eval_epsilon,
        augment=augment,
        project=project,
        hyperparameters={
            f"qlearner/{name}": value
            if isinstance(value, bool | int | float | str)
            else str(value)
            for name, value in hyperparameters.items()
        },
        key=agent_key,
    )
