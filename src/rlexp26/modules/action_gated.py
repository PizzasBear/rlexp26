from collections.abc import Hashable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .batch_renorm import BatchRenorm
from .noisy_linear import NoisyLinear
from .swiglu import NormalisedSwiGLUBlock


class ActionGatedHead(eqx.nn.StatefulLayer):
    """
    MXQ2's head, from one sample's features to log-probabilities ``[num_actions, num_atoms]``.
    A weight-normalised grouped NoisyLinear stem is multiplied by each action's gate vector,
    used at a root-mean-square of 1, and everything after runs once per action with shared
    weights: a NormalisedSwiGLUBlock, a LayerNorm without a weight, a SiLU, and a bias-free
    linear readout through a batch renorm per atom, whose bias starts at ``init_logits``.

    The batch renorms read their batch over ``axis_name``, the caller's vmapped batch axis, and
    over the actions, which are vmapped here under the name ``"action"``. ``mask``,
    ``[num_actions]``, picks the actions whose rows the batch's moments are read from.
    """

    stem: NoisyLinear
    gates: jax.Array
    block: NormalisedSwiGLUBlock
    norm: eqx.nn.LayerNorm
    readout: eqx.nn.Linear
    readout_norm: BatchRenorm

    def __init__(
        self,
        in_features: int,
        num_actions: int,
        init_logits: jax.Array,
        *,
        gate_init_std: float,
        readout_scale_init: float,
        axis_name: Hashable,
        stem_groups: int = 4,
        features: int = 512,
        hidden: int = 1536,
        key: jax.Array,
    ) -> None:
        stem_key, gate_key, block_key, readout_key = jr.split(key, 4)
        axis_names = (axis_name, "action")
        self.stem = NoisyLinear(
            in_features,
            features,
            groups=stem_groups,
            use_bias=False,
            weight_norm=True,
            key=stem_key,
        )
        self.gates = 1 + gate_init_std * jr.normal(gate_key, (num_actions, features))
        self.block = NormalisedSwiGLUBlock(
            features, hidden, axis_name=axis_names, key=block_key
        )
        self.norm = eqx.nn.LayerNorm(features, eps=1e-6, use_weight=False)
        self.readout = eqx.nn.Linear(
            features, init_logits.size, use_bias=False, key=readout_key
        )
        readout_norm = BatchRenorm(init_logits.size, axis_names)
        self.readout_norm = eqx.tree_at(
            lambda norm: (norm.weight, norm.bias),
            readout_norm,
            (jnp.full_like(init_logits, readout_scale_init), init_logits),
        )

    def project_kernels(self) -> "ActionGatedHead":
        """
        Every neuron's weights rescaled to the square root of their fan-in, a weight RMS of 1:
        the NoisyLinears' means with their sigmas, the readout's rows, and each action's gates.
        Each is normalised upon use, by weight normalisation, by the batch renorm it feeds or
        by the gates' own RMS, so this leaves the function unchanged, the batch renorms'
        running moments aside.
        """

        def project(linear: NoisyLinear) -> NoisyLinear:
            norms = jnp.linalg.norm(linear.kernel_mean, axis=1, keepdims=True)
            scale = linear.kernel_mean.shape[1] ** 0.5 / norms
            return eqx.tree_at(  # type: ignore[no-any-return]
                lambda linear: (linear.kernel_mean, linear.kernel_sigma),
                linear,
                (linear.kernel_mean * scale, linear.kernel_sigma * scale),
            )

        def unit_rms(rows: jax.Array) -> jax.Array:
            return rows * jax.lax.rsqrt(jnp.mean(jnp.square(rows), -1, keepdims=True))

        head = eqx.tree_at(
            lambda head: (head.stem, head.block.linear0, head.block.linear1),
            self,
            replace_fn=project,
        )
        return eqx.tree_at(  # type: ignore[no-any-return]
            lambda head: (head.gates, head.readout.weight), head, replace_fn=unit_rms
        )

    @jax.named_scope("ActionGatedHead")
    def __call__(
        self,
        x: jax.Array,
        state: eqx.nn.State,
        *,
        key: jax.Array,
        mask: jax.Array | bool = True,
    ) -> tuple[jax.Array, eqx.nn.State]:
        stem_key, block_key = jr.split(key)
        gates = self.gates * jax.lax.rsqrt(
            jnp.mean(jnp.square(self.gates), -1, keepdims=True)
        )
        x = self.stem(x, key=stem_key) * gates

        def action(
            x: jax.Array, state: eqx.nn.State, mask: jax.Array
        ) -> tuple[jax.Array, eqx.nn.State]:
            x, state = self.block(x, state, key=block_key, mask=mask)
            x = self.readout(jax.nn.silu(self.norm(x)))
            return self.readout_norm(x, state, mask=mask)

        logits, state = jax.vmap(
            action, in_axes=(0, None, 0), out_axes=(0, None), axis_name="action"
        )(x, state, jnp.broadcast_to(mask, gates.shape[:1]))
        return jax.nn.log_softmax(logits), state
