from functools import partial
from typing import Any, assert_never

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from .action_gated import ActionGatedHead
from .categorical_head import CategoricalHead
from .dueling_iqn import DuelingIQNHead
from .impala import ImpalaCNNLarge
from .norm_impala import NormalisedImpalaCNN

# The name QNet vmaps its batch under, for the batch renorms inside it to read.
BATCH_AXIS = "batch"


class QNet(eqx.Module):
    """
    A trunk and a head, read on a batch of stacked observations ``[batch, stack, H, W]`` in
    0..255. A quantile head returns its values at ``taus``, ``[batch, levels, actions]``; a
    categorical one returns log-probabilities ``[batch, actions, atoms]``. ``key`` is one draw
    of noise for the whole batch. ``taken``, the actions a loss reads, is what the batch
    renorms after an ActionGatedHead's gates take their batch's moments from.
    ``advantage_only`` is a DuelingIQNHead's.
    """

    trunk: ImpalaCNNLarge | NormalisedImpalaCNN
    head: DuelingIQNHead | CategoricalHead | ActionGatedHead

    def __call__(
        self,
        obs: jax.Array,
        state: eqx.nn.State,
        *,
        key: jax.Array,
        taus: jax.Array | None = None,
        taken: jax.Array | None = None,
        advantage_only: bool = False,
    ) -> tuple[jax.Array, eqx.nn.State]:
        assert not advantage_only or isinstance(self.head, DuelingIQNHead)
        x, state = jax.vmap(
            self.trunk, in_axes=(0, None), out_axes=(0, None), axis_name=BATCH_AXIS
        )(obs.astype(jnp.float32) / 255, state)
        x = x.astype(jnp.float32)
        head = self.head
        match head:
            case DuelingIQNHead():
                assert taus is not None
                return jax.vmap(partial(head, key=key, advantage_only=advantage_only))(
                    x, taus
                ), state
            case CategoricalHead():
                return jax.vmap(partial(head, key=key))(x), state
            case ActionGatedHead():
                num_actions = head.gates.shape[0]
                mask = (
                    jnp.ones((x.shape[0], num_actions), bool)
                    if taken is None
                    else jax.nn.one_hot(taken, num_actions, dtype=bool)
                )
                return jax.vmap(
                    lambda x, mask: head(x, state, key=key, mask=mask),
                    out_axes=(0, None),
                    axis_name=BATCH_AXIS,
                )(x, mask)
            case _:
                assert_never(head)

    def parameter_groups(self) -> PyTree[str]:
        """
        A label per floating array, over the tree ``eqx.filter(self, eqx.is_inexact_array)``
        keeps: ``<part>/<kind>``. The part is ``trunk``, ``readout`` (the head's output layer) or
        ``head`` (the rest of it). The kind is ``kernel`` for a weight matrix or convolution
        kernel, ``noisy_kernel`` for a NoisyLinear's mean kernel, ``sigma`` for its kernel's
        noise scale, and ``other`` for the rest: biases (a convolution's has three axes), norms'
        scales and shifts.
        """
        head = self.head
        match head:
            case DuelingIQNHead():
                readout = {"value1", "advantage1"}
            case CategoricalHead():
                readout = {"linear1"}
            case ActionGatedHead():
                readout = {"readout", "readout_norm"}
            case _:
                assert_never(head)

        def label(path: tuple[Any, ...], leaf: jax.Array) -> str:
            names = [k.name for k in path if isinstance(k, jax.tree_util.GetAttrKey)]
            part = (
                "trunk"
                if names[0] == "trunk"
                else "readout"
                if names[1] in readout
                else "head"
            )
            if names[-1] == "kernel_sigma":
                kind = "sigma"
            elif leaf.ndim < 2 or names[-1].startswith("bias"):
                kind = "other"
            elif names[-1] == "kernel_mean":
                kind = "noisy_kernel"
            else:
                kind = "kernel"
            return f"{part}/{kind}"

        return jax.tree_util.tree_map_with_path(
            label, eqx.filter(self, eqx.is_inexact_array)
        )

    def weight_stats(self) -> dict[str, jax.Array]:
        """
        What the head's own weights say, by the tag it is logged under: for an
        ActionGatedHead, ``weights/action_gate_cosine``, the mean cosine between two actions'
        gate vectors, 1 while every action reads the stem alike.
        """
        head = self.head
        match head:
            case DuelingIQNHead() | CategoricalHead():
                return {}
            case ActionGatedHead():
                unit = head.gates / jnp.linalg.norm(head.gates, axis=-1, keepdims=True)
                cosines = unit @ unit.T
                num_actions = unit.shape[0]
                return {
                    "weights/action_gate_cosine": (jnp.sum(cosines) - num_actions)
                    / (num_actions * (num_actions - 1))
                }
            case _:
                assert_never(head)
