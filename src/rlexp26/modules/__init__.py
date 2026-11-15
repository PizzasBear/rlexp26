"""
The network's building blocks, free of any algorithm's hyperparameters. Every module reads one
sample, the convolutions CHW, and is vmapped over the batch; BatchRenorm reads its batch across
the vmapped axes it names, as ``eqx.nn.BatchNorm`` does. ``vendored.spectral_norm`` is vendored
Equinox -- see that module's header.
"""

from .action_gated import ActionGatedHead
from .batch_renorm import AxisName, BatchRenorm
from .categorical_head import CategoricalHead
from .dueling_iqn import DuelingIQNHead
from .impala import ImpalaBlock, ImpalaCNNLarge, ImpalaResSubBlock
from .iqn import IQNCosineEmbedding
from .noisy_linear import NoisyLinear
from .norm_impala import (
    NormalisedImpalaBlock,
    NormalisedImpalaCNN,
    NormalisedResidualBlock,
    filter_norms,
)
from .qnet import BATCH_AXIS, QNet
from .swiglu import NormalisedSwiGLUBlock

__all__ = [
    "BATCH_AXIS",
    "ActionGatedHead",
    "AxisName",
    "BatchRenorm",
    "CategoricalHead",
    "DuelingIQNHead",
    "IQNCosineEmbedding",
    "ImpalaBlock",
    "ImpalaCNNLarge",
    "ImpalaResSubBlock",
    "NoisyLinear",
    "NormalisedImpalaBlock",
    "NormalisedImpalaCNN",
    "NormalisedResidualBlock",
    "NormalisedSwiGLUBlock",
    "QNet",
    "filter_norms",
]
