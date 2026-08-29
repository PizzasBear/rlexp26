# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is copy of some Flax primitives with bugs fixes applied.

import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
    Array,
    Axes,
    Dtype,
    Initializer,
    PromoteDtypeFn,
)
from jax import lax


def _canonicalize_axes(rank: int, axes: Axes) -> tp.Tuple[int, ...]:
    """Returns a tuple of deduplicated, sorted, and positive axes."""
    if not isinstance(axes, tp.Iterable):
        axes = (axes,)
    return tuple({rank + axis if axis < 0 else axis for axis in axes})


def _l2_normalize(x, axis=None, eps=1e-12):
    """Normalizes along dimension `axis` using an L2 norm.
    This specialized function exists for numerical stability reasons.
    Args:
      x: An input ndarray.
      axis: Dimension along which to normalize, e.g. `1` to separately normalize
        vectors in a batch. Passing `None` views `t` as a flattened vector when
        calculating the norm (equivalent to Frobenius norm).
      eps: Epsilon to avoid dividing by zero.
    Returns:
      An array of the same shape as 'x' L2-normalized along 'axis'.
    """
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


class WeightNorm(nnx.Module):
    """L2 weight normalization (https://arxiv.org/abs/1602.07868).

    Weight normalization normalizes the weight params so that the l2-norm of
    the matrix is equal to 1. This is implemented as a layer wrapper where
    each wrapped layer will have its params l2-normalized before computing
    its ``__call__`` output.

    Example usage::

      >>> import jax
      >>> import numpy as np
      >>> from flax import nnx

      >>> class Foo(nnx.Module):
      ...   def __init__(self, rngs: nnx.Rngs):
      ...     self.normed_linear = nnx.WeightNorm(
      ...       nnx.Linear(8, 4, rngs=rngs),
      ...       variable_filter=nnx.PathContains('kernel'),
      ...       rngs=rngs,
      ...     )
      ...
      ...   def __call__(self, x: jax.Array) -> jax.Array:
      ...     return self.normed_linear(x)

      >>> rng = jax.random.key(42)
      >>> model = Foo(rngs=nnx.Rngs(rng))

      >>> x = jax.random.normal(rng, (5, 8))
      >>> y = model(x)
      >>> y.shape
      (5, 4)

      >>> w = model.normed_linear.layer_instance.kernel[...]
      >>> col_norms = np.linalg.norm(np.array(w), axis=0)
      >>> np.testing.assert_allclose(col_norms, np.ones(4))

    Args:
      layer_instance: The layer instance to wrap.
      feature_axes: The axes to normalize.
      use_scale: Whether to use a scale parameter.
      scale_init: The initializer for the scale parameter, by default ones.
      epsilon: The epsilon value for the normalization, by default 1e-12.
      dtype: The dtype of the result, by default infer from input and params.
      param_dtype: The dtype of the parameters, by default float32.
      variable_filter: The variable filter, by default ``nnx.PathContains('kernel')``.
      promote_dtype: function to promote the dtype of all input array arguments
        (including Variables accessed through ``self``) to the desired dtype. This
        is used internally by WeightNorm when normalizing weights.
      rngs: The rng key.
    """

    def __init__(
        self,
        layer_instance: nnx.Module,
        *,
        feature_axes: Axes | None = -1,
        use_scale: bool = True,
        scale_init: Initializer = initializers.ones,
        epsilon: float = 1e-12,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        variable_filter: nnx.filterlib.Filter = nnx.PathContains("kernel"),
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: rnglib.Rngs,
    ):
        self.layer_instance = layer_instance
        self.feature_axes = () if feature_axes is None else feature_axes
        self.use_scale = use_scale
        self.scale_init = scale_init
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.variable_filter = nnx.filterlib.to_predicate(variable_filter)
        self.promote_dtype = promote_dtype
        self.scales: tp.Optional[dict] = None

        if use_scale:
            state = nnx.state(self.layer_instance, nnx.Param)

            def init_scales(param):
                feature_axes = _canonicalize_axes(param.ndim, self.feature_axes)
                scale_shape = tuple(param.shape[ax] for ax in feature_axes)
                return nnx.Param(
                    scale_init(rngs["params"], scale_shape)
                )  # fix: added nnx.Param(...) here to make scales learnable

            self.scales = nnx.data(
                {
                    path: init_scales(param)
                    for path, param in nnx.to_flat_state(state)
                    if self.variable_filter(path, param)
                }
            )

    def _weightnorm_inplace(self, path, param):
        if not self.variable_filter(path, param):
            return

        if self.feature_axes is None:
            feature_axes = ()
            reduction_axes = tuple(range(param.ndim))
        else:
            feature_axes = _canonicalize_axes(param.ndim, self.feature_axes)
            reduction_axes = tuple(
                i for i in range(param.ndim) if i not in feature_axes
            )

        value_bar = _l2_normalize(param, axis=reduction_axes, eps=self.epsilon)

        if self.use_scale:
            if path not in self.scales:
                raise RuntimeError(
                    f"Could not find the scale corresponding to the param {path} "
                    "in scales dict. Parameters of the layer_instance should not change!"
                )
            scale_value = self.scales[path]

            if len(feature_axes) < param.ndim:
                broadcast_shape = [1] * param.ndim
                for ax in feature_axes:
                    broadcast_shape[ax] = param.shape[ax]
                scale_value = scale_value.reshape(broadcast_shape)
            value_bar = value_bar * scale_value

        cast_args = [param]
        if self.use_scale:
            cast_args.append(scale_value)

        final_dtype = dtypes.canonicalize_dtype(*cast_args, dtype=self.dtype)
        param.set_value(jnp.asarray(value_bar, final_dtype))

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        """Compute the l2-norm of the weights in ``self.layer_instance``
        and normalize the weights using this value before computing the
        ``__call__`` output.

        Args:
          *args: positional arguments to be passed into the call method of the
            underlying layer instance in ``self.layer_instance``.
          **kwargs: keyword arguments to be passed into the call method of the
            underlying layer instance in ``self.layer_instance``.

        Returns:
          Output of the layer using l2-normalized weights.
        """
        state = nnx.state(self.layer_instance)

        originals: list[tuple[nnx.Param, Array]] = []
        for path, param in nnx.to_flat_state(state):
            originals.append((param, param[...]))
            self._weightnorm_inplace(path, param)

        try:
            return self.layer_instance(x, *args, **kwargs)  # type: ignore
        finally:
            # fix: reset parameter values after we modified them in-place
            for param, original_value in originals:
                param[...] = original_value


class SpectralNorm(Module):
    """Spectral normalization.

    See:

    - https://arxiv.org/abs/1802.05957
    - https://arxiv.org/abs/1805.08318
    - https://arxiv.org/abs/1809.11096

    Spectral normalization normalizes the weight params so that the spectral
    norm of the matrix is equal to 1. This is implemented as a layer wrapper
    where each wrapped layer will have its params spectral normalized before
    computing its ``__call__`` output.

    .. note::
      The initialized variables dict will contain, in addition to a 'params'
      collection, a separate 'batch_stats' collection that will contain a
      ``u`` vector and ``sigma`` value, which are intermediate values used
      when performing spectral normalization. During training, we pass in
      ``update_stats=True`` so that ``u`` and ``sigma`` are updated with
      the most recently computed values using power iteration. This will
      help the power iteration method approximate the true singular value
      more accurately over time. During eval, we pass in ``update_stats=False``
      to ensure we get deterministic behavior from the model.

    Example usage::

      >>> from flax import nnx
      >>> import jax
      >>> rngs = nnx.Rngs(0)
      >>> x = jax.random.normal(jax.random.key(0), (3, 4))
      >>> layer = nnx.SpectralNorm(nnx.Linear(4, 5, rngs=rngs), rngs=rngs)
      >>> jax.tree.map(jax.numpy.shape, nnx.state(layer, nnx.Param))
      State({
        'layer_instance': {
          'bias': Param(
            value=(5,)
          ),
          'kernel': Param(
            value=(4, 5)
          )
        }
      })
      >>> y = layer(x, update_stats=True)

    Args:
      layer_instance: Module instance that is wrapped with SpectralNorm
      n_steps: How many steps of power iteration to perform to approximate the
        singular value of the weight params.
      epsilon: A small float added to l2-normalization to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      error_on_non_matrix: Spectral normalization is only defined on matrices. By
        default, this module will return scalars unchanged and flatten
        higher-order tensors in their leading dimensions. Setting this flag to
        True will instead throw an error if a weight tensor with dimension greater
        than 2 is used by the layer.
      update_stats: if True, the stored batch statistics will be
        used instead of computing the batch statistics on the input.
      rngs: rng key.
    """

    def __init__(
        self,
        layer_instance: Module,
        *,
        n_steps: int = 1,
        epsilon: float = 1e-12,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        error_on_non_matrix: bool = False,
        update_stats: bool = True,
        rngs: rnglib.Rngs,
    ):
        self.layer_instance = layer_instance
        self.n_steps = n_steps
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.error_on_non_matrix = error_on_non_matrix

        # We define here self.use_running_average attribute to make
        # .train() and .eval() work. These methods internally change
        # self.use_running_average to False in train mode and
        # to True in eval mode
        # update_stats flag has the opposite logic.
        self.use_running_average = not update_stats

        # Initialize batch stat variables:
        state = nnx.state(self.layer_instance, nnx.Param)

        def init_batch_stats(path, param):
            if param.ndim <= 1 or self.n_steps < 1:
                return None
            elif param.ndim > 2:
                if self.error_on_non_matrix:
                    raise ValueError(
                        f"Layer instance parameter is {param.ndim}D but error_on_non_matrix is True"
                    )
                else:
                    param = jnp.reshape(param, (-1, param.shape[-1]))

            path_u = path + ("u",)
            path_sigma = path + ("sigma",)

            key = rngs.params()
            return [
                (
                    path_u,
                    nnx.BatchStat(
                        initializers.normal()(
                            key, (1, param.shape[-1]), self.param_dtype
                        )
                    ),
                ),
                (
                    path_sigma,
                    nnx.BatchStat(initializers.ones(key, (), self.param_dtype)),
                ),
            ]

        batch_stats: dict[tuple, nnx.BatchStat] = {}
        for path, param in nnx.to_flat_state(state):
            batch_stats_per_param = init_batch_stats(path, param)
            if batch_stats_per_param is None:
                continue
            for new_path, bstat in batch_stats_per_param:
                batch_stats[new_path] = bstat

        self.batch_stats = nnx.data(batch_stats)

    def __call__(
        self,
        x,
        update_stats: tp.Optional[bool] = None,
    ):
        """Compute the largest singular value of the weights in ``self.layer_instance``
        using power iteration and normalize the weights using this value before
        computing the ``__call__`` output.

        Args:
          x: the input array of the nested layer
          update_stats: if True, update the internal ``u`` vector and ``sigma``
            value after computing their updated values using power iteration. This
            will help the power iteration method approximate the true singular value
            more accurately over time.

        Returns:
          Output of the layer using spectral normalized weights.
        """
        update_stats = first_from(
            update_stats,
            not self.use_running_average,
            error_msg="""No `update_stats` argument was provided to SpectralNorm
        as either a __call__ argument or __init__ argument.""",
        )
        state = nnx.state(self.layer_instance, nnx.Param)

        originals: list[tuple[nnx.Param, Array]] = []
        for path, param in nnx.to_flat_state(state):
            originals.append((param, param[...]))
            self._spectral_normalize_inplace(path, param, update_stats=update_stats)

        try:
            return self.layer_instance(x)
        finally:
            # fix: reset parameter values after we modified them in-place
            for param, original_value in originals:
                param[...] = original_value

    def _spectral_normalize_inplace(self, path, orig_param, update_stats):
        param = orig_param
        param_shape = param.shape
        if param.ndim <= 1 or self.n_steps < 1:
            return
        elif param.ndim > 2:
            if self.error_on_non_matrix:
                raise ValueError(
                    f"Layer instance parameter is {param.ndim}D but error_on_non_matrix is True"
                )
            else:
                param = jnp.reshape(param, (-1, param.shape[-1]))

        path_u = path + ("u",)
        path_sigma = path + ("sigma",)

        if path_u not in self.batch_stats:
            raise RuntimeError(
                f"Could not find the path for u batch stat corresponding to the param {path} "
                "in the batch stats dict. Parameters of the layer_instance should not change!"
            )
        if path_sigma not in self.batch_stats:
            raise RuntimeError(
                f"Could not find the path for sigma batch stat corresponding to the param {path} "
                "in the batch stats dict. Parameters of the layer_instance should not change!"
            )

        u = self.batch_stats[path_u][...]

        for _ in range(self.n_steps):
            v = _l2_normalize(jnp.matmul(u, param.T), eps=self.epsilon)
            u = _l2_normalize(jnp.matmul(v, param), eps=self.epsilon)

        u = lax.stop_gradient(u)
        v = lax.stop_gradient(v)

        sigma = jnp.matmul(jnp.matmul(v, param), u.T)[0, 0]
        param = param / jnp.where(sigma != 0, sigma, 1)
        param = param.reshape(param_shape)

        if update_stats:
            self.batch_stats[path_u][...] = u
            self.batch_stats[path_sigma][...] = sigma

        dtype = dtypes.canonicalize_dtype(param, u, v, sigma, dtype=self.dtype)
        orig_param[...] = jnp.asarray(param, dtype)
