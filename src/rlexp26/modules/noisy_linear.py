import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class NoisyLinear(eqx.Module):
    """
    Factorised NoisyNets linear layer, block-diagonal over its kernels' first axis, the
    groups: group g reads the g-th contiguous slice of the inputs and writes the g-th of the
    outputs. The noise is factorised over all the inputs and outputs whatever the grouping, and
    folded into the weights, ``mu + sigma * eps_in eps_out^T``: a ``key`` the vmap leaves
    unbatched is one draw for the whole batch and one matmul.

    With ``weight_norm`` the layer has no bias and divides each output by its mean weights'
    norm, the noise term as well, so the noise keeps the share sigma / mu its parameters hold,
    and rescaling a neuron's mean and sigma together leaves the function unchanged.
    """

    kernel_mean: jax.Array
    kernel_sigma: jax.Array
    bias_mean: jax.Array | None
    bias_sigma: jax.Array | None
    weight_norm: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.Array,
        groups: int = 1,
        use_bias: bool = True,
        weight_norm: bool = False,
    ) -> None:
        if in_features % groups or out_features % groups:
            raise ValueError(
                f"groups={groups} must divide in_features={in_features} "
                f"and out_features={out_features}"
            )
        if weight_norm and use_bias:
            raise ValueError("a weight-normalised NoisyLinear has no bias")
        init_range = (in_features // groups) ** -0.5
        sigma = 0.5 * init_range
        shape = (groups, in_features // groups, out_features // groups)
        kernel_key, bias_key = jr.split(key)
        self.kernel_mean = jr.uniform(
            kernel_key, shape, minval=-init_range, maxval=init_range
        )
        self.kernel_sigma = jnp.full_like(self.kernel_mean, sigma)
        self.bias_mean = self.bias_sigma = None
        if use_bias:
            self.bias_mean = jr.uniform(
                bias_key, (out_features,), minval=-init_range, maxval=init_range
            )
            self.bias_sigma = jnp.full_like(self.bias_mean, sigma)
        self.weight_norm = weight_norm

    @jax.named_scope("NoisyLinear")
    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        groups, in_group, out_group = self.kernel_mean.shape
        in_key, out_key = jr.split(key)
        eps_in, eps_out = (
            jnp.sign(e) * jnp.sqrt(jnp.abs(e))
            for e in (
                jr.normal(in_key, (groups, in_group, 1)),
                jr.normal(out_key, (groups, 1, out_group)),
            )
        )
        kernel = self.kernel_mean + self.kernel_sigma * eps_in * eps_out
        y = jnp.einsum("gi,gio->go", x.reshape(groups, in_group), kernel).reshape(-1)
        if self.bias_mean is not None and self.bias_sigma is not None:
            y += self.bias_mean + eps_out.reshape(-1) * self.bias_sigma
        if self.weight_norm:
            y /= jnp.linalg.norm(self.kernel_mean, axis=1).reshape(-1)
        return y
