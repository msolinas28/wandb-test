import flax.linen as nn
import jax
import jax.numpy as jnp

from jVMC_exp.global_defs import DT_PARAMS_REAL

def log_cosh(x):
    two = jnp.asarray(2, dtype=x.real.dtype)
    half_log_two = jnp.log(two)
    sgn_x = -two * jnp.signbit(x.real) + 1
    sgn_x = sgn_x.astype(x.dtype)
    x = x * sgn_x

    return x + jnp.log1p(jnp.exp(-two * x)) - half_log_two

class RBM(nn.Module):
    n_hidden: int
    bias: bool = False
    param_dtype: jnp.dtype = DT_PARAMS_REAL

    @nn.compact
    def __call__(self, s):
        layer = nn.Dense(
            self.n_hidden, 
            self.bias, 
            param_dtype=self.param_dtype,
            kernel_init=jax.nn.initializers.lecun_normal(dtype=self.param_dtype),
            bias_init=jax.nn.initializers.zeros
        )

        x = log_cosh(layer(s))

        return jnp.sum(x, dtype=x.dtype)

    @property
    def label(self):
        return "RBM"

    @property
    def info(self):
        return dict(
            n_hidden=self.n_hidden,
            bias=self.bias,
        )