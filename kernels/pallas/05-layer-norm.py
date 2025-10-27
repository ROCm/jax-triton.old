import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import numpy as np


eps=1e-5

def layer_norm_kernel(x_ref, w_ref, b_ref, y_ref):
    row_idx = pl.program_id(0)
    row = x_ref[row_idx, :]
    # Compute mean and variance
    mean = jnp.sum(row) / row.size
    var = jnp.sum((row - mean) * (row - mean)) / row.size
    rstd = 1 / jnp.sqrt(var + eps)
    # Normalize and apply linear transformation
    x_hat = (row - mean) * rstd
    y_ref[row_idx, :] = x_hat * w_ref[:] + b_ref[:]


@jax.jit
def layer_norm(x, w, b):
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    grid = (x.shape[0],)
    return pl.pallas_call(
        layer_norm_kernel,
        out_shape=out_shape,
        grid=grid
    )(x, w, b)


def main(unused_argv):
    key = jax.random.PRNGKey(0)
    key_x, key_w, key_b, key_dy = jax.random.split(key, 4)

    x = -2.3 + 0.5 * jax.random.normal(key_x, shape=(1151, 8192), dtype=jnp.float32)
    w = jax.random.uniform(key_w, shape=(8192,), dtype=jnp.float32)
    b = jax.random.uniform(key_b, shape=w.shape, dtype=jnp.float32)

    y = layer_norm(x, w, b).block_until_ready()
    print(y)


if __name__ == "__main__":
    from absl import app
    app.run(main)
