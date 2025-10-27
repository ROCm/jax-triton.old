import tabulate
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def dropout_kernel(x_ref, p_ref, x_keep_ref, y_ref):
    x = x_ref[:]
    p = p_ref[()]
    x_keep = x_keep_ref[:]

    y = jnp.where(x_keep, x / (1 - p), 0.0)
    y_ref[:] = y


@jax.jit
def dropout(x, p, seed):
    BLOCK_SIZE = 16

    key = jax.random.PRNGKey(seed)
    random = jax.random.uniform(key, shape=x.shape, dtype=x.dtype)
    x_keep = random > p

    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    grid = (x.size // BLOCK_SIZE,)
    return pl.pallas_call(
        dropout_kernel,
        out_shape=out_shape,
        grid=grid
    )(x, p, x_keep)


def main(unused_argv):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (512,), dtype=jnp.float32)
    y = dropout(x, p=0.5, seed=123).block_until_ready()
    print(
        tabulate.tabulate([
            ["input"] + x.tolist(),
            ["output (seed = 123)"] + y.tolist(),
        ]))


if __name__ == "__main__":
    from absl import app
    app.run(main)
