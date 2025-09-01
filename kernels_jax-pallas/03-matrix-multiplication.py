import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import numpy as np
from functools import partial


def matmul_kernel(x_ref, y_ref, z_ref, activation):
    z_ref[...] = activation(x_ref[...] @ y_ref[...])


def matmul(x: jax.Array, y: jax.Array, activation) -> jax.Array:
    assert x.shape[1] == y.shape[0], "Incompatible dimensions"
    M, K = x.shape
    _, N = y.shape
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16

    out_shape = jax.ShapeDtypeStruct((M, N), x.dtype)
    grid = (M // BLOCK_SIZE_M,
            N // BLOCK_SIZE_N)
    return pl.pallas_call(
        partial(matmul_kernel, activation=activation),
        out_shape=out_shape,
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_SIZE_M, K), lambda i, j: (i, 0)),
            pl.BlockSpec((K, BLOCK_SIZE_N), lambda i, j: (0, j))
        ],
        out_specs=pl.BlockSpec((BLOCK_SIZE_M, BLOCK_SIZE_N), lambda i, j: (i, j))
    )(x, y)


def main(unused_argv):
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(k1, (512, 512), dtype=jnp.float32)
    y = jax.random.normal(k2, (512, 512), dtype=jnp.float32)
    z = jax.jit(matmul, static_argnums=2)(x, y, activation=jax.nn.leaky_relu).block_until_ready()

    expected = jax.nn.leaky_relu(x @ y).block_until_ready()
    np.testing.assert_allclose(z, expected, rtol=1e-2, atol=1e-2)
    print("Test passed!")


if __name__ == "__main__":
    from absl import app
    app.run(main)
