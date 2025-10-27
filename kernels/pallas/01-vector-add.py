import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import numpy as np


def add_vectors_kernel(x_ref, y_ref, z_ref):
    z_ref[...] = x_ref[...] + y_ref[...]


@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
    assert x.shape == y.shape
    BLOCK_SIZE = 16

    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    grid = (x.size // BLOCK_SIZE,)
    return pl.pallas_call(
        add_vectors_kernel,
        out_shape=out_shape,
        grid=grid,
    )(x, y)


def main(unused_argv):
    x = jnp.arange(512)
    y = jnp.arange(512, 1024)
    z = add_vectors(x, y)

    expected = x + y
    np.testing.assert_allclose(z, expected)
    print("Test passed!")


if __name__ == "__main__":
    from absl import app
    app.run(main)
