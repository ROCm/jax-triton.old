import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import numpy as np


def softmax_kernel(x_ref, y_ref):
    x = x_ref[:]
    x_minus_max = x - jnp.max(x)
    
    numerator = jnp.exp(x_minus_max)
    denominator = jnp.sum(numerator)
    y = numerator / denominator
    y_ref[:] = y


@jax.jit
def softmax(x: jax.Array) -> jax.Array:
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    grid = (x.shape[0],)
    return pl.pallas_call(
        softmax_kernel,
        out_shape=out_shape,
        grid=grid,
    )(x)


def main(unused_argv):
    x = jnp.ones((1024, 512), dtype="float32")
    y = softmax(x).block_until_ready()

    expected = jax.nn.softmax(x).block_until_ready()
    np.testing.assert_allclose(y, expected)
    print("Test passed!")


if __name__ == "__main__":
    from absl import app
    app.run(main)
