"""
Low-Memory Dropout
"""

import tabulate
import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl
import numpy as np


@triton.jit
def seeded_dropout_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    n_elements = x.size
    BLOCK_SIZE = 16

    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    return jt.triton_call(
        x,
        kernel=seeded_dropout_kernel,
        out_shape=out_shape,
        grid=grid,
        n_elements=n_elements,
        p=p,
        seed=seed,
        BLOCK_SIZE=BLOCK_SIZE
    )


def main(unused_argv):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (512,), dtype=jnp.float32)
    jit_dropout = jax.jit(seeded_dropout, static_argnums=(1, 2))
    output = jit_dropout(x, p=0.5, seed=123).block_until_ready()
    print(
        tabulate.tabulate([
            ["input"] + x.tolist(),
            ["output (seed = 123)"] + output.tolist(),
        ]))


if __name__ == "__main__":
    from absl import app
    app.run(main)
