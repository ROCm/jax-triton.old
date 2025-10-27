"""
Fused Softmax
"""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl
import numpy as np


@triton.jit
def softmax_kernel(input_ptr, output_ptr,
                   input_row_stride, output_row_stride,
                   n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    # Each block processes rows with a step size of num_blocks for load balancing,
    # while vectorizing column operations within each row.
    for row_idx in tl.range(row_start, n_rows, row_step):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: jnp.ndarray) -> jnp.ndarray:
    n_rows, n_cols = x.shape
    strides = jt.strides_from_shape(x.shape)
    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    grid = (n_rows,)
    return jt.triton_call(
        x,  # Input array
        kernel=softmax_kernel,
        out_shape=out_shape,
        grid=grid,
        # Kernel parameters
        input_row_stride=strides[0],
        output_row_stride=strides[0],
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )


def main(unused_argv):
    x = jnp.ones((1024, 512), dtype="float32")
    y = jax.jit(softmax)(x).block_until_ready()

    expected = jax.nn.softmax(x).block_until_ready()
    np.testing.assert_allclose(y, expected)
    print("Test passed!")

if __name__ == "__main__":
    from absl import app
    app.run(main)
