"""
Layer Normalization
"""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Y,  # pointer to the output
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


def layer_norm(x, w, b, eps, BLOCK_SIZE):
    # reshape input data into 2D tensor
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    strides = jt.strides_from_shape(x_2d.shape)
    # specify output data shape
    out_shape = [
        jax.ShapeDtypeStruct(shape=x_2d.shape, dtype=x_2d.dtype),
        jax.ShapeDtypeStruct(shape=(M,), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(M,), dtype=jnp.float32)
    ]
    grid = (M,)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # enqueue kernel
    y, mean, rstd = jt.triton_call(
        x_2d, w, b,
        kernel=layer_norm_kernel,
        out_shape=out_shape,
        grid=grid,
        stride=strides[0],
        N=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def test_layer_norm(M, N, dtype, eps=1e-5):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_b, key_dy = jax.random.split(key, 4)

    x = -2.3 + 0.5 * jax.random.normal(key_x, shape=x_shape, dtype=dtype)
    weight = jax.random.uniform(key_w, shape=w_shape, dtype=dtype)
    bias = jax.random.uniform(key_b, shape=w_shape, dtype=dtype)

    # Less than 64KB per feature: enqueue fused kernel
    element_size = 4 if x.dtype == jnp.float32 else 2
    MAX_FUSED_SIZE = 65536 // element_size
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # forward pass
    jit_layer_norm = jax.jit(layer_norm, static_argnums=(3, 4))
    y = jit_layer_norm(x, weight, bias, eps, BLOCK_SIZE).block_until_ready()
    print(y)


def main(unused_argv):
    test_layer_norm(1151, 8192, jnp.float32)


if __name__ == "__main__":
    from absl import app
    app.run(main)
