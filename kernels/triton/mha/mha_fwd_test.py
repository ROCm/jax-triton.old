# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import jax
from jax import random
import jax.numpy as jnp

from mha_fwd import _flash_attn_forward
from common import jax_attn_fwd_reference

_USE_INT64_STRIDES = True


BATCH_SIZE: int = 1
SEQ_LEN: int = 128
NUM_HEADS: int = 16
HEAD_SIZE: int = 32
MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)
dtype = jnp.float16


def main(unused_argv):
    # create input data
    key = random.PRNGKey(42)
    q_key, k_key, v_key = random.split(key, 3)

    q = random.normal(q_key, MHA_SHAPE, dtype=dtype)
    k = random.normal(k_key, MHA_SHAPE, dtype=dtype)
    v = random.normal(v_key, MHA_SHAPE, dtype=dtype)
    bias = None

    # configurations
    softmax_scale = q.shape[-1] ** (-0.5)
    alibi_slopes = None
    causal = True
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlen_q = SEQ_LEN
    max_seqlen_k = SEQ_LEN
    dropout_p = 0.0
    if dropout_p > 0.0:
        dropout_mask = sd_mask >= 0
    else:
        dropout_mask = None
    philox_seed = 0
    philox_offset = 0

    # for i in range(100):
    # Triton attention fwd
    triton_out, triton_lse, _, _, _ = _flash_attn_forward(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        bias=bias,
        alibi_slopes=alibi_slopes,
        return_lse=True,
        return_softmax=True,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )

    # Reference attention fwd
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    out, lse = jax_attn_fwd_reference(q, k, v)
    out = out.block_until_ready()
    
    out = jnp.transpose(out, (0, 2, 1, 3))

    # numeric check
    atol = 1e-2 if dtype in (jnp.float16, jnp.bfloat16) else 1e-3
    max_diff = jnp.max(jnp.abs(out - triton_out))

    assert jnp.allclose(out, triton_out.astype(out.dtype), atol=atol), \
        f"Max diff ({max_diff}) of out exceeds tolerance (atol={atol})"
    print("Test passed!")


if __name__ == "__main__":
    from absl import app
    app.run(main)
