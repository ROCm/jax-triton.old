# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton  # type: ignore
import triton.language as tl  # type: ignore

import jax
from jax import random
import jax.numpy as jnp
import jax_triton as jt

from mha_fwd import _flash_attn_forward
from mha_onekernel_bwd import _flash_attn_onekernel_backward
from common import jax_attn_bwd_reference

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
    q_key, k_key, v_key, do_key = random.split(key, 4)

    q = random.normal(q_key, MHA_SHAPE, dtype=dtype)
    k = random.normal(k_key, MHA_SHAPE, dtype=dtype)
    v = random.normal(v_key, MHA_SHAPE, dtype=dtype)
    bias = None

    do = random.normal(v_key, q.shape, dtype=q.dtype)
    dq = jnp.zeros_like(q)
    dk = jnp.empty_like(k)
    dv = jnp.empty_like(v)
    dbias = jnp.empty_like(bias) if bias is not None else None

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
        return_softmax=False,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )

    # Triton attention bwd
    dq_triton, dk_triton, dv_triton = _flash_attn_onekernel_backward(
        do,
        q, k, v,
        triton_out,
        triton_lse,
        dq, dk, dv,
        dbias,
        softmax_scale,
        alibi_slopes,
        causal,
        None,
        None,
        max_seqlen_q=q.shape[1],
        max_seqlen_k=k.shape[1],
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        USE_INT64_STRIDES=_USE_INT64_STRIDES,
    )

    # Reference attention bwd
    dq, dk, dv = jax_attn_bwd_reference(q, k, v, do)

    # numeric check
    atol = 1e-2 if MHA_DTYPE in (jnp.float16, jnp.bfloat16) else 1e-3
    rtol = 1e-2 if MHA_DTYPE in (jnp.float16, jnp.bfloat16) else 1e-3

    assert jnp.allclose(dq, triton_dq.astype(dq.dtype), atol=atol, rtol=rtol), \
        "Max diff of dq exceeds tolerance"
    assert jnp.allclose(dk, triton_dk.astype(dk.dtype), atol=atol, rtol=rtol), \
        "Max diff of dk exceeds tolerance"
    assert jnp.allclose(dv, triton_dv.astype(dv.dtype), atol=atol, rtol=rtol), \
        "Max diff of dv exceeds tolerance"


if __name__ == "__main__":
    from absl import app
    app.run(main)
