# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import logging

import jax
import jax.numpy as jnp
import numpy as np

from jax_triton.ops.mha import (
    flash_attn_func,
    mha_set_use_fused_bwd_kernel
)
from jax_triton.test_mha_common import jax_attn_reference

DEBUG_MODE = False


@pytest.mark.parametrize("BATCH", [1, 4, 57, 128])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (4, 4), (128, 128), (2, 1), (1, 2), (32, 16), (64, 128)],
)
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [8, 32, 128])
# @pytest.mark.parametrize("DROPOUT, CAUSAL", [(0.0, False), (0.0, True), (0.2, False)])  # TODO
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
@pytest.mark.parametrize("FUSED", [False, True])
def test_mha_backward(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    FUSED: bool,
    DROPOUT: float = 0.0,
    dtype=jnp.float16,
):
    if SEQLEN_Q != SEQLEN_K or NUM_Q_HEADS != NUM_K_HEADS:
        pytest.skip("Currently only q, k of the same shape are supported")

    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")
    mha_set_use_fused_bwd_kernel(FUSED)

    # create input data
    key = jax.random.PRNGKey(42)
    q_key, k_key, v_key, do_key = jax.random.split(key, 4)

    q = jax.random.normal(q_key, (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), dtype=dtype)
    k = jax.random.normal(k_key, (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), dtype=dtype)
    v = jax.random.normal(v_key, (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), dtype=dtype)
    do = jax.random.normal(do_key, q.shape, dtype=dtype)

    if DEBUG_MODE:
        print("--------------Triton----------------")
        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"do.shape={do.shape} do={do}")

    # Triton attention function
    def triton_mha_fwd(q, k, v):
        return flash_attn_func(
            q, k, v,
            causal=CAUSAL,
        )
    triton_out, triton_vjp_fn = jax.vjp(triton_mha_fwd, q, k, v)

    # Triton attention bwd
    triton_dq, triton_dk, triton_dv, *_ = triton_vjp_fn(do)
    triton_dq = triton_dq.block_until_ready()
    triton_dk = triton_dk.block_until_ready()
    triton_dv = triton_dv.block_until_ready()

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
        print(f"triton_dq.shape={triton_dq.shape} triton_dq={triton_dq}")
        print(f"triton_dk.shape={triton_dk.shape} triton_dk={triton_dk}")
        print(f"triton_dv.shape={triton_dv.shape} triton_dv={triton_dv}")

    if DEBUG_MODE:
        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"do.shape={do.shape} do={do}")

    # Jax reference attention function
    def mha_fwd(q, k, v):
        return jax_attn_reference(
            q, k, v,
            causal=CAUSAL
        )
    jax_out, vjp_fn = jax.vjp(mha_fwd, q, k, v)

    # Jax reference attention bwd
    jax_dq, jax_dk, jax_dv = vjp_fn(do)
    jax_dq = jax_dq.block_until_ready()
    jax_dk = jax_dk.block_until_ready()
    jax_dv = jax_dv.block_until_ready()

    if DEBUG_MODE:
        print(f"jax_out={jax_out}")
        print(f"jax_dq.shape={jax_dq.shape} jax_dq={jax_dq}")
        print(f"jax_dk.shape={jax_dk.shape} jax_dk={jax_dk}")
        print(f"jax_dv.shape={jax_dv.shape} jax_dv={jax_dv}")

    np.testing.assert_allclose(triton_dq, jax_dq, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(triton_dk, jax_dk, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(triton_dv, jax_dv, atol=1e-2, rtol=1e-2)
