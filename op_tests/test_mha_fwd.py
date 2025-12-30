# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import logging

import jax
import jax.numpy as jnp

from jax_triton.ops.mha import flash_attn_func
from jax_triton.test_mha_common import jax_attn_reference

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
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
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
def test_mha_forward(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float = 0.0,
    CAUSAL: bool,
    RETURN_LSE: bool = False,
    RETURN_SOFTMAX: bool = False,
    dtype=jnp.float16,
):
    # create input data
    key = jax.random.PRNGKey(42)
    q_key, k_key, v_key = jax.random.split(key, 3)

    q = jax.random.normal(q_key, (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), dtype=dtype)
    k = jax.random.normal(k_key, (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), dtype=dtype)
    v = jax.random.normal(v_key, (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), dtype=dtype)

    # Triton attention fwd
    triton_out = flash_attn_func(
        q, k, v,
        dropout_p=DROPOUT,
        causal=CAUSAL,
        return_lse=RETURN_LSE,
        return_softmax=RETURN_SOFTMAX,
    )
    triton_out = triton_out.block_until_ready()
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    # Jax reference attention fwd
    jax_out = jax_attn_reference(
        q, k, v,
        causal=CAUSAL
    )
    jax_out = jax_out.block_until_ready()
    if DEBUG_MODE:
        print(f"jax_out.shape={jax_out.shape}, jax_out={jax_out}")

    jnp.testing.assert_allclose(triton_out, jax_out, atol=1e-2, rtol=1e-2)
