# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import logging
import jax
from jax import random
import jax.numpy as jnp
from jax_triton.kernels.triton.mha.mha_fwd import _flash_attn_forward
from jax_triton.kernels.triton.mha.common import jax_attn_fwd_reference

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
@pytest.mark.parametrize(
    "DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ", [(0.2, True, True), (0.0, False, False)]
)
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
def test_mha_forward(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    CAUSAL: bool,
    dtype=jnp.float16,
):
    # create input data
    key = random.PRNGKey(42)
    q_key, k_key, v_key = random.split(key, 3)

    q = random.normal(q_key, (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), dtype=dtype)
    k = random.normal(k_key, (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), dtype=dtype)
    v = random.normal(v_key, (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), dtype=dtype)
    bias = None

    # other parameters
    softmax_scale = q.shape[-1] ** (-0.5)
    alibi_slopes = None
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlen_q = SEQLEN_Q
    max_seqlen_k = SEQLEN_K
    dropout_mask = None

    # Triton attention fwd
    triton_outs = _flash_attn_forward(
        q, k, v,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_p=DROPOUT,
        causal=CAUSAL,
        window_size_left=-1,
        window_size_right=-1,
        alibi_slopes=alibi_slopes,
        return_lse=RETURN_LSE,
        return_softmax=RETURN_SOFTMAX,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )

    if RETURN_LSE:
        lse = triton_outs[1]
        if DEBUG_MODE:
            print(f"lse.shape={lse.shape}, lse={lse}")

    if DROPOUT > 0.0 and RETURN_SOFTMAX:
        if RETURN_LSE:
            sd_mask = triton_outs[2]
        else:
            sd_mask = triton_outs[1]
        dropout_mask = sd_mask >= 0
        if DEBUG_MODE:
            print(f"sd_mask.shape={sd_mask.shape}, sd_mask={sd_mask}")
            print(
                f"dropout_mask.shape={dropout_mask.shape}, dropout_mask={dropout_mask}"
            )

    if RETURN_SOFTMAX or RETURN_LSE:
        triton_out = triton_outs[0]
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    torch_out = attention_ref(
        q, k, v, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL
    )
    torch_out, attention_scores, _ = torch_out
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(
            f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}"
        )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)
