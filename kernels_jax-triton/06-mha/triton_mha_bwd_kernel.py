# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Dict

import jax
from jax import random
import jax.numpy as jnp
import jax_triton as jt

import triton
import triton.language as tl

# from aiter.ops.triton.utils.types import _is_fp8
# from aiter.ops.triton.utils.logger import AiterTritonLogger
from _triton_kernels.mha_fused_bwd import (
    _bwd_preprocess,
    _bwd_kernel_dkdvdq_causal,
    _bwd_kernel_dkdvdq_noncausal,
    _get_config,
)


_LOGGER = AiterTritonLogger()


def flash_attn_fused_backward(
    do: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    o: jnp.ndarray,
    softmax_lse: jnp.ndarray,
    dq: jnp.ndarray,
    dk: jnp.ndarray,
    dv: jnp.ndarray,
    dbias: jnp.ndarray,
    sm_scale: float,
    alibi_slopes: Optional[jnp.Tensor],
    causal: bool,
    cu_seqlens_q: Optional[jnp.Tensor],
    cu_seqlens_k: Optional[jnp.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    descale_q: Optional[jnp.Tensor] = None,
    descale_k: Optional[jnp.Tensor] = None,
    descale_v: Optional[jnp.Tensor] = None,
    descale_do: Optional[jnp.Tensor] = None,
    USE_INT64_STRIDES: Optional[bool] = False,
    config: Optional[Dict[str, any]] = None,
):
    _LOGGER.info(
        f"FLASH_ATTN_FUSED_BKWD: do={tuple(do.shape)} q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)} "
        + f"dq={tuple(dq.shape)}  dk={tuple(dk.shape)}  dv={tuple(dv.shape)}"
    )
    if dbias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")

    stride_descale_q_z = stride_descale_k_z = stride_descale_v_z = (
        stride_descale_do_z
    ) = None
    descale_strides = (
        stride_descale_q_z,
        stride_descale_k_z,
        stride_descale_v_z,
        stride_descale_do_z,
    )

    IS_VARLEN = True if cu_seqlens_q is not None else False

    # get strides and shape
    if IS_VARLEN:
        # Layout for q,k,v is thd ie [total tokens, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
            q.shape[2],
        )
        _, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
        dq_strides = (0, dq.stride(1), dq.stride(0), dq.stride(2))
        dk_strides = (0, dk.stride(1), dk.stride(0), dk.stride(2))
        do_strides = (0, do.stride(1), do.stride(0), do.stride(2))
    else:
        # Layout for q,k,v is bshd ie [batch, seq_len, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = q.shape
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
        dq_strides = (dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3))
        dk_strides = (dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3))
        do_strides = (do.stride(0), do.stride(2), do.stride(1), do.stride(3))

    # BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    # padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_D_MODEL_POW2 = max(BLOCK_D_MODEL_POW2, 16)

    # init delta
    delta = jnp.zeros_like(softmax_lse)
    if IS_VARLEN:
        # [total_tokens, num_q_heads, seqlen_q]
        delta_strides = (0, delta.stride(1), delta.stride(0))
    else:
        # [batch, num_q_heads, seqlen_q]
        delta_strides = delta.stride()

    # preprocess
    # compute D(delta) = rowsum(dO*O). Note, multiplication is element-wise.
    if config is None:
        config = _get_config()

    pre_grid = (
        triton.cdiv(max_seqlen_q, config["preprocess_kernel"]["PRE_BLOCK"]),
        batch,
        num_q_heads,
    )

    metaparams = dict(
        BLOCK_M=config["preprocess_kernel"]["PRE_BLOCK"],
        BLOCK_D_MODEL=head_sz,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        # num_warps=4,
        # num_stages=1,
    )

    delta = jt.triton_call(
        o,
        do,
        delta,
        *o_strides,
        *delta_strides,
        descale_strides[3],
        cu_seqlens_q,
        max_seqlen_q,
        descale_do,
        kernel=_bwd_preprocess,
        out_shape=out_shape,
        grid=pre_grid,
        **metaparams
    )

    # dropout_mask
    use_dropout = dropout_p > 0.0
    if use_dropout:
        dropout_mask = jnp.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            dtype=jnp.float32,
        )
        dropout_strides = dropout_mask.stride()
    else:
        dropout_mask = None
        dropout_strides = (0, 0, 0, 0)

    # Fuses dk,dv and dq computations into one kernel using atomics
    if BLOCK_D_MODEL_POW2 > 160 or q.dtype == jnp.float32:
        config_dkdvdq = config["dkdvdq_kernel_N64"]
    else:
        config_dkdvdq = config["dkdvdq_kernel_N128"]

    num_k_pids = (max_seqlen_k + config_dkdvdq["BLOCK_N"] - 1) // config_dkdvdq[
        "BLOCK_N"
    ]
    # NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    metaparams = dict(
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        BATCH=batch,
        NUM_K_PIDS=num_k_pids,
        BLOCK_D_MODEL=head_sz,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        ENABLE_DROPOUT=use_dropout,
        IS_VARLEN=IS_VARLEN,
        USE_INT64_STRIDES=USE_INT64_STRIDES,
        NUM_XCD=get_num_xcds(),
        # num_warps=4,
        # num_stages=1,
    )

    if causal:
        grid_dkdvdq = (batch * num_q_heads * num_k_pids,)

        _, _, delta = jt.triton_call(
            q,
            k,
            v,
            sm_scale,
            do,
            dk,
            dv,
            dq,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dk_strides,
            *dq_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            kernel=_bwd_kernel_dkdvdq_causal,
            out_shape=out_shape,
            grid=grid_dkdvdq,
            **metaparams,
            **config_dkdvdq
        )
    else:
        # in non causal inner loop over grouped q heads
        grid_dkdvdq = (batch * num_k_heads * num_k_pids,)

        _, _, delta = jt.triton_call(
            q,
            k,
            v,
            sm_scale,
            do,
            dk,
            dv,
            dq,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dk_strides,
            *dq_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            kernel=_bwd_kernel_dkdvdq_causal,
            out_shape=out_shape,
            grid=grid_dkdvdq,
            **metaparams,
            **config_dkdvdq
        )

    return delta
