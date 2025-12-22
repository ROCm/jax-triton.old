# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
from typing import Optional, Dict
import triton
import triton.language as tl

import jax
from jax import random
import jax.numpy as jnp
import jax_triton as jt

from _triton_kernels.mha_fused_bwd_kernel import (
    _bwd_preprocess,
    _bwd_kernel_dkdvdq_causal,
    _bwd_kernel_dkdvdq_noncausal,
    _get_config,
)


# `jax_triton.triton_call`` does not accept `None`` inputs; 
# Replace `None`` input with a dummy tensor instead.
def safe_tensor(x, dtype=jnp.int32):
    if x is None:
        return jnp.zeros((1,), dtype=dtype)
    return x


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "causal",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "dropout_p",
        "philox_seed",
        "philox_offset",
        "USE_INT64_STRIDES",
    ],
)
def _flash_attn_fused_backward(
    # Input tensors
    do: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    o: jnp.ndarray,
    softmax_lse: jnp.ndarray,
    # Output tensors
    dq: jnp.ndarray,
    dk: jnp.ndarray,
    dv: jnp.ndarray,
    dbias: jnp.ndarray,
    # Configurations
    sm_scale: float,
    causal: bool,
    cu_seqlens_q: Optional[jnp.ndarray],
    cu_seqlens_k: Optional[jnp.ndarray],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    USE_INT64_STRIDES: Optional[bool] = False,
    config: Optional[Dict[str, any]] = None,
):
    if dbias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")
    if q.shape[-1] == k.shape[-1] and k.shape[-1] > v.shape[-1]:
        raise ValueError(
            "'Fused' backward doesn't support Positional Encoding (PE). Please use 'one kernel' backward implementation for PE."
        )

    IS_VARLEN = True if cu_seqlens_q is not None else False

    q_strides_in = jt.strides_from_shape(q.shape)
    k_strides_in = jt.strides_from_shape(k.shape)
    v_strides_in = jt.strides_from_shape(v.shape)
    o_strides_in = jt.strides_from_shape(o.shape)
    dq_strides_in = jt.strides_from_shape(dq.shape)
    dk_strides_in = jt.strides_from_shape(dk.shape)
    do_strides_in = jt.strides_from_shape(do.shape)

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
        q_strides = (0, q_strides_in[1], q_strides_in[0], q_strides_in[2])
        k_strides = (0, k_strides_in[1], k_strides_in[0], k_strides_in[2])
        v_strides = (0, v_strides_in[1], v_strides_in[0], v_strides_in[2])
        o_strides = (0, o_strides_in[1], o_strides_in[0], o_strides_in[2])
        dq_strides = (0, dq_strides_in[1], dq_strides_in[0], dq_strides_in[2])
        dk_strides = (0, dk_strides_in[1], dk_strides_in[0], dk_strides_in[2])
        do_strides = (0, do_strides_in[1], do_strides_in[0], do_strides_in[2])
    else:
        # Layout for q,k,v is bshd ie [batch, seq_len, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = q.shape
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = (q_strides_in[0], q_strides_in[2], q_strides_in[1], q_strides_in[3])
        k_strides = (k_strides_in[0], k_strides_in[2], k_strides_in[1], k_strides_in[3])
        v_strides = (v_strides_in[0], v_strides_in[2], v_strides_in[1], v_strides_in[3])
        o_strides = (o_strides_in[0], o_strides_in[2], o_strides_in[1], o_strides_in[3])
        dq_strides = (dq_strides_in[0], dq_strides_in[2], dq_strides_in[1], dq_strides_in[3])
        dk_strides = (dk_strides_in[0], dk_strides_in[2], dk_strides_in[1], dk_strides_in[3])
        do_strides = (do_strides_in[0], do_strides_in[2], do_strides_in[1], do_strides_in[3])

    # BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    # padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_D_MODEL_POW2 = max(BLOCK_D_MODEL_POW2, 16)

    # init delta
    delta = jnp.zeros_like(softmax_lse)
    delta_strides_in = jt.strides_from_shape(delta.shape)
    if IS_VARLEN:
        # [total_tokens, num_q_heads, seqlen_q]
        delta_strides = (0, delta_strides_in[1], delta_strides_in[0])
    else:
        # [batch, num_q_heads, seqlen_q]
        delta_strides = delta_strides_in

    # Configs
    if config is None:
        config = _get_config()

    # preprocess
    # compute D(delta) = rowsum(dO*O). Note, multiplication is element-wise.
    pre_grid = (
        triton.cdiv(max_seqlen_q, config["preprocess_kernel"]["PRE_BLOCK"]),
        batch,
        num_q_heads,
    )
    out_shape = jax.ShapeDtypeStruct(shape=delta.shape, dtype=delta.dtype)

    metaparams_pre = dict(
        BLOCK_M=config["preprocess_kernel"]["PRE_BLOCK"],
        BLOCK_D_MODEL=head_sz,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        IS_VARLEN=IS_VARLEN,
    )

    delta = jt.triton_call(
        o,
        do,
        *o_strides,
        *delta_strides,
        safe_tensor(cu_seqlens_q),
        max_seqlen_q,
        kernel=_bwd_preprocess,
        grid=pre_grid,
        out_shape=out_shape,
        **metaparams_pre
    )

    # dropout_mask
    use_dropout = dropout_p > 0.0
    if use_dropout:
        dropout_mask = jnp.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            dtype=jnp.float32,
        )
        dropout_strides = jt.strides_from_shape(dropout_mask.shape)
    else:
        dropout_mask = None
        dropout_strides = (0, 0, 0, 0)

    # Fuses dk,dv and dq computations into one kernel using atomics
    if BLOCK_D_MODEL_POW2 > 160 or q.dtype == jnp.float32:
        config_dkdvdq = config["dkdvdq_kernel_N64"]
    else:
        config_dkdvdq = config["dkdvdq_kernel_N128"]

    num_k_pids = (max_seqlen_k + config_dkdvdq["BLOCK_N"] - 1) // config_dkdvdq["BLOCK_N"]

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
        NUM_XCD=8,
    )

    if causal:
        kernel = _bwd_kernel_dkdvdq_causal
        grid_dkdvdq = (batch * num_q_heads * num_k_pids,)
    else:
        # in non causal inner loop over grouped q heads
        kernel = _bwd_kernel_dkdvdq_noncausal
        grid_dkdvdq = (batch * num_k_heads * num_k_pids,)

    out_shape = [
        jax.ShapeDtypeStruct(shape=dq.shape, dtype=dq.dtype), 
        jax.ShapeDtypeStruct(shape=dk.shape, dtype=dk.dtype),
        jax.ShapeDtypeStruct(shape=dv.shape, dtype=dv.dtype),       
    ]

    dq, dk, dv = jt.triton_call(
        # inputs
        q, k, v,
        do,
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
        sm_scale,
        safe_tensor(cu_seqlens_q),
        safe_tensor(cu_seqlens_k),
        max_seqlen_q,
        max_seqlen_k,
        safe_tensor(dropout_mask),
        dropout_p,
        philox_seed,
        philox_offset,
        kernel=kernel,
        out_shape=out_shape,
        grid=grid_dkdvdq,
        # static parameters
        **metaparams,
        **config_dkdvdq
    )

    return dq, dk, dv
