# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Dict

import jax
from jax import random
import jax.numpy as jnp
import jax_triton as jt

import triton
import triton.language as tl

from _triton_kernels.mha_fused_bwd_kernel import (
    _bwd_preprocess,
    _bwd_kernel_dkdvdq_causal,
    _bwd_kernel_dkdvdq_noncausal,
)

# from common import (
#     MhaMetaData,
#     mha_input_helper,
#     get_shape_from_layout,
#     get_strides_from_layout,
# )


_LOGGER = AiterTritonLogger()


def flash_attn_fused_backward(
    # inputs
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    o: jnp.ndarray,
    softmax_lse: jnp.ndarray,
    do: jnp.ndarray,
    sm_scale: float,
    # outputs
    dq: jnp.ndarray,
    dk: jnp.ndarray,
    dv: jnp.ndarray,
    # others
    causal: bool,
    alibi_slopes: Optional[jnp.Tensor],
    cu_seqlens_q: Optional[jnp.Tensor],
    cu_seqlens_k: Optional[jnp.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    USE_INT64_STRIDES: Optional[bool] = False,
    # config: Optional[Dict[str, any]] = None,
):

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

    # if config is None:
    #     config = _get_config()
    config = {
        "preprocess_kernel": {
        "PRE_BLOCK": 128
        }, 
        "dkdvdq_kernel_N64" : {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "num_warps": 8,
            "num_stages": 1,
            "waves_per_eu": 2,
            "BLK_SLICE_FACTOR": 1,
            "matrix_instr_nonkdim": 16
        }, 
        "dkdvdq_kernel_N128" : {
            "BLOCK_M": 16,
            "BLOCK_N": 128,
            "num_warps": 8,
            "num_stages": 1,
            "waves_per_eu": 2,
            "BLK_SLICE_FACTOR": 1,
            "matrix_instr_nonkdim": 16
        },
    }

    out_shape = jax.ShapeDtypeStruct(shape=delta.shape, dtype=delta.dtype)
    pre_grid = (
        # triton.cdiv(max_seqlen_q, config["preprocess_kernel"]["PRE_BLOCK"]),
        triton.cdiv(max_seqlen_q, PRE_BLOCK),
        batch,
        num_q_heads,
    )

    metaparams_pre = dict(
        BLOCK_M=config["preprocess_kernel"]["PRE_BLOCK"],
        BLOCK_D_MODEL=head_sz,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        IS_VARLEN=IS_VARLEN,
        # num_warps=4,
        # num_stages=1,
    )

    delta = jt.triton_call(
        o,
        do,
        delta,
        *o_strides,
        *delta_strides,
        cu_seqlens_q,
        max_seqlen_q,
        kernel=_bwd_preprocess,
        out_shape=out_shape,
        grid=pre_grid,
        **metaparams_pre
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

    num_k_pids = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N
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
        NUM_XCD=8,
        # num_warps=4,
        # num_stages=1,
    )

    if causal:
        kernel = _bwd_kernel_dkdvdq_causal
        grid_dkdvdq = (batch * num_q_heads * num_k_pids,)
    else:
        # in non causal inner loop over grouped q heads
        kernel = _bwd_kernel_dkdvdq_noncausal
        grid_dkdvdq = (batch * num_k_heads * num_k_pids,)

    out_shape = (
        jax.ShapeDtypeStruct(shape=dk.shape, dtype=dk.dtype),
        jax.ShapeDtypeStruct(shape=dv.shape, dtype=dv.dtype),
        jax.ShapeDtypeStruct(shape=dq.shape, dtype=dq.dtype),        
    )

    dv, dk = jt.triton_call(
        # inputs
        q, k, v,
        do,
        softmax_lse, delta,
        sm_scale,
        # outputs
        dk, dv, dq,
        # strides
        *q_strides,
        *k_strides,
        *v_strides,
        *dk_strides,
        *dq_strides,
        *delta_strides,
        *do_strides,
        *dropout_strides,
        # others
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_mask,
        dropout_p,
        philox_seed,
        philox_offset,
        kernel=kernel,
        out_shape=out_shape,
        grid=grid_dkdvdq,
        **metaparams,
        **config_dkdvdq
    )

    return dv, dk


# MHA shape
BATCH_SIZE: int = 2
SEQ_LEN: int = 4096
NUM_HEADS: int = 64
HEAD_SIZE: int = 128

MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)

# MHA dtype
MHA_DTYPE = jnp.float32

RNG_SEED = 42


def main(unused_argv):
    # generate random key
    jax.random.PRNGKey(RNG_SEED)
    q_key, k_key, v_key, do_key = jax.random.split(key, 4)

    # fwd inputs
    q = jax.random.normal(q_key, MHA_SHAPE, dtype=MHA_DTYPE)
    k = jax.random.normal(k_key, MHA_SHAPE, dtype=MHA_DTYPE)
    v = jax.random.normal(v_key, MHA_SHAPE, dtype=MHA_DTYPE)

    sm_scale = HEAD_SIZE ** -0.5

    # metaparams
    causal = True
    alibi_slopes = None
    cu_seqlens_q = cu_seqlens_k = None
    max_seqlen_q = max_seqlen_k = SEQ_LEN

    # save fwd outputs for bwd
    o, softmax_lse = mha_fwd_reference(q, k, v, sm_scale=sm_scale, causal=causal)

    do = jax.random.normal(do_key, o.shape, dtype=MHA_DTYPE)

    # bwd outputs
    dq = jnp.zeros_like(q)
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    # call function
    dv, dk = flash_attn_fused_backward(
        # inputs
        q=q,
        k=k,
        v=v,
        o=o,
        softmax_lse=softmax_lse,
        do=do,
        sm_scale=sm_scale,
        # outputs
        dq=dq,
        dk=dk,
        dv=dv,
        # metaparams
        causal=causal,
        alibi_slopes=alibi_slopes,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        # dropout_p=dropout_p,
        # philox_seed=philox_seed,
        # philox_offset=philox_offset,
        # USE_INT64_STRIDES=False,
        # config=config,
    )


if __name__ == "__main__":
    from absl import app
    app.run(main)
