# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Dict
import triton  # type: ignore
import triton.language as tl  # type: ignore

import jax
import jax.numpy as jnp
import jax_triton as jt

from _triton_kernels.mha_onekernel_bwd_kernel import (
    _bwd_preprocess,
    _bwd_kernel_causal,
    _bwd_kernel_noncausal,
    _get_config,
)

from common import (
    generate_mha_input,
    jax_attn_fwd_reference,
    jax_attn_bwd_reference,
)


def safe_tensor(x, dtype=jnp.int32):
    if x is None:
        return jnp.zeros((1,), dtype=dtype)
    return x


def _flash_attn_onekernel_backward(
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
    dropout_p: float,
    alibi_slopes: Optional[jnp.ndarray],
    cu_seqlens_q: Optional[jnp.ndarray],
    cu_seqlens_k: Optional[jnp.ndarray],
    max_seqlens_q: int,
    max_seqlens_k: int,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    USE_INT64_STRIDES: Optional[bool] = False,
    config: Optional[Dict[str, any]] = None,
):
    if dbias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")

    use_alibi, (stride_az, stride_ah) = (
        (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))
    )

    IS_VARLEN = True if cu_seqlens_q is not None else False

    q_strides_in = jt.strides_from_shape(q.shape)
    k_strides_in = jt.strides_from_shape(k.shape)
    v_strides_in = jt.strides_from_shape(v.shape)
    o_strides_in = jt.strides_from_shape(o.shape)
    dq_strides_in = jt.strides_from_shape(dq.shape)
    dk_strides_in = jt.strides_from_shape(dk.shape)
    dv_strides_in = jt.strides_from_shape(dv.shape)
    do_strides_in = jt.strides_from_shape(do.shape)

    # get strides and shape
    if IS_VARLEN:
        # Layout is thd.
        # q and k are [total_tokens, num_head, head_dim_qk].
        # v is [total_tokens, num_head, head_dim_v].
        batch, seqlen_q, num_q_heads = (
            len(cu_seqlens_q) - 1,
            max_seqlens_q,
            q.shape[1],
        )
        _, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = (0, q_strides_in[1], q_strides_in[0], q_strides_in[2])
        k_strides = (0, k_strides_in[1], k_strides_in[0], k_strides_in[2])
        v_strides = (0, v_strides_in[1], v_strides_in[0], v_strides_in[2])
        o_strides = (0, o_strides_in[1], o_strides_in[0], o_strides_in[2])
        dq_strides = (0, dq_strides_in[1], dq_strides_in[0], dq_strides_in[2])
        dk_strides = (0, dk_strides_in[1], dk_strides_in[0], dk_strides_in[2])
        dv_strides = (0, dv_strides_in[1], dv_strides_in[0], dv_strides_in[2])
        do_strides = (0, do_strides_in[1], do_strides_in[0], do_strides_in[2])
    else:
        # Layout is bshd.
        # q and k are [batch, seq_len, num_head, head_dim_qk].
        # v is [batch, seq_len, num_head, head_dim_v]
        batch, seqlen_q, num_q_heads = q.shape[:-1]
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = (q_strides_in[0], q_strides_in[2], q_strides_in[1], q_strides_in[3])
        k_strides = (k_strides_in[0], k_strides_in[2], k_strides_in[1], k_strides_in[3])
        v_strides = (v_strides_in[0], v_strides_in[2], v_strides_in[1], v_strides_in[3])
        o_strides = (o_strides_in[0], o_strides_in[2], o_strides_in[1], o_strides_in[3])
        dq_strides = (dq_strides_in[0], dq_strides_in[2], dq_strides_in[1], dq_strides_in[3])
        dk_strides = (dk_strides_in[0], dk_strides_in[2], dk_strides_in[1], dk_strides_in[3])
        dv_strides = (dv_strides_in[0], dv_strides_in[2], dv_strides_in[1], dv_strides_in[3])
        do_strides = (do_strides_in[0], do_strides_in[2], do_strides_in[1], do_strides_in[3])

    qk_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]
    pe_head_dim = qk_head_dim - v_head_dim
    # BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    # padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = max(triton.next_power_of_2(v_head_dim), 16)
    BLOCK_D_MODEL_PE_POW2 = (
        0 if pe_head_dim == 0 else max(triton.next_power_of_2(pe_head_dim), 16)
    )
    assert (pe_head_dim == 0 and BLOCK_D_MODEL_PE_POW2 == 0) or (
        v_head_dim == BLOCK_D_MODEL_POW2 and pe_head_dim == BLOCK_D_MODEL_PE_POW2
    ), "Positional encoding support requires NOPE and PE head sizes to be unpadded powers of 2."

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
        triton.cdiv(max_seqlens_q, config["preprocess_kernel"]["PRE_BLOCK"]),
        batch,
        num_q_heads,
    )
    out_shape = jax.ShapeDtypeStruct(shape=delta.shape, dtype=delta.dtype)

    metaparams_pre = dict(
        BLOCK_M=config["preprocess_kernel"]["PRE_BLOCK"],
        BLOCK_D_MODEL=v_head_dim,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        IS_VARLEN=IS_VARLEN,
    )

    delta = jt.triton_call(
        o,
        do,
        delta,
        *o_strides,
        *delta_strides,
        safe_tensor(cu_seqlens_q),
        max_seqlens_q,
        kernel=_bwd_preprocess,
        grid=pre_grid,
        out_shape=out_shape,
        **metaparams_pre
    )

    # dropout_mask
    use_dropout = dropout_p > 0.0
    if use_dropout:
        dropout_mask = jnp.zeros(
            (batch, num_q_heads, max_seqlens_q, max_seqlens_k),
            dtype=jnp.float32,
        )
        dropout_strides = jt.strides_from_shape(dropout_mask.shape)
    else:
        dropout_mask = None
        dropout_strides = (0, 0, 0, 0)

    seqlen = max(max_seqlens_q, max_seqlens_k)

    # "onekernel_pe" is for Positional Encoding (PE) causal case, it's going to be
    # used if present. Otherwise, fallback to default "onekernel" config.
    config_onekernel = (
        config["onekernel_pe"]
        if (pe_head_dim > 0 and causal and "onekernel_pe" in config)
        else config["onekernel"]
    )

    grid = (
        num_k_heads,
        triton.cdiv(seqlen, config_onekernel["BLOCK_N1"]),
        batch,
    )
    out_shape = [
        jax.ShapeDtypeStruct(shape=dq.shape, dtype=dq.dtype),  
        jax.ShapeDtypeStruct(shape=dk.shape, dtype=dk.dtype),
        jax.ShapeDtypeStruct(shape=dv.shape, dtype=dv.dtype),      
    ]

    metaparams = dict(
        HEAD_DIM=v_head_dim,
        ACTUAL_HEAD_DIM=BLOCK_D_MODEL_POW2,
        PE_HEAD_DIM=pe_head_dim,
        ENABLE_DROPOUT=use_dropout,
        IS_VARLEN=IS_VARLEN,
        USE_ALIBI=use_alibi,
        USE_EXP2=True,
        DEBUG_TRITON=False,
        DEBUG_TRITON_DETAIL=False,
        USE_INT64_STRIDES=USE_INT64_STRIDES,
    )

    if causal:
        kernel = _bwd_kernel_causal
    else:
        kernel = _bwd_kernel_noncausal

    dq, dk, dv = jt.triton_call(
        # Input tensors
        q, k, v,
        do,
        softmax_lse, 
        delta,
        # Output tensors
        dq, dk, dv,
        # Strides
        *q_strides,
        *k_strides,
        *v_strides,
        *dq_strides,
        *dk_strides,
        *dv_strides,
        *delta_strides,
        *do_strides,
        *dropout_strides,
        stride_az,
        stride_ah,
        # Configurations
        sm_scale,
        num_q_heads,
        num_k_heads,
        safe_tensor(cu_seqlens_q),
        safe_tensor(cu_seqlens_k),
        max_seqlens_q,
        max_seqlens_k,
        dropout_mask,
        dropout_p,
        philox_seed,
        philox_offset,
        safe_tensor(alibi_slopes),
        kernel=kernel,
        out_shape=out_shape,
        grid=grid,
        **metaparams,
        **config_onekernel
    )

    return dq, dk, dv


def flash_attn_onekernel_backward(do, q, k, v, input_meta):
    # save fwd outputs for bwd
    o, softmax_lse = jax_attn_fwd_reference(q, k, v)

    # Transpose to (B, S, H, D)
    do = jnp.transpose(do, (0, 2, 1, 3))
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    o = jnp.transpose(o, (0, 2, 1, 3))
    softmax_lse = jnp.transpose(softmax_lse, (0, 2, 1))

    # initialize output gradients
    dq = jnp.zeros_like(q)
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    # unpack meta-parameters
    input_metaparams = dict(
        sm_scale=input_meta.sm_scale,
        causal=input_meta.causal,
        dropout_p=input_meta.dropout_p,
        alibi_slopes=input_meta.alibi_slopes,
        cu_seqlens_q=input_meta.cu_seqlens_q,
        cu_seqlens_k=input_meta.cu_seqlens_k,
        max_seqlens_q=input_meta.max_seqlens_q,
        max_seqlens_k=input_meta.max_seqlens_k,
    )

    dq, dk, dv = _flash_attn_onekernel_backward(
        # Input tensors
        do=do,
        q=q,
        k=k,
        v=v,
        o=o,
        softmax_lse=softmax_lse,
        # Output tensors
        dq=dq,
        dk=dk,
        dv=dv,
        dbias=None,
        **input_metaparams
    )
    dq.block_until_ready()
    dk.block_until_ready()
    dv.block_until_ready()

    # Transpose to (B, H, S, D)
    dq = jnp.transpose(dq, (0, 2, 1, 3))
    dk = jnp.transpose(dk, (0, 2, 1, 3))
    dv = jnp.transpose(dv, (0, 2, 1, 3))

    return dq, dk, dv


BATCH_SIZE: int = 1
SEQ_LEN: int = 128
NUM_HEADS: int = 16
HEAD_SIZE: int = 32
MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)
dtype = jnp.float16


def main(unused_argv):
    # generate input data, causal = True
    q, k, v, input_meta = generate_mha_input(MHA_SHAPE, MHA_DTYPE)
    input_meta.need_causal()

    # random upstream gradient
    key = jax.random.PRNGKey(RNG_SEED)
    do = jax.random.normal(key, q.shape, dtype=q.dtype)  # only when D_v == D_q

    # Triton results
    dq, dk, dv = flash_attn_onekernel_backward(do, q, k, v, input_meta)

    # Reference results
    dq_ref, dk_ref, dv_ref = jax_attn_bwd_reference(q, k, v, do)

    # numeric check
    atol = 1e-2 if MHA_DTYPE in (jnp.float16, jnp.bfloat16) else 1e-3
    max_diff = jnp.max(jnp.abs(dq - dq_ref))
        
    assert jnp.allclose(dq, dq_ref, atol=atol), \
        f"Max diff ({max_diff}) exceeds tolerance (atol={atol})"


if __name__ == "__main__":
    from absl import app
    app.run(main)
