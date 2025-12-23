# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
from typing import Optional, Tuple
import triton
import triton.language as tl

import jax
from jax import random
import jax.numpy as jnp
import jax_triton as jt

from jax_triton.ops.utils.device_info import get_num_xcds
from jax_triton.ops._kernels.mha_fwd_kernel import _attn_fwd, _get_config

_USE_INT64_STRIDES = True


def mha_set_use_int64_strides(value: bool):
    """Use 64-bit integer strides to prevent integer overflows with very large tensors."""
    global _USE_INT64_STRIDES
    _USE_INT64_STRIDES = value


# `jax_triton.triton_call`` does not accept `None`` inputs; 
# Replace `None`` input with a dummy tensor instead.
def safe_tensor(x, dtype=jnp.int32):
    if x is None:
        return jnp.zeros((1,), dtype=dtype)
    return x


@functools.partial(
    jax.jit,
    static_argnames=[
        "softmax_scale",
        "dropout_p",
        "causal",
        "window_size_left",
        "window_size_right",
        "return_lse",
        "return_softmax",
        "max_seqlen_q",
        "max_seqlen_k",
        # "config",  # TODO: `dict`` cannot be used as static arg.
    ],
)
def _flash_attn_forward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: Optional[jnp.ndarray],
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[jnp.ndarray],
    return_lse: bool,
    return_softmax: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_q: Optional[jnp.ndarray] = None,
    cu_seqlens_k: Optional[jnp.ndarray] = None,
    config: Optional[dict[str, any]] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if bias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")
    if window_size_left != -1 or window_size_right != -1:
        raise ValueError("Sliding Window is not supported yet in the Triton Backend")

    enable_alibi = True if alibi_slopes is not None else False
    is_varlen = True if cu_seqlens_q is not None else False

    o = jnp.zeros((q.shape[:-1] + v.shape[-1:]), dtype=q.dtype)

    q_strides = jt.strides_from_shape(q.shape)
    k_strides = jt.strides_from_shape(k.shape)
    v_strides = jt.strides_from_shape(v.shape)
    o_strides = jt.strides_from_shape(o.shape)
    if alibi_slopes is not None:
        alibi_slopes_strides = jt.strides_from_shape(alibi_slopes.shape)

    if is_varlen:
        # Layout is thd.
        # q and k are [total_tokens, num_head, head_dim_qk].
        # v is [total_tokens, num_head, head_dim_v].
        batch, seqlen_q, num_q_heads = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
        )
        num_k_heads = k.shape[1]
        q_strides = (0, q_strides[1], q_strides[0], q_strides[2])
        k_strides = (0, k_strides[1], k_strides[0], k_strides[2])
        v_strides = (0, v_strides[1], v_strides[0], v_strides[2])
        o_strides = (0, o_strides[1], o_strides[0], o_strides[2])
    else:
        # Layout is bshd.
        # q and k are [batch, seq_len, num_head, head_dim_qk].
        # v is [batch, seq_len, num_head, head_dim_v].
        batch, seqlen_q, num_q_heads = q.shape[:-1]
        num_k_heads = k.shape[2]
        q_strides = (q_strides[0], q_strides[2], q_strides[1], q_strides[3])
        k_strides = (k_strides[0], k_strides[2], k_strides[1], k_strides[3])
        v_strides = (v_strides[0], v_strides[2], v_strides[1], v_strides[3])
        o_strides = (o_strides[0], o_strides[2], o_strides[1], o_strides[3])

    qk_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]
    pe_head_dim = qk_head_dim - v_head_dim
    # padding for head_dim. Power of 2 or 16
    BLOCK_DMODEL_POW2 = max(triton.next_power_of_2(v_head_dim), 16)
    BLOCK_DMODEL_PE_POW2 = (
        0 if pe_head_dim == 0 else max(triton.next_power_of_2(pe_head_dim), 16)
    )
    assert (pe_head_dim == 0 and BLOCK_DMODEL_PE_POW2 == 0) or (
        v_head_dim == BLOCK_DMODEL_POW2 and pe_head_dim == BLOCK_DMODEL_PE_POW2
    ), "Positional encoding support requires NOPE and PE head sizes to be unpadded powers of 2."

    # softmax_lse [batch, num_q_heads, seqlen_q]
    if is_varlen:
        softmax_lse = jnp.zeros(
            (q.shape[0], num_q_heads), dtype=jnp.float32
        )
        lse_strides = jt.strides_from_shape(softmax_lse.shape)
        stride_lse_z, stride_lse_h, stride_lse_m = (
            0,
            lse_strides[1],
            lse_strides[0],
        )
    else:
        softmax_lse = jnp.zeros(
            (batch, num_q_heads, max_seqlen_q), dtype=jnp.float32
        )
        lse_strides = jt.strides_from_shape(softmax_lse.shape)
        stride_lse_z, stride_lse_h, stride_lse_m = lse_strides

    # exp_scores [batch, num_q_heads, seqlen_q, seqlen_k]
    enable_dropout = dropout_p > 0.0
    if enable_dropout:
        key = random.PRNGKey(42)
        seed_key, offset_key = random.split(key, 2)
        # Pass in an int, not Tensor
        philox_seed = random.randint(seed_key, (), 0, 0xFFFFFF).item()
        philox_offset = random.randint(offset_key, (), 0, 0xFFFFFF).item()
    else:
        philox_seed = 0
        philox_offset = 0

    if return_softmax or enable_dropout:
        enable_sdmask = True
        s_dmask = jnp.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            dtype=jnp.float32,
        )
        s_dmask_strides = jt.strides_from_shape(s_dmask.shape)
        dropout_mask = jnp.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            dtype=jnp.float32,
        )
    else:
        enable_sdmask = False
        s_dmask = None
        dropout_mask = None

    if config is None:
        config = _get_config(enable_dropout, q.dtype, has_pe=pe_head_dim > 0)

    """
    # Tuned for MI300x
    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "waves_per_eu": 2,
        "num_warps": 4,
        "num_ctas": 1,
        "num_stages": 1,
    }
    # Dropout significantly increases VGPR usage so use small tiles
    if enable_dropout or q.dtype == jnp.float32:
        config = {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "waves_per_eu": 1,
            "num_warps": 2,
            "num_ctas": 1,
            "num_stages": 1,
        }
    """
    out_shape = [
        jax.ShapeDtypeStruct(shape=o.shape, dtype=o.dtype), 
        jax.ShapeDtypeStruct(shape=softmax_lse.shape, dtype=softmax_lse.dtype),       
    ]
    grid = lambda META: (  # noqa: E731
        batch * num_q_heads * triton.cdiv(seqlen_q, META["BLOCK_M"]),
    )

    o, softmax_lse = jt.triton_call(
        # inputs
        q,
        k,
        v,
        # optional
        safe_tensor(alibi_slopes),
        safe_tensor(s_dmask),
        safe_tensor(dropout_mask),
        # strides
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        alibi_slopes_strides[0] if alibi_slopes is not None else 0,
        alibi_slopes_strides[1] if alibi_slopes is not None else 0,
        s_dmask_strides[0] if s_dmask is not None else 0,
        s_dmask_strides[1] if s_dmask is not None else 0,
        s_dmask_strides[2] if s_dmask is not None else 0,
        s_dmask_strides[3] if s_dmask is not None else 0,
        stride_lse_z if softmax_lse is not None else 0,
        stride_lse_h if softmax_lse is not None else 0,
        stride_lse_m if softmax_lse is not None else 0,
        softmax_scale,
        safe_tensor(cu_seqlens_q),
        safe_tensor(cu_seqlens_k),
        dropout_p,
        philox_seed,
        philox_offset,
        max_seqlen_q,
        max_seqlen_k,
        # static parameters
        IS_CAUSAL=causal,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        BLOCK_DMODEL=v_head_dim,
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        BLOCK_DMODEL_PE=pe_head_dim,
        # flags for optional tensors
        ENABLE_ALIBI=enable_alibi,
        ENABLE_SDMASK=enable_sdmask,
        ENABLE_DROPOUT=enable_dropout,
        RETURN_SCORES=return_softmax,
        VARLEN=is_varlen,
        BATCH=batch,
        NUM_XCD=get_num_xcds(),
        USE_INT64_STRIDES=_USE_INT64_STRIDES,
        kernel=_attn_fwd,
        out_shape=out_shape,
        grid=grid,
        **config,
    )

    return o, softmax_lse, s_dmask, philox_seed, philox_offset
