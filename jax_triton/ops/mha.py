# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp

from jax_triton.ops.mha_fwd import _flash_attn_forward
from jax_triton.ops.mha_onekernel_bwd import _flash_attn_onekernel_backward
from jax_triton.ops.mha_fused_bwd import _flash_attn_fused_backward


global _USE_FUSED_BWD_KERNEL
_USE_FUSED_BWD_KERNEL = False


def mha_set_use_fused_bwd_kernel(value: bool):
    global _USE_FUSED_BWD_KERNEL
    _USE_FUSED_BWD_KERNEL = value


_USE_INT64_STRIDES = True


def mha_set_use_int64_strides(value: bool):
    """Use 64-bit integer strides to prevent integer overflows with very large tensors."""
    global _USE_INT64_STRIDES
    _USE_INT64_STRIDES = value


@functools.partial(
    jax.custom_vjp, nondiff_argnums=[3, 4, 5, 6, 9, 10, 11]
)
@functools.partial(
    jax.jit,
    static_argnames=[
        "dropout_p",
        "softmax_scale",
        "causal",
        "window_size",
        "return_lse",
        "return_softmax",
        "config"
    ],
)
def flash_attn_func(
    q, k, v,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    return_lse: bool = False,
    return_softmax: bool = False,
    config: Optional[Dict[str, Any]] = None
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    out, _, _, _, _ =  _flash_attn_forward(
        q, k, v,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size_left=int(window_size[0]),
        window_size_right=int(window_size[1]),
        alibi_slopes=alibi_slopes,
        return_lse=return_lse,
        return_softmax=return_softmax,
        max_seqlen_q=q.shape[1],
        max_seqlen_k=k.shape[1],
        config=config
    )
    return out


def _mha_forward(
    q, k, v,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    return_lse: bool = False,
    return_softmax: bool = False,
    config: Optional[Dict[str, Any]] = None
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    out, lse, s_dmask, philox_seed, philox_offset =  _flash_attn_forward(
        q, k, v,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size_left=int(window_size[0]),
        window_size_right=int(window_size[1]),
        alibi_slopes=alibi_slopes,
        return_lse=return_lse,
        return_softmax=return_softmax,
        max_seqlen_q=q.shape[1],
        max_seqlen_k=k.shape[1],
        config=config
    )
    # Save for backward
    residuals = (
        q, k, v, bias, alibi_slopes,  # input tensors
        out, lse, s_dmask, philox_seed, philox_offset  # intermediates
    )
    return out, residuals


def _mha_backward(
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size: Tuple[int, int],
    return_lse: bool,
    return_softmax: bool,
    config: Optional[Dict[str, Any]],
    residuals, 
    do
):
    q, k, v, bias, alibi_slopes = residuals[:5]
    out, lse, s_dmask, philox_seed, philox_offset = residuals[5:]

    dq = jnp.zeros_like(q)
    dk = jnp.empty_like(k)
    dv = jnp.empty_like(v)
    dbias = jnp.empty_like(bias) if bias is not None else None

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if _USE_FUSED_BWD_KERNEL:
        dq, dk, dv = _flash_attn_fused_backward(
            do,
            q, k, v,
            out,
            lse,
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
            config=config,
        )
    else:
        dq, dk, dv = _flash_attn_onekernel_backward(
            do,
            q, k, v,
            out,
            lse,
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
            config=config,
        )
    return dq, dk, dv, None, None


flash_attn_func.defvjp(_mha_forward, _mha_backward)
