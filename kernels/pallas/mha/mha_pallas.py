#!/usr/bin/env python3
# pylint: skip-file

import torch
import jax
import inspect
import jax.numpy as jnp
import triton  # pytype: disable=import-error

from gpu_attention_pallas_cj import mha as pallas_mha


def flash_attention_forward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: Optional[jnp.ndarray],
):
    segment_ids = None
    causal = True
    softmax_scale = q.shape[-1] ** (-0.5)
    num_warps = 2
    num_stages = 1
    # waves_per_eu = 4
    block_k, block_q = 16, 16

    for i in range(100):
        pallas_out = pallas_mha(
            q, k, v,
            segment_ids=None,
            causal=True,
        )

    return pallas_out


def flash_attention_backward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: Optional[jnp.ndarray],
):
    segment_ids = None
    causal = True
    softmax_scale = q.shape[-1] ** (-0.5)
    num_warps = 2
    num_stages = 1
    # waves_per_eu = 4
    block_k, block_q = 16, 16

    @jax.jit
    def pallas_fn(q, k, v):
        return pallas_mha(
            q, k, v, 
            segment_ids=segment_ids,
            causal=causal,
            sm_scale=softmax_scale,
            num_warps=num_warps,
            num_stages=num_stages,
            # waves_per_eu=waves_per_eu,
            block_k=block_k,
            block_q=block_q,
        ).sum()

    pallas_bwd = jax.grad(pallas_fn, argnums=(0, 1, 2))
    dq, dk, dv, _ = pallas_bwd(q, k, v)

    return dq, dk, dv
