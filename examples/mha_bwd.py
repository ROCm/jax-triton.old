# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton  # type: ignore
import triton.language as tl  # type: ignore

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from jax_triton.ops.mha import (
    flash_attn_func,
    mha_set_use_fused_bwd_kernel
)
from jax_triton.test_mha_common import jax_attn_reference

# MHA_SHAPE = (1, 128, 16, 32)
MHA_SHAPE = (2, 1024, 64, 128)
MHA_DTYPE = jnp.float16


def main(unused_argv):
    # create input data
    key = random.PRNGKey(42)
    q_key, k_key, v_key, do_key = random.split(key, 4)

    q = random.normal(q_key, MHA_SHAPE, dtype=MHA_DTYPE)
    k = random.normal(k_key, MHA_SHAPE, dtype=MHA_DTYPE)
    v = random.normal(v_key, MHA_SHAPE, dtype=MHA_DTYPE)
    do = random.normal(do_key, q.shape, dtype=MHA_DTYPE)
    causal = False
    mha_set_use_fused_bwd_kernel(False)

    # Triton attention function
    def triton_mha_fwd(q, k, v):
        return flash_attn_func(
            q, k, v,
            causal=causal,
        )
    triton_out, triton_vjp_fn = jax.vjp(triton_mha_fwd, q, k, v)

    # Triton attention bwd
    triton_dq, triton_dk, triton_dv, *_ = triton_vjp_fn(do)
    triton_dq = triton_dq.block_until_ready()
    triton_dk = triton_dk.block_until_ready()
    triton_dv = triton_dv.block_until_ready()
    print(triton_dq)


if __name__ == "__main__":
    from absl import app
    app.run(main)
