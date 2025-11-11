# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Dict
import triton  # type: ignore
import triton.language as tl  # type: ignore

import jax
import jax.numpy as jnp
import jax_triton as jt

from mha_onekernel_bwd_kernel import _flash_attn_onekernel_backward
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

_USE_INT64_STRIDES = True


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
    dq, dk, dv = _flash_attn_onekernel_backward(do, q, k, v, input_meta)

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
