import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.gpu.attention import mha as pallas_mha

from common import (
    jax_attn_fwd_reference,
    jax_attn_bwd_reference
)


BATCH_SIZE: int = 1
SEQ_LEN: int = 128
NUM_HEADS: int = 16
HEAD_SIZE: int = 32
SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE)

RUN_BWD: bool = True


def main() -> None:
    target_device = jax.devices("gpu")[0]

    np.random.seed(0)
    q = jnp.array(np.random.randn(*SHAPE).astype(np.float16), device=target_device)
    k = jnp.array(np.random.randn(*SHAPE).astype(np.float16), device=target_device)
    v = jnp.array(np.random.randn(*SHAPE).astype(np.float16), device=target_device)

    segment_ids = None
    sm_scale = q.shape[-1] ** (-0.5)
    causal = True

    if not RUN_BWD:
        for i in range(100):
            # Jax-Pallas attention fwd
            pallas_out = pallas_mha(
                q, k, v,
                segment_ids=segment_ids,
                sm_scale=sm_scale,
                causal=causal
            )
            pallas_out = pallas_out.block_until_ready()

        # # Reference attention fwd
        # q = jnp.transpose(q, (0, 2, 1, 3))
        # k = jnp.transpose(k, (0, 2, 1, 3))
        # v = jnp.transpose(v, (0, 2, 1, 3))

        # out, _ = jax_attn_fwd_reference(q, k, v)
        # out = out.block_until_ready()

        # out = jnp.transpose(out, (0, 2, 1, 3))

        # # numeric check
        # atol = 1e-2 if q.dtype in (jnp.float16, jnp.bfloat16) else 1e-3

        # max_diff = jnp.max(jnp.abs(out - pallas_out))
        # assert jnp.allclose(out, pallas_out.astype(out.dtype), atol=atol), \
        #     f"Max diff ({max_diff}) of out exceeds tolerance (atol={atol})"
        # print("Test passed!")
    else:
        do = jnp.array(np.random.randn(*SHAPE).astype(np.float16), device=target_device)
        # Jax-Pallas attention bwd
        mha_fwd = functools.partial(
            pallas_mha,
            segment_ids=segment_ids,
            sm_scale=sm_scale,
            causal=causal
        )
        for i in range(100):
            pallas_out, grad_fn = jax.vjp(mha_fwd, q, k, v)

            pallas_dq, pallas_dk, pallas_dv = grad_fn(do)
            pallas_dq = pallas_dq.block_until_ready()
            pallas_dk = pallas_dk.block_until_ready()
            pallas_dv = pallas_dv.block_until_ready()

        # # Reference attention bwd
        # q = jnp.transpose(q, (0, 2, 1, 3))
        # k = jnp.transpose(k, (0, 2, 1, 3))
        # v = jnp.transpose(v, (0, 2, 1, 3))
        # do = jnp.transpose(do, (0, 2, 1, 3))

        # dq, dk, dv = jax_attn_bwd_reference(q, k, v, do)
        # dq = dq.block_until_ready()
        # dk = dk.block_until_ready()
        # dv = dv.block_until_ready()

        # dq = jnp.transpose(dq, (0, 2, 1, 3))
        # dk = jnp.transpose(dk, (0, 2, 1, 3))
        # dv = jnp.transpose(dv, (0, 2, 1, 3))

        # # numeric check
        # atol = 1e-2 if dq.dtype in (jnp.float16, jnp.bfloat16) else 1e-3

        # max_diff = jnp.max(jnp.abs(dq - pallas_dq))
        # assert jnp.allclose(dq, pallas_dq.astype(dq.dtype), atol=atol), \
        #     f"Max diff ({max_diff}) of dq exceeds tolerance (atol={atol})"

        # max_diff = jnp.max(jnp.abs(dk - pallas_dk))
        # assert jnp.allclose(dk, pallas_dk.astype(dk.dtype), atol=atol), \
        #     f"Max diff ({max_diff}) of dk exceeds tolerance (atol={atol})"

        # max_diff = jnp.max(jnp.abs(dv - pallas_dv))
        # assert jnp.allclose(dv, pallas_dv.astype(dv.dtype), atol=atol), \
        #     f"Max diff ({max_diff}) of dv exceeds tolerance (atol={atol})"
        # print("Test passed!")


if __name__ == "__main__":
    main()
