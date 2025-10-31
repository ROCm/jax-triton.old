import jax
import jax.numpy as jnp
import jax_triton as jt
import ..triton.mha.mha_onekernel_bwd as backend  # adjust import to wherever you've defined the kernel

from absl.testing import absltest, parameterized

# common func
from common import (
    generate_mha_input,
    jax_attn_fwd_reference,
    jax_attn_bwd_reference,
)


class MHABWDTest(parameterized.TestCase):
  @parameterized.named_parameters(
      # (test_name, shape(B, S, H, D), dtype)
      ("mha_test1", (2, 1024, 32, 128), jnp.float32),
      ("mha_test2", (2, 1024, 32, 128), jnp.bfloat16),
  )
  def test_mha_bwd_triton_kernel(
                              self,
                              shape,
                              dtype):

    q, k, v, input_meta = generate_mha_input(shape, dtype)

    ### axlearn forward input: q, k, v
    ### axlearn forward output: out(), lse

    # Transpose to (B, H, S, D)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    out, softmax_lse = jax_attn_fwd_reference(q, k, v, input_meta)

    # run flash_attn_onekernel_backward
    dq, dk, dv = backend.flash_attn_onekernel_backward(q, k, v, input_meta)
    dq.block_until_ready()
    dq = jnp.transpose(dq, (0, 2, 1, 3))
    
    # expected via JAX
    _, dq_ref, dk_ref, dv_ref = jax_attn_bwd_reference(q, k, v)
    dq_ref.block_until_ready()
    dq_ref = jnp.transpose(dq_ref, (0, 2, 1, 3))

    # # shape & dtype sanity
    # self.assertEqual(out.shape, shape)
    # self.assertEqual(out.dtype, dtype)

    # numeric check (allow fp16 tolerance)
    atol = 1e-2 if dtype in (jnp.float16, jnp.bfloat16) else 1e-3
    max_diff = jnp.max(jnp.abs(dq - dq_ref))
    self.assertTrue(jnp.allclose(dq, dq_ref, atol=atol),
                    msg=f"Max diff ({max_diff}) exceeds tolerance (atol={atol})!")

if __name__ == "__main__":
  absltest.main()