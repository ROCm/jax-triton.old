import jax
import jax.numpy as jnp
import jax_triton as jt
import mha_fused_bwd as backend  # adjust import to wherever you've defined the kernel

from absl.testing import absltest, parameterized

# common func
from common import (
    generate_mha_input,
    jax_attn_fwd_reference,
    jax_attn_bwd_reference,
)

RNG_SEED = 42


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

    # generate input data, causal = True
    q, k, v, input_meta = generate_mha_input(shape, dtype)
    input_meta.need_causal()

    # Transpose to (B, H, S, D)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    # random upstream gradient
    key = jax.random.PRNGKey(RNG_SEED)
    do = jax.random.normal(key, q.shape, dtype=q.dtype)  # only when D_v == D_q

    # Triton results
    dq, dk, dv = backend.flash_attn_fused_backward(do, q, k, v, input_meta)
    dq.block_until_ready()
    dq = jnp.transpose(dq, (0, 2, 1, 3))
    
    # Reference results
    dq_ref, dk_ref, dv_ref = jax_attn_bwd_reference(q, k, v, do)
    dq_ref.block_until_ready()
    dq_ref = jnp.transpose(dq_ref, (0, 2, 1, 3))

    # shape & dtype sanity
    self.assertEqual(dq.shape, shape)
    self.assertEqual(dq.dtype, dtype)

    # numeric check (allow fp16 tolerance)
    atol = 1e-2 if dtype in (jnp.float16, jnp.bfloat16) else 1e-3
    max_diff = jnp.max(jnp.abs(dq - dq_ref))
    self.assertTrue(jnp.allclose(dq, dq_ref, atol=atol),
                    msg=f"Max diff ({max_diff}) exceeds tolerance (atol={atol})!")


if __name__ == "__main__":
  absltest.main()
