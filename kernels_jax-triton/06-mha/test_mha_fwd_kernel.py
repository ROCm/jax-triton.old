import jax
import jax.numpy as jnp
import triton_mha_fwd_kernel as backend  # adjust import to wherever you've defined gmm
import jax_triton as jt

from absl.testing import absltest, parameterized

#common func
from common import (
    generate_mha_input,
    jax_attn_fwd_reference
)


class MHAFWDTest(parameterized.TestCase):
  @parameterized.named_parameters(
      # (test_name, shape(B, S, H, D), dtype)
      # ("mha_test1", (2, 1024, 32, 128), jnp.float32),
      ("mha_test2", (2, 1024, 64, 128), jnp.bfloat16),
  )
  def test_mha_fwd_triton_kernel(
                              self,
                              shape,
                              dtype):


    q_tensor, k_tensor, v_tensor, input_meta = generate_mha_input(shape, dtype)

    ### axlearn forward input: q, k, v
    ### axlearn forward output: out(), lse

    # run _attention_fwd
    
    out = backend._attention_fwd(q_tensor, k_tensor, v_tensor, input_meta)
    out.block_until_ready()
    
    # expected via JAX ragged_dot
    expected = jax_attn_fwd_reference(q_tensor, k_tensor, v_tensor)
    
    # shape & dtype sanity
    self.assertEqual(out.shape, shape)
    self.assertEqual(out.dtype, dtype)

    # numeric check (allow fp16 tolerance)
    atol = 1e-2 if dtype == jnp.float16 else 1e-3
    self.assertTrue(jnp.allclose(out, expected, atol=atol),
                    msg=f"Mismatch: maxdiff={jnp.max(jnp.abs(out-expected))}")

if __name__ == "__main__":
  absltest.main()

  # B, S, H, D = 1, 4, 2, 8
  # key = jax.random.PRNGKey(42)
  # q = jax.random.normal(key, (B, S, H, D))
  # k = jax.random.normal(key, (B, S, H, D))
  # v = jax.random.normal(key, (B, S, H, D))