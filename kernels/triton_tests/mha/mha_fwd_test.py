import jax
import jax.numpy as jnp
import triton_mha_fwd_kernel as backend  # adjust import to wherever you've defined gmm
import jax_triton as jt

from absl.testing import absltest, parameterized

# common func
from common import (
    generate_mha_input,
    jax_attn_fwd_reference
)


class MHAFWDTest(parameterized.TestCase):
  @parameterized.named_parameters(
      # (test_name, shape(B, S, H, D), dtype)
      ("mha_test1", (2, 1024, 32, 128), jnp.float32),
      ("mha_test2", (2, 1024, 32, 128), jnp.bfloat16),
  )
  def test_mha_fwd_triton_kernel(
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

    # run _attention_fwd
    out = backend._attention_fwd(q, k, v, input_meta)
    out.block_until_ready()
    out = jnp.transpose(out, (0, 2, 1, 3))
    
    # expected via JAX ragged_dot
    expected = jax_attn_fwd_reference(q, k, v)
    expected.block_until_ready()
    expected = jnp.transpose(expected, (0, 2, 1, 3))

    # shape & dtype sanity
    self.assertEqual(out.shape, shape)
    self.assertEqual(out.dtype, dtype)

    # numeric check (allow fp16 tolerance)
    atol = 1e-2 if dtype in (jnp.float16, jnp.bfloat16) else 1e-3
    max_diff = jnp.max(jnp.abs(out - expected))
    self.assertTrue(jnp.allclose(out, expected, atol=atol),
                    msg=f"Max diff ({max_diff}) exceeds tolerance (atol={atol})!")

if __name__ == "__main__":
  absltest.main()

  # B, S, H, D = 1, 4, 2, 8
  # key = jax.random.PRNGKey(42)
  # q = jax.random.normal(key, (B, S, H, D))
  # k = jax.random.normal(key, (B, S, H, D))
  # v = jax.random.normal(key, (B, S, H, D))