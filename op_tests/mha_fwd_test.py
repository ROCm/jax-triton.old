import jax
import jax.numpy as jnp

from jax_triton.ops.mha import flash_attn_func
from jax_triton.test_mha_common import jax_attn_reference

from absl.testing import absltest, parameterized


class MHAFWDTest(parameterized.TestCase):
  @parameterized.named_parameters(
      # (test_name, shape(B, S, H, D), dtype)
      ("mha_test1", (2, 1024, 32, 128), jnp.float32),
      ("mha_test2", (2, 1024, 32, 128), jnp.bfloat16),
      ("mha_test3", (2, 1024, 64, 128), jnp.float16),
  )
  def test_mha_fwd(self, shape, dtype):
    # create input data
    key = jax.random.PRNGKey(42)
    q_key, k_key, v_key = jax.random.split(key, 3)

    q = jax.random.normal(q_key, shape, dtype=dtype)
    k = jax.random.normal(k_key, shape, dtype=dtype)
    v = jax.random.normal(v_key, shape, dtype=dtype)
    causal = True

    # Triton attention fwd
    triton_out = flash_attn_func(
        q, k, v,
        causal=causal,
    )
    triton_out = triton_out.block_until_ready()

    # Jax reference attention fwd
    jax_out = jax_attn_reference(
        q, k, v,
        causal=causal,
    )
    jax_out = jax_out.block_until_ready()

    # shape & dtype sanity
    self.assertEqual(triton_out.shape, shape)
    self.assertEqual(triton_out.dtype, dtype)

    # numeric check (allow fp16 tolerance)
    atol = 1e-2 if dtype in (jnp.float16, jnp.bfloat16) else 1e-3
    max_diff = jnp.max(jnp.abs(triton_out - jax_out))
    self.assertTrue(jnp.allclose(triton_out, jax_out, atol=atol),
                    msg=f"Max diff ({max_diff}) of out exceeds tolerance (atol={atol})!")


if __name__ == "__main__":
  absltest.main()
