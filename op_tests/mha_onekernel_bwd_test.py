import jax
import jax.numpy as jnp

from jax_triton.ops.mha import flash_attn_func
from jax_triton.test_mha_common import jax_attn_reference

from absl.testing import absltest, parameterized


class MHABWDTest(parameterized.TestCase):
  @parameterized.named_parameters(
      # (test_name, shape(B, S, H, D), dtype)
      ("mha_test1", (2, 1024, 32, 128), jnp.float32),
      ("mha_test2", (2, 1024, 32, 128), jnp.bfloat16),
  )
  def test_mha_onekernel_bwd(self, shape, dtype):
    # create input data
    key = jax.random.PRNGKey(42)
    q_key, k_key, v_key, do_key = jax.random.split(key, 4)

    q = jax.random.normal(q_key, MHA_SHAPE, dtype=MHA_DTYPE)
    k = jax.random.normal(k_key, MHA_SHAPE, dtype=MHA_DTYPE)
    v = jax.random.normal(v_key, MHA_SHAPE, dtype=MHA_DTYPE)
    do = jax.random.normal(do_key, q.shape, dtype=MHA_DTYPE)
    causal = True

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

    # Jax reference attention function
    def mha_fwd(q, k, v):
        return jax_attn_reference(
            q, k, v,
            causal=causal
        )
    jax_out, vjp_fn = jax.vjp(mha_fwd, q, k, v)
    
    # Jax reference attention bwd
    jax_dq, jax_dk, jax_dv = vjp_fn(do)
    jax_dq = jax_dq.block_until_ready()
    jax_dk = jax_dk.block_until_ready()
    jax_dv = jax_dv.block_until_ready()

    # shape & dtype sanity
    self.assertEqual(dq.shape, shape)
    self.assertEqual(dq.dtype, dtype)

    # numeric check (allow fp16 tolerance)
    atol = 1e-2 if dtype in (jnp.float16, jnp.bfloat16) else 1e-3
    max_diff = jnp.max(jnp.abs(triton_dq - jax_dq))
    self.assertTrue(jnp.allclose(triton_dq, jax_dq, atol=atol),
                    msg=f"Max diff ({max_diff}) of dq exceeds tolerance (atol={atol})!")
    max_diff = jnp.max(jnp.abs(triton_dk - jax_dk))
    self.assertTrue(jnp.allclose(triton_dk, jax_dk, atol=atol),
                    msg=f"Max diff ({max_diff}) of dk exceeds tolerance (atol={atol})!")
    max_diff = jnp.max(jnp.abs(triton_dv - jax_dv))
    self.assertTrue(jnp.allclose(triton_dv, jax_dv, atol=atol),
                    msg=f"Max diff ({max_diff}) of dv exceeds tolerance (atol={atol})!")


if __name__ == "__main__":
  absltest.main()
