import jax
from jax import random
import jax.numpy as jnp
import triton  # pytype: disable=import-error

from gpu_attention_pallas_cj import mha as pallas_mha


segment_ids = None
causal = True
softmax_scale = 1.0
num_warps = 2
num_stages = 1
waves_per_eu = 4
block_k, block_q = 16, 16


BATCH_SIZE: int = 1
SEQ_LEN: int = 128
NUM_HEADS: int = 16
HEAD_SIZE: int = 32
MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)
dtype = jnp.float16


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
        block_q=block_q
    ).sum()


def main(unused_argv):
    # create input data
    key = random.PRNGKey(42)
    q_key, k_key, v_key, do_key = random.split(key, 4)

    q = random.normal(q_key, MHA_SHAPE, dtype=dtype)
    k = random.normal(k_key, MHA_SHAPE, dtype=dtype)
    v = random.normal(v_key, MHA_SHAPE, dtype=dtype)
    bias = None

    pallas_bwd = jax.grad(pallas_fn, argnums=(0, 1, 2))
    dq, dk, dv = pallas_bwd(q, k, v)
    print(dq)


if __name__ == "__main__":
    from absl import app
    app.run(main)
