import jax
import jax.numpy as jnp
from mha_pallas import flash_attention_forward


BATCH_SIZE: int = 1
SEQ_LEN: int = 128
NUM_HEADS: int = 16
HEAD_SIZE: int = 32
MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)
dtype = jnp.float16


def main(unused_argv):
    # create input data
    key = random.PRNGKey(42)
    q_key, k_key, v_key = random.split(key, 3)

    q = random.normal(q_key, MHA_SHAPE, dtype=dtype)
    k = random.normal(k_key, MHA_SHAPE, dtype=dtype)
    v = random.normal(v_key, MHA_SHAPE, dtype=dtype)
    # Bias is not supported in pallas, so we don't include it here.
    bias = None

    pallas_out = flash_attention_forward(q, k, v, bias)


if __name__ == "__main__":
    from absl import app
    app.run(main)
