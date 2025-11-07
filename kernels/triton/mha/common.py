# Python standard library
import argparse
import typing
import math

# JAX
import jax
import jax.numpy as jnp
from jax._src import dtypes
import jax_triton as jt


# MHA Shape
BATCH_SIZE: int = 2
NUM_HEADS: int = 64
SEQ_LEN: int = 4096
HEAD_SIZE: int = 128
MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)

# MHA dtype
MHA_DTYPE = jnp.float32

RNG_SEED = 42


def is_power_of_2(x: int) -> bool:
    return (x > 0) and (x & (x - 1) == 0)


def next_power_of_2(n: int) -> int:
    if n < 1:
        raise ValueError("n must be >= 1")
    return 2 ** math.ceil(math.log2(n))


def get_shape_from_layout(q, k, metadata):
    batch, nheads_q, _, head_size = q.shape
    nheads_k = k.shape[1]
    return batch, nheads_q, nheads_k, head_size


# TODO: This can probably optimized to have fewer lines of code.
# def get_strides_from_layout(q, k, v, o, metadata):
def get_strides_from_layout(q, k, v, metadata):
    q_strides = jt.strides_from_shape(q.shape)
    k_strides = jt.strides_from_shape(k.shape)
    v_strides = jt.strides_from_shape(v.shape)
    # o_strides = jt.strides_from_shape(o.shape)
    return q_strides, k_strides, v_strides#, o_strides


class MhaMetaData():
    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

        self.causal = False
        self.dropout_p = 0.0
        self.bias = None
        self.alibi_slopes = None

        self.varlen = False
        self.cu_seqlens_q = None
        self.cu_seqlens_k = None
        self.max_seqlens_q = 0
        self.max_seqlens_k = 0

    def need_causal(self):
        self.causal = True

    def need_dropout(self, dropout_p):
        self.dropout_p = dropout_p

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k):
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(
                cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(),
                self.max_seqlens_q
            )
            self.max_seqlens_k = max(
                cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(),
                self.max_seqlens_k
            )

    #def check_args(self, q, k, v, o):
    def check_args(self, q, k, v):
        assert q.ndim == k.ndim and q.ndim == v.ndim

        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, self)
        if self.varlen:
            assert q.ndim == 3
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias is None
            # TODO: Remove once dropout is supported with varlen
            assert self.dropout_p == 0.0
        else:
            assert q.ndim == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        # assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0    


# q.shape = k.shape = v.shape = (B, H, S, D)
def generate_mha_input(
    shape: tuple[int, ...] = MHA_SHAPE, 
    dtype: jnp.dtype = MHA_DTYPE
):

    q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(RNG_SEED), 3)
    q = jax.random.normal(q_key, shape, dtype=dtype)
    k = jax.random.normal(k_key, shape, dtype=dtype)
    v = jax.random.normal(v_key, shape, dtype=dtype)

    sm_scale = shape[-1] ** -0.5
    input_metadata = MhaMetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = shape[2]
    input_metadata.max_seqlens_k = shape[2]

    return q, k, v, input_metadata


# q.shape != k.shape
def mha_input_helper(
    Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD,
    dtype=jnp.float16,
    requires_grad=True
):

    q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(RNG_SEED), 3)

    q_shape = (Z, HQ, N_CTX_Q, D_HEAD)
    k_shape = (Z, HK, N_CTX_K, D_HEAD)
    v_shape = (Z, HK, N_CTX_K, D_HEAD)

    q = jax.random.normal(q_key, q_shape, dtype=dtype)
    k = jax.random.normal(k_key, k_shape, dtype=dtype)
    v = jax.random.normal(v_key, v_shape, dtype=dtype)

    sm_scale = D_HEAD ** -0.5
    input_metadata = MhaMetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K

    return q, k, v, input_metadata  


def jax_attn_fwd_reference(q, k, v):
    """
    Compute attention output and softmax_lse

    Args:
        q, k, v: [batch, heads, seq_len, head_dim]

    Returns:        
        out: [batch, heads, seq_len, head_dim]
        softmax_lse: [batch, heads, seq_len]
    """

    head_dim = q.shape[-1]
    sm_scale = 1.0 / jnp.sqrt(head_dim)

    # einsum for clarity: [B, H, Lq, D] @ [B, H, D, Lk] -> [B, H, Lq, Lk]
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * sm_scale

    # causal mask: [Lq, Lk]
    mask = jnp.tril(jnp.ones_like(logits))
    logits = jnp.where(mask == 1, logits, -jnp.inf)

    weights = jax.nn.softmax(logits, axis=-1)  # [B, H, Lq, Lk]
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)  # [B, H, Lq, D]

    # softmax_lse = logsumexp(logits, axis=-1), shape [B, H, Lq]
    softmax_lse = jax.scipy.special.logsumexp(logits, axis=-1)

    return out, softmax_lse


def jax_attn_bwd_reference(q, k, v, dy=None):
    """
    Compute attention output and its gradients w.r.t q, k, v

    Args:
        q, k, v: tensors of shape [batch, heads, seq_len, head_dim]
        dy: optional upstream gradient (same shape as output). If None, uses ones.

    Returns:
        y: attention output
        (dq, dk, dv): gradients of output w.r.t q, k, v
    """

    def attention(q, k, v):
        head_dim = q.shape[-1]
        sm_scale = 1.0 / jnp.sqrt(head_dim)
        logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * sm_scale
        mask = jnp.tril(jnp.ones_like(logits))
        logits = jnp.where(mask == 1, logits, -jnp.inf)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.matmul(weights, v)
        return out

    def attention_wrapped(qkv):
        return attention(*qkv)

    y, vjp_fn = jax.vjp(attention_wrapped, (q, k, v))

    if dy is None:
        dy = jnp.ones_like(y)  # Default upstream gradient

    (dq, dk, dv), = vjp_fn(dy)

    return dq, dk, dv
