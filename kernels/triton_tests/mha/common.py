# Python standard library
import argparse
import typing
import math

# JAX
import jax
import jax.numpy as jnp
from jax._src import dtypes
import jax_triton as jt

#numpy
import numpy as np

import ctypes


# TODO: Figure out a sensible tiling default.
TILING: tuple[int, int, int] = (64, 64, 64)


# Default transposition.
TRANS_LHS: bool = False
TRANS_RHS: bool = False
TRANS_OUT: bool = False

def is_power_of_2(x: int) -> bool:
    return (x > 0) and (x & (x - 1) == 0)

def next_power_of_2(n: int) -> int:
    if n < 1:
        raise ValueError("n must be >= 1")
    return 2 ** math.ceil(math.log2(n))

# MHA Shape
BATCH_SIZE: int = 2
SEQ_LEN: int = 4096
NUM_HEADS: int = 64
HEAD_SIZE: int = 128
MHA_DATATYPE = jnp.float32
RNG_SEED = 42

MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)


def get_shape_from_layout(q, k, metadata):
    batch, nheads_q, _, head_size = q.shape
    nheads_k = k.shape[1]
    return batch, nheads_q, nheads_k, head_size

# TODO: This can probably optimized to have fewer lines of code.
#def get_strides_from_layout(q, k, v, o, metadata):
def get_strides_from_layout(q, k, v, metadata):
    q_strides = jt.strides_from_shape(q.shape)
    k_strides = jt.strides_from_shape(k.shape)
    v_strides = jt.strides_from_shape(v.shape)
    #o_strides = jt.strides_from_shape(o.shape)
    return q_strides, k_strides, v_strides#, o_strides

class MhaMetaData():
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    num_contexts = 0
    varlen = False
    dropout_p, return_encoded_softmax = 0.0, False

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

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

    def need_causal(self):
        self.causal = True

    def need_dropout(self, dropout_p, return_encoded_softmax):
        self.dropout_p = dropout_p
        self.return_encoded_softmax = return_encoded_softmax

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
            # TODO:Remove once dropout is supported with varlen
            assert self.dropout_p == 0.0
            assert not self.return_encoded_softmax
        else:
            assert q.ndim == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        #assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0    


# q.shape == k.shape == v.shape = (B, S, H, D)
def generate_mha_input(
    shape:tuple[int, ...] = MHA_SHAPE, 
    dtype:jnp.dtype = MHA_DATATYPE
):

    q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(RNG_SEED), 3)
    q = jax.random.normal(q_key, shape, dtype=dtype)
    k = jax.random.normal(k_key, shape, dtype=dtype)
    v = jax.random.normal(v_key, shape, dtype=dtype)

    sm_scale = shape[-1] ** -0.5
    input_metadata = MhaMetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = shape[1]
    input_metadata.max_seqlens_k = shape[1]
    return q, k, v, input_metadata


# q.shape != k.shape
def mha_input_helper(
    Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD,
    dtype=jnp.float16,
    requires_grad=True
):

    q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(RNG_SEED), 3)

    q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
    k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)

    q = jax.random.normal(q_key, q_tensor_shape, dtype=dtype)
    k = jax.random.normal(k_key, k_tensor_shape, dtype=dtype)
    v = jax.random.normal(v_key, k_tensor_shape, dtype=dtype)

    sm_scale = D_HEAD ** -0.5
    input_metadata = MhaMetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K

    return q, k, v, input_metadata  


def jax_attn_fwd_reference(q, k, v, dy=None):
    """
    Args:
        q, k, v: [batch, heads, seq_len, head_dim]
        dy: optional upstream gradient [batch, heads, seq_len, head_dim]
    
    Returns:
        y: [batch, heads, seq_len, head_dim]
        (dq, dk, dv): gradients, all [batch, heads, seq_len, head_dim]
    """

    _, _, seq_len, head_dim = q.shape
    dtype = q.dtype

    # Scaled dot-product attention
    logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)).astype("float32") / jnp.sqrt(head_dim)  # [B, H, N, N]

    # Causal mask: lower triangular
    # mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=q.dtype))
    # logits = jnp.where(mask == 1, logits, -jnp.inf)

    weights = jax.nn.softmax(logits, axis=-1)  # Softmax along keys
    output = jnp.matmul(weights, v)            # [B, H, N, d_v]

    return output.astype(dtype)


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
        logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(head_dim)
        mask = jnp.tril(jnp.ones_like(logits))
        logits = jnp.where(mask == 1, logits, -jnp.inf)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.matmul(weights, v)
        return out

    def attention_wrapped(qkv):
        return attention(*qkv)

    y_, vjp_fn = jax.vjp(attention_wrapped, (q, k, v))

    if dy is None:
        dy = jnp.ones_like(y_)  # Default upstream gradient
    else:
        dy = jnp.transpose(dy, (0, 2, 1, 3))

    (dq_, dk_, dv_), = vjp_fn(dy)

    return y_, (dq_, dk_, dv_)


def ragged_dot_reference(
    lhs,
    rhs,
    group_sizes,
) -> np.array:
  """Reference ragged dot implementation."""
  m, lk = lhs.shape
  group_count, rk, n = rhs.shape
  assert lk == rk
  assert group_count == group_sizes.shape[0]
  assert lhs.dtype == rhs.dtype

  out = np.zeros((m, n), dtype=lhs.dtype)
  result_iota = np.expand_dims(np.arange(out.shape[0]), list(range(1, out.ndim)))
  start = 0
  for i, size in enumerate(group_sizes):
    out += np.where(
        np.logical_and(start <= result_iota, result_iota < (start + size)),
        np.einsum(
          "nk,km->nm",
          lhs,
          rhs[i, :, :],
          dtype=np.float32 if lhs.dtype == dtypes.bfloat16 else np.float32,
        ),
        np.zeros(out.shape, dtype=out.dtype),
    )
    start += size
  return out.astype(dtypes.bfloat16) if lhs.dtype == dtypes.bfloat16 else out