# Python standard library
import functools
import argparse
import typing
# import math
# from einops import rearrange, repeat

# JAX
import jax
import jax.numpy as jnp
from jax._src import dtypes
import jax_triton as jt

RNG_SEED = 42


# TODO: This can probably optimized to have fewer lines of code.
# def get_strides_from_layout(q, k, v, o, metadata):
def get_strides_from_layout(q, k, v, metadata):
    q_strides = jt.strides_from_shape(q.shape)
    k_strides = jt.strides_from_shape(k.shape)
    v_strides = jt.strides_from_shape(v.shape)
    # o_strides = jt.strides_from_shape(o.shape)
    return q_strides, k_strides, v_strides#, o_strides


@functools.partial(jax.jit, static_argnames=['causal'])
def jax_attn_reference(q, k, v, causal=False):
    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[1]
    sm_scale = 1.0 / jnp.sqrt(q.shape[-1])

    logits = jnp.einsum('bqhd,bkhd->bhqk', q, k)
    mask = None
    if causal:
        causal_mask = jnp.tril(jnp.ones((1, 1, q_seq_len, kv_seq_len), dtype=bool))
        causal_mask = jnp.broadcast_to(causal_mask, logits.shape)
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    logits = logits if mask is None else jnp.where(mask, logits, float("-inf"))

    weights = jax.nn.softmax(logits * sm_scale)
    out = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
    return out


# TODO: More args for pytest.
# def construct_local_mask(
#     seqlen_q,
#     seqlen_k,
#     window_size=(-1, -1),  # -1 means infinite window size
#     query_padding_mask=None,
#     key_padding_mask=None,
#     key_leftpad=None,
# ):
#     row_idx = rearrange(
#         jnp.arange(seqlen_q, dtype=jnp.int32), "s -> s 1"
#     )
#     col_idx = jnp.arange(seqlen_k, dtype=jnp.int32)
#     if key_leftpad is not None:
#         key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
#         col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
#         col_idx = jnp.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
#     sk = (
#         seqlen_k
#         if key_padding_mask is None
#         else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
#     )
#     sq = (
#         seqlen_q
#         if query_padding_mask is None
#         else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
#     )
#     if window_size[0] < 0:
#         return col_idx > row_idx + sk - sq + window_size[1]
#     else:
#         sk = jnp.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
#         return jnp.logical_or(
#             col_idx > jnp.minimum(row_idx + sk - sq + window_size[1], sk),
#             col_idx < row_idx + sk - sq - window_size[0],
#         )


# TODO: More args for pytest.
# def jax_attn_reference(
#     q,
#     k,
#     v,
#     query_padding_mask=None,
#     key_padding_mask=None,
#     attn_bias=None,
#     dropout_p=0.0,
#     dropout_mask=None,
#     causal=False,
#     window_size=(-1, -1),  # -1 means infinite window size
#     softcap=0.0,
#     upcast=True,
#     reorder_ops=False,
#     key_leftpad=None,
# ):
#     """
#     Arguments:
#         q: (batch_size, seqlen_q, nheads, head_dim_q)
#         k: (batch_size, seqlen_k, nheads_k, head_dim_q)
#         v: (batch_size, seqlen_k, nheads_k, head_dim_v)
#         query_padding_mask: (batch_size, seqlen_q)
#         key_padding_mask: (batch_size, seqlen_k)
#         attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
#         dropout_p: float
#         dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
#         causal: whether to apply causal masking
#         window_size: (int, int), left and right window size
#         upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
#             output back to fp16/bf16.
#         reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
#             without changing the math. This is to estimate the numerical error from operation
#             reordering.
#     Output:
#         output: (batch_size, seqlen_q, nheads, head_dim_v)
#         attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
#         lse: (batch_size, nheads, seqlen_q), logsumexp of scores
#     """
#     if causal:
#         window_size = (window_size[0], 0)
#     dtype_og = q.dtype
#     if upcast:
#         q = q.astype(jnp.float32)
#         k = k.astype(jnp.float32)
#         v = v.astype(jnp.float32)
#     seqlen_q, seqlen_k = q.shape[1], k.shape[1]
#     k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
#     v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
#     d = q.shape[-1]
#     if not reorder_ops:
#         scores = jnp.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
#     else:
#         scores = jnp.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
#     if softcap > 0:
#         scores = scores / softcap
#         scores = jnp.tanh(scores)
#         scores = scores * softcap
#     if key_padding_mask is not None:
#         scores = jnp.where(
#             rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"), scores
#         )
#     if window_size[0] >= 0 or window_size[1] >= 0:
#         local_mask = construct_local_mask(
#             seqlen_q,
#             seqlen_k,
#             window_size,
#             query_padding_mask,
#             key_padding_mask,
#             key_leftpad=key_leftpad,
#         )
#         scores = jnp.where(local_mask, float("-inf"), scores)
#     if attn_bias is not None:
#         scores = scores + attn_bias
#     lse = jax.nn.logsumexp(scores, dim=-1).astype(v.dtype)
#     attention = jax.nn.softmax(scores, dim=-1).astype(v.dtype)
#     # Some rows might be completely masked out so we fill them with zero instead of NaN
#     if window_size[0] >= 0 or window_size[1] >= 0:
#         attention = jnp.where(
#             jnp.all(local_mask, dim=-1, keepdim=True), 0.0, attention
#         )
#     # We want to mask here so that the attention matrix doesn't have any NaNs
#     # Otherwise we'll get NaN in dV
#     if query_padding_mask is not None:
#         attention = jnp.where(
#             rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0, attention
#         )
#     dropout_scaling = 1.0 / (1 - dropout_p)
#     # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
#     # output = jnp.einsum('bhts,bshd->bthd', attention_drop , v)
#     if dropout_mask is not None:
#         attention_drop = jnp.where(~dropout_mask, 0.0, attention)
#     else:
#         attention_drop = attention
#     output = jnp.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
#     if query_padding_mask is not None:
#         output = jnp.where(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0, output)
#     return (
#         output.astype(dtype_og),
#         attention.astype(dtype_og),
#         lse.astype(dtype_og),
#     )
