# Python standard library
import argparse
import typing
import math

# JAX
import jax
import jax.numpy as jnp
from jax._src import dtypes
import jax_triton as jt


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


@jax.jit
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


@jax.jit
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
