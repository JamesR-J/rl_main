from collections.abc import Sequence
from functools import partial
import math
from operator import index
import typing
from typing import Hashable, Optional, Union
import warnings

import numpy as np

import jax.numpy as jnp
from jax import lax
from jax.numpy.linalg import cholesky, svd, eigh

from jax._src import config as config_lib
from jax._src import core
from jax._src import dtypes
from jax._src import prng
from jax._src import xla_bridge
from jax._src.api import jit, vmap
from jax._src.config import config
from jax._src.core import NamedShape
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import lax as lax_internal
from jax._src.numpy.lax_numpy import _convert_and_clip_integer
from jax._src.numpy.util import _arraylike, check_arraylike, promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike, DTypeLike
from jax._src.util import canonicalize_axis

RealArray = ArrayLike
IntegerArray = ArrayLike
# TODO: Import or define these to match
# https://github.com/numpy/numpy/blob/main/numpy/typing/_dtype_like.py.
DTypeLikeInt = DTypeLike
DTypeLikeUInt = DTypeLike
DTypeLikeFloat = DTypeLike
Shape = Sequence[int]

PRNGImpl = prng.PRNGImpl

# TODO(frostig,vanderplas): remove after deprecation window
KeyArray = Union[Array, prng.PRNGKeyArray]
KeyArrayLike = ArrayLike
PRNGKeyArray = prng.PRNGKeyArray

UINT_DTYPES = prng.UINT_DTYPES


def _isnan(x: ArrayLike) -> Array:
    return lax.ne(x, x)


def _check_prng_key(key) -> tuple[prng.PRNGKeyArray, bool]:
  # TODO(frostig): remove once we always enable_custom_prng
  if isinstance(key, prng.PRNGKeyArray):
    return key, False
  elif _arraylike(key):
    # Call random_wrap here to surface errors for invalid keys.
    wrapped_key = prng.random_wrap(key, impl=default_prng_impl())
    if config.jax_legacy_prng_key == 'error':
      raise ValueError(
        'Legacy uint32 key array passed as key to jax.random function. '
        'Please create keys using jax.random.key(). If use of a raw key array '
        'was intended, set jax_legacy_prng_key="allow".')
    elif config.jax_legacy_prng_key == 'warn':
      warnings.warn(
        'Legacy uint32 key array passed as key to jax.random function. '
        'Please create keys using jax.random.key(). If use of a raw key array '
        'was intended, set jax_legacy_prng_key="allow".', stacklevel=2)
    elif config.jax_enable_custom_prng:
      # TODO(jakevdp): possibly remove this warning condition.
      warnings.warn(
          'Raw arrays as random keys to jax.random functions are deprecated. '
          'Assuming valid threefry2x32 key for now.',
          FutureWarning)
    return wrapped_key, True
  else:
    raise TypeError(f'unexpected PRNG key type {type(key)}')


# TODO(frostig,vanderplas): remove from public API altogether, or at
# least change to return after asserting presence in `prng.prngs`
def default_prng_impl():
    """Get the default PRNG implementation.

  The default implementation is determined by ``config.jax_default_prng_impl``,
  which specifies it by name.
  """
    impl_name = config.jax_default_prng_impl
    assert impl_name in prng.prngs, impl_name
    return prng.prngs[impl_name]


def _check_shape(name: str, shape: Union[Shape, NamedShape], *param_shapes) -> None:
    shape = core.as_named_shape(shape)

    if param_shapes:
        shape_ = lax.broadcast_shapes(shape.positional, *param_shapes)
        if shape.positional != shape_:
            msg = ("{} parameter shapes must be broadcast-compatible with shape "
                   "argument, and the result of broadcasting the shapes must equal "
                   "the shape argument, but got result {} for shape argument {}.")
            raise ValueError(msg.format(name, shape_, shape))


@partial(jit, static_argnums=(3, 4, 5), inline=True)
def _binomial_inversion(key, count, prob, shape, dtype, max_iters):
    if config.jax_enable_checks:
        assert jnp.issubdtype(prob.dtype, jnp.floating)

    log1minusprob = jnp.log1p(-prob)

    def body_fn(carry):
        i, num_geom, geom_sum, key = carry
        subkey, key = split(key)
        num_geom_out = lax.select(geom_sum <= count, num_geom + 1, num_geom)
        u = uniform(subkey, shape, prob.dtype)
        geom = jnp.ceil(jnp.log(u) / log1minusprob)
        geom_sum = geom_sum + geom
        return i + 1, num_geom_out, geom_sum, key

    def cond_fn(carry):
        i, geom_sum = carry[0], carry[2]
        return (geom_sum <= count).any() & (i < max_iters)

    num_geom_init = lax.full_like(prob, 0, prob.dtype, shape)
    geom_sum_init = lax.full_like(prob, 0, prob.dtype, shape)
    carry = (0, num_geom_init, geom_sum_init, key)
    k = lax.while_loop(cond_fn, body_fn, carry)[1]
    return (k - 1).astype(dtype)


def uniform(key: KeyArrayLike,
            shape: Shape = (),
            dtype: DTypeLikeFloat = float,
            minval: RealArray = 0.,
            maxval: RealArray = 1.) -> Array:
    """Sample uniform random values in [minval, maxval) with given shape/dtype.

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    minval: optional, a minimum (inclusive) value broadcast-compatible with shape for the range (default 0).
    maxval: optional, a maximum (exclusive) value broadcast-compatible with shape for the range (default 1).

  Returns:
    A random array with the specified shape and dtype.
  """
    key, _ = _check_prng_key(key)
    dtypes.check_user_dtype_supported(dtype)
    shape = core.canonicalize_shape(shape)

    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(f"dtype argument to `uniform` must be a float dtype, "
                         f"got {dtype}")
    dtype = dtypes.canonicalize_dtype(dtype)
    return _uniform(key, shape, dtype, minval, maxval)


@partial(jit, static_argnums=(1, 2))
def _uniform(key, shape, dtype, minval, maxval) -> Array:
    _check_shape("uniform", shape)
    if not jnp.issubdtype(dtype, np.floating):
        raise TypeError("uniform only accepts floating point dtypes.")

    minval = lax.convert_element_type(minval, dtype)
    maxval = lax.convert_element_type(maxval, dtype)
    minval = lax.broadcast_to_rank(minval, len(shape))
    maxval = lax.broadcast_to_rank(maxval, len(shape))

    finfo = jnp.finfo(dtype)
    nbits, nmant = finfo.bits, finfo.nmant

    if nbits not in (8, 16, 32, 64):
        raise TypeError(
            f"uniform only accepts 8-, 16-, 32-, or 64-bit dtypesgot {dtype}."
        )

    rng_bits = nbits
    if nmant < 8:
        rng_bits = 8
    bits = _random_bits(key, rng_bits, shape)
    uint_dtype = UINT_DTYPES[nbits]
    if rng_bits != nbits:
        bits = lax.convert_element_type(bits, uint_dtype)

    # The strategy here is to randomize only the mantissa bits with an exponent of
    # 1 (after applying the bias), then shift and scale to the desired range. The
    # bit-level transformation we use relies on Numpy and XLA having bit-for-bit
    # equivalent float representations, which might not be true on all platforms.
    float_bits = lax.bitwise_or(
        lax.shift_right_logical(bits, np.array(rng_bits - nmant, uint_dtype)),
        np.array(1.0, dtype).view(uint_dtype),
    )
    floats = lax.bitcast_convert_type(float_bits, dtype) - np.array(1., dtype)
    return lax.max(
        minval,
        lax.reshape(floats * (maxval - minval) + minval, shape))


def _random_bits(key: KeyArray, bit_width: int, shape: Shape) -> Array:
    assert jnp.issubdtype(key.dtype, dtypes.prng_key)
    return prng.random_bits(key, bit_width=bit_width, shape=shape)


def _split(key: KeyArray, num: int | tuple[int, ...] = 2) -> KeyArray:
    # Alternative to split() to use within random samplers.
    # TODO(frostig): remove and use split(); we no longer need to wait
    # to always enable_custom_prng
    assert jnp.issubdtype(key.dtype, dtypes.prng_key)
    if key.ndim:
        raise TypeError("split accepts a single key, but was given a key array of "
                        f"shape {key.shape} != (). Use jax.vmap for batching.")
    shape = tuple(num) if isinstance(num, Sequence) else (num,)
    return prng.random_split(key, shape=shape)


def _return_prng_keys(was_wrapped, key):
    # TODO(frostig): remove once we always enable_custom_prng
    assert jnp.issubdtype(key.dtype, dtypes.prng_key)
    if config.jax_enable_custom_prng:
        return key
    else:
        return prng.random_unwrap(key) if was_wrapped else key


def split(key: KeyArrayLike, num: int | tuple[int, ...] = 2) -> KeyArray:
    """Splits a PRNG key into `num` new keys by adding a leading axis.

Args:
  key: a PRNG key (from ``key``, ``split``, ``fold_in``).
  num: optional, a positive integer (or tuple of integers) indicating
    the number (or shape) of keys to produce. Defaults to 2.

Returns:
  An array-like object of `num` new PRNG keys.
"""
    typed_key, wrapped = _check_prng_key(key)
    # typed_key, wrapped = _check_prng_key("split", key)  # TODO removed the string
    return _return_prng_keys(wrapped, _split(typed_key, num))


def _stirling_approx_tail(k):
    stirling_tail_vals = jnp.array(
        [
            0.0810614667953272,
            0.0413406959554092,
            0.0276779256849983,
            0.02079067210376509,
            0.0166446911898211,
            0.0138761288230707,
            0.0118967099458917,
            0.0104112652619720,
            0.00925546218271273,
            0.00833056343336287,
        ],
        dtype=k.dtype,
    )
    use_tail_values = k <= 9
    k = lax.clamp(0.0, k, 9.0)
    kp1sq = (k + 1) * (k + 1)
    approx = (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1)
    k = jnp.floor(k)
    return lax.select(use_tail_values, stirling_tail_vals[jnp.int32(k)], approx)


@partial(jit, static_argnums=(3, 4, 5), inline=True)
def _btrs(key, count, prob, shape, dtype, max_iters):
    # transforman-rejection algorithm
    # https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
    stddev = jnp.sqrt(count * prob * (1 - prob))
    b = 1.15 + 2.53 * stddev
    a = -0.0873 + 0.0248 * b + 0.01 * prob
    c = count * prob + 0.5
    v_r = 0.92 - 4.2 / b
    r = prob / (1 - prob)
    alpha = (2.83 + 5.1 / b) * stddev
    m = jnp.floor((count + 1) * prob)

    def body_fn(carry):
        i, k_out, accepted, key = carry
        key, subkey_0, subkey_1 = split(key, 3)
        u = uniform(subkey_0, shape, prob.dtype)
        v = uniform(subkey_1, shape, prob.dtype)
        u = u - 0.5
        us = 0.5 - jnp.abs(u)
        accept1 = (us >= 0.07) & (v <= v_r)
        k = jnp.floor((2 * a / us + b) * u + c)
        reject = (k < 0) | (k > count)
        v = jnp.log(v * alpha / (a / (us * us) + b))
        ub = (
                (m + 0.5) * jnp.log((m + 1) / (r * (count - m + 1)))
                + (count + 1) * jnp.log((count - m + 1) / (count - k + 1))
                + (k + 0.5) * jnp.log(r * (count - k + 1) / (k + 1))
                + _stirling_approx_tail(m)
                + _stirling_approx_tail(count - m)
                - _stirling_approx_tail(k)
                - _stirling_approx_tail(count - k)
        )
        accept2 = v <= ub
        accept = accept1 | (~reject & accept2)
        k_out = lax.select(accept, k, k_out)
        accepted |= accept
        return i + 1, k_out, accepted, key

    def cond_fn(carry):
        i, accepted = carry[0], carry[2]
        return (~accepted).any() & (i < max_iters)

    k_init = lax.full_like(prob, -1, prob.dtype, shape)
    carry = (0, k_init, jnp.full(shape, False, jnp.bool_), key)
    return lax.while_loop(cond_fn, body_fn, carry)[1].astype(dtype)


@partial(jit, static_argnums=(3, 4), inline=True)
def _binomial(key, count, prob, shape, dtype) -> Array:
    # The implementation matches TensorFlow and TensorFlow Probability:
    # https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/core/kernels/random_binomial_op.cc
    # and tensorflow_probability.substrates.jax.distributions.Binomial
    # For n * p < 10, we use the binomial inverse algorithm; otherwise btrs.
    if shape is None:
        shape = jnp.broadcast_shapes(jnp.shape(count), jnp.shape(prob))
    else:
        _check_shape("binomial", shape, np.shape(count), np.shape(prob))
    (prob,) = promote_dtypes_inexact(prob)
    count = lax.convert_element_type(count, prob.dtype)
    count = jnp.broadcast_to(count, shape)
    prob = jnp.broadcast_to(prob, shape)
    p_lt_half = prob < 0.5
    q = lax.select(p_lt_half, prob, 1.0 - prob)
    count_nan_or_neg = _isnan(count) | (count < 0.0)
    count_inf = jnp.isinf(count)
    q_is_nan = _isnan(q)
    q_l_0 = q < 0.0
    q = lax.select(q_is_nan | q_l_0, lax.full_like(q, 0.01), q)
    use_inversion = count_nan_or_neg | (count * q <= 10.0)

    # consistent with np.random.binomial behavior for float count input
    count = jnp.floor(count)

    count_inv = lax.select(use_inversion, count, lax.full_like(count, 0.0))
    count_btrs = lax.select(use_inversion, lax.full_like(count, 1e4), count)
    q_btrs = lax.select(use_inversion, lax.full_like(q, 0.5), q)
    max_iters = dtype.type(jnp.finfo(dtype).max)
    samples = lax.select(
        use_inversion,
        _binomial_inversion(key, count_inv, q, shape, dtype, max_iters),
        _btrs(key, count_btrs, q_btrs, shape, dtype, max_iters),
    )
    # ensure nan q always leads to nan output and nan or neg count leads to nan
    # as discussed in https://github.com/jax-ml/jax/pull/16134#pullrequestreview-1446642709
    invalid = (q_l_0 | q_is_nan | count_nan_or_neg)
    samples = lax.select(
        invalid,
        jnp.full_like(samples, jnp.nan, dtype),
        samples,
    )

    # +inf count leads to inf
    samples = lax.select(
        count_inf & (~invalid),
        jnp.full_like(samples, jnp.inf, dtype),
        samples,
    )

    samples = lax.select(
        p_lt_half | count_nan_or_neg | q_is_nan | count_inf,
        samples,
        count.astype(dtype) - samples,
    )
    return samples


def binomial(
        key: KeyArray,
        n: RealArray,
        p: RealArray,
        shape: Shape | None = None,
        dtype: DTypeLikeFloat = float,
) -> Array:
    r"""Sample Binomial random values with given shape and float dtype.

  The values are returned according to the probability mass function:

  .. math::
      f(k;n,p) = \binom{n}{k}p^k(1-p)^{n-k}

  on the domain :math:`0 < p < 1`, and where :math:`n` is a nonnegative integer
  representing the number of trials and :math:`p` is a float representing the
  probability of success of an individual trial.

  Args:
    key: a PRNG key used as the random key.
    n: a float or array of floats broadcast-compatible with ``shape``
      representing the number of trials.
    p: a float or array of floats broadcast-compatible with ``shape``
      representing the probability of success of an individual trial.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``n`` and ``p``.
      The default (None) produces a result shape equal to ``np.broadcast(n, p).shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by
    ``np.broadcast(n, p).shape``.
  """
    # key, _ = _check_prng_key("binomial", key)  # TODO removed the string
    key, _ = _check_prng_key(key)
    check_arraylike("binomial", n, p)
    dtypes.check_user_dtype_supported(dtype)
    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(
            "dtype argument to `binomial` must be a float dtype, got {dtype}"
        )
    dtype = dtypes.canonicalize_dtype(dtype)
    if shape is not None:
        shape = core.canonicalize_shape(shape)
    return _binomial(key, n, p, shape, dtype)
