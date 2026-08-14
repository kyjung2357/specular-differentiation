"""Internal mathematical primitives for the JAX backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

from jax import Array, vmap
import jax.numpy as jnp
from jax.typing import ArrayLike

from ._types import (
    Matrix,
    Scalar,
    ScalarToScalarFunc,
    ScalarToVectorFunc,
    Vector,
    VectorToScalarFunc,
    VectorToVectorFunc,
)


__all__ = ["derivative", "gradient", "jacobian"]


def _real_dtype(*arrays: Array) -> jnp.dtype:
    """Choose a real dtype without downcasting weak Python floats."""

    dtype = jnp.result_type(*arrays)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise TypeError("the specular kernels accept real inputs only")
    if not jnp.issubdtype(dtype, jnp.floating):
        return jnp.asarray(0.0).dtype
    if jnp.finfo(dtype).bits < 32:
        return jnp.dtype(jnp.float32)
    return dtype


def _broadcast_real(*values: ArrayLike) -> tuple[Array, ...]:
    """Broadcast real inputs and promote integral or sub-float32 values."""
    arrays = jnp.broadcast_arrays(*(jnp.asarray(value) for value in values))
    dtype = _real_dtype(*arrays)
    return tuple(jnp.asarray(array, dtype=dtype) for array in arrays)


def _one_infinite(x: Array, sign: Array) -> Array:
    """Evaluate ``C(x, sign * inf)`` without subtractive cancellation."""
    radius = jnp.hypot(jnp.ones_like(x), x)
    direct = x + sign * radius
    rationalized = (sign / radius) / (1 - sign * x / radius)
    return jnp.where(sign * x >= 0, direct, rationalized)


def _same_sign_C(
    alpha: Array,
    beta: Array,
    radius_a: Array,
    radius_b: Array,
) -> Array:
    """Evaluate the same-sign mean as a stable convex combination."""

    a_radius_high = radius_a >= radius_b
    ratio_a = radius_b / radius_a
    ratio_b = radius_a / radius_b
    from_a = beta + (alpha - beta) * (ratio_a / (1 + ratio_a))
    from_b = alpha + (beta - alpha) * (ratio_b / (1 + ratio_b))
    result = jnp.where(a_radius_high, from_a, from_b)
    return jnp.clip(result, jnp.minimum(alpha, beta), jnp.maximum(alpha, beta))


def _A(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> Array:
    r"""Evaluate the defining secant kernel
    :math:`\mathcal A:\mathbb R^2\times(0,\infty)\to\mathbb R` elementwise.

    For :math:`c>0`,
    :math:`\mathcal A(a,b,c)=\mathcal B(a/c,b/c)=\mathcal C(a/c,b/c)`.
    Inputs are converted to broadcast-compatible JAX arrays. Invalid entries
    produce ``NaN`` so the function remains JIT-compatible.
    """
    a_array, b_array, c_array = _broadcast_real(a, b, c)

    radius_a = jnp.hypot(a_array, c_array)
    radius_b = jnp.hypot(b_array, c_array)
    unit_a = a_array / radius_a
    unit_b = b_array / radius_b
    inverse_a = c_array / radius_a
    inverse_b = c_array / radius_b

    same_sign = jnp.signbit(a_array) == jnp.signbit(b_array)
    slope_a = a_array / c_array
    slope_b = b_array / c_array
    slope_radius_a = jnp.hypot(jnp.ones_like(slope_a), slope_a)
    slope_radius_b = jnp.hypot(jnp.ones_like(slope_b), slope_b)
    finite_slope_value = _same_sign_C(
        slope_a,
        slope_b,
        slope_radius_a,
        slope_radius_b,
    )
    direct_value = (unit_a + unit_b) / (inverse_a + inverse_b)
    direct = jnp.where(
        jnp.isfinite(slope_a) & jnp.isfinite(slope_b),
        finite_slope_value,
        direct_value,
    )
    radius_high = jnp.maximum(radius_a, radius_b)
    radius_low = jnp.minimum(radius_a, radius_b)
    rationalized = (
        ((a_array + b_array) / radius_high) * (c_array / radius_low)
    ) / (1 + inverse_a * inverse_b - unit_a * unit_b)
    result = jnp.where(same_sign, direct, rationalized)

    result = jnp.where(a_array == b_array, a_array / c_array, result)
    result = jnp.where(a_array == -b_array, jnp.zeros_like(result), result)

    valid = (
        jnp.isfinite(a_array)
        & jnp.isfinite(b_array)
        & jnp.isfinite(c_array)
        & (c_array > 0)
    )
    return jnp.where(valid, result, jnp.full_like(result, jnp.nan))


def _C(alpha: ArrayLike, beta: ArrayLike) -> Array:
    r"""Evaluate the algebraic slope mean
    :math:`\mathcal C:\overline{\mathbb R}^2\to\overline{\mathbb R}`
    elementwise.

    It equals :math:`\mathcal B`; its restriction to :math:`\mathbb R^2` is
    smooth and is the computational representation used for finite slopes.
    Inputs are converted to broadcast-compatible JAX arrays.
    """
    alpha_array, beta_array = _broadcast_real(alpha, beta)

    radius_alpha = jnp.hypot(jnp.ones_like(alpha_array), alpha_array)
    radius_beta = jnp.hypot(jnp.ones_like(beta_array), beta_array)
    unit_alpha = alpha_array / radius_alpha
    unit_beta = beta_array / radius_beta
    inverse_alpha = 1 / radius_alpha
    inverse_beta = 1 / radius_beta

    same_sign = jnp.signbit(alpha_array) == jnp.signbit(beta_array)
    direct = _same_sign_C(
        alpha_array,
        beta_array,
        radius_alpha,
        radius_beta,
    )
    radius_high = jnp.maximum(radius_alpha, radius_beta)
    radius_low = jnp.minimum(radius_alpha, radius_beta)
    rationalized = (
        ((alpha_array + beta_array) / radius_high) / radius_low
    ) / (1 + inverse_alpha * inverse_beta - unit_alpha * unit_beta)
    result = jnp.where(same_sign, direct, rationalized)

    alpha_infinite = jnp.isinf(alpha_array)
    beta_infinite = jnp.isinf(beta_array)
    result = jnp.where(
        alpha_infinite & ~beta_infinite,
        _one_infinite(beta_array, jnp.sign(alpha_array)),
        result,
    )
    result = jnp.where(
        ~alpha_infinite & beta_infinite,
        _one_infinite(alpha_array, jnp.sign(beta_array)),
        result,
    )

    result = jnp.where(alpha_array == beta_array, alpha_array, result)
    return jnp.where(
        alpha_array == -beta_array,
        jnp.zeros_like(result),
        result,
    )


def _B(alpha: ArrayLike, beta: ArrayLike) -> Array:
    r"""Evaluate the angular slope mean
    :math:`\mathcal B:\overline{\mathbb R}^2\to\overline{\mathbb R}`
    elementwise.

    It is the angle-based representation
    :math:`\tan((\arctan\alpha+\arctan\beta)/2)` and equals
    :math:`\mathcal C`. Inputs are converted to broadcast-compatible JAX
    arrays; :math:`\mathcal C` supplies saturated and extended-real cases.
    """
    alpha_array, beta_array = _broadcast_real(alpha, beta)
    half_angle = (
        jnp.arctan(alpha_array) + jnp.arctan(beta_array)
    ) / 2
    angular = jnp.tan(half_angle)

    half_pi = jnp.asarray(jnp.pi / 2, dtype=half_angle.dtype)
    angular_limit = jnp.nextafter(half_pi, jnp.zeros_like(half_pi))
    saturated = (
        ~jnp.isfinite(alpha_array)
        | ~jnp.isfinite(beta_array)
        | (jnp.abs(half_angle) >= angular_limit)
    )
    result = jnp.where(saturated, _C(alpha_array, beta_array), angular)

    result = jnp.where(alpha_array == beta_array, alpha_array, result)
    return jnp.where(
        alpha_array == -beta_array,
        jnp.zeros_like(result),
        result,
    )


def _point_and_step(
    x: ArrayLike,
    h: ArrayLike,
    *,
    point_ndim: int,
) -> tuple[Array, Array]:
    """Convert a point and scalar step using only static rank checks."""
    x_array = jnp.asarray(x)
    h_array = jnp.asarray(h)

    if x_array.ndim != point_ndim:
        kind = "scalar" if point_ndim == 0 else "vector"
        raise TypeError(
            f"Input 'x' must be a {kind}; got shape {x_array.shape}."
        )
    if point_ndim == 1 and x_array.shape[0] == 0:
        raise ValueError("Input 'x' must be a nonempty vector.")
    if h_array.ndim != 0:
        raise TypeError(
            f"Step size 'h' must be a scalar; got shape {h_array.shape}."
        )

    dtype = _real_dtype(x_array, h_array)
    return (
        jnp.asarray(x_array, dtype=dtype),
        jnp.asarray(h_array, dtype=dtype),
    )


def _function_value(
    f: Callable[[Array], ArrayLike],
    x: Array,
) -> Array:
    """Evaluate ``f`` and promote its result before differencing."""
    value = jnp.asarray(f(x))
    return jnp.asarray(value, dtype=_real_dtype(value))


def _require_output_rank(
    value: Array,
    expected_ndim: int | tuple[int, ...],
) -> None:
    """Enforce a function-output rank known during JAX tracing."""
    ranks = (
        (expected_ndim,)
        if isinstance(expected_ndim, int)
        else expected_ndim
    )
    if value.ndim not in ranks:
        expected = " or ".join(str(rank) for rank in ranks)
        raise TypeError(
            "Function 'f' returned an array with "
            f"ndim={value.ndim}; expected ndim={expected}."
        )
    if value.ndim == 1 and value.shape[0] == 0:
        raise ValueError("Function 'f' must return a nonempty vector.")


def _line_derivative(
    f: ScalarToScalarFunc | ScalarToVectorFunc,
    x: Array,
    h: Array,
) -> Array:
    """Differentiate a scalar-input function with scalar or vector output."""
    value = _function_value(f, x)
    _require_output_rank(value, (0, 1))

    right_value = _function_value(f, x + h)
    left_value = _function_value(f, x - h)
    if right_value.shape != value.shape or left_value.shape != value.shape:
        raise ValueError(
            "Function 'f' must return the same shape at x and x +/- h."
        )

    return _A(right_value - value, value - left_value, h)


def _coordinate_derivatives(
    f: VectorToScalarFunc | VectorToVectorFunc,
    x: Array,
    h: Array,
    *,
    output_ndim: int,
) -> Array:
    """Evaluate all coordinate derivatives with one cached centre value."""
    value = _function_value(f, x)
    _require_output_rank(value, output_ndim)

    offsets = h * jnp.eye(x.shape[0], dtype=x.dtype)

    def sample(offset: Array) -> tuple[Array, Array]:
        return (
            _function_value(f, x + offset),
            _function_value(f, x - offset),
        )

    right_values, left_values = vmap(sample)(offsets)
    expected_shape = (x.shape[0], *value.shape)
    if (
        right_values.shape != expected_shape
        or left_values.shape != expected_shape
    ):
        raise ValueError(
            "Function 'f' must return the same shape at x and x +/- h e_i."
        )

    return _A(right_values - value, value - left_values, h)


@overload
def derivative(
    f: ScalarToScalarFunc,
    x: ArrayLike,
    h: ArrayLike = 1e-6,
) -> Scalar: ...


@overload
def derivative(
    f: ScalarToVectorFunc,
    x: ArrayLike,
    h: ArrayLike = 1e-6,
) -> Vector: ...


def derivative(
    f: ScalarToScalarFunc | ScalarToVectorFunc,
    x: ArrayLike,
    h: ArrayLike = 1e-6,
) -> Scalar | Vector:
    r"""Approximate the specular derivative of a map from
    :math:`\mathbb R` to :math:`\mathbb R` or :math:`\mathbb R^m`.

    The center value is evaluated once, and the defining increment kernel
    :math:`\mathcal A` is applied directly without first forming quotients.
    """
    x_array, h_array = _point_and_step(x, h, point_ndim=0)
    return _line_derivative(f, x_array, h_array)


def gradient(
    f: VectorToScalarFunc,
    x: ArrayLike,
    h: ArrayLike = 1e-6,
) -> Vector:
    r"""Approximate the specular gradient of
    :math:`f:\mathbb R^n\to\mathbb R`.

    The result has shape ``(n,)``.
    """
    x_array, h_array = _point_and_step(x, h, point_ndim=1)
    return _coordinate_derivatives(
        f,
        x_array,
        h_array,
        output_ndim=0,
    )


def jacobian(
    f: VectorToVectorFunc,
    x: ArrayLike,
    h: ArrayLike = 1e-6,
) -> Matrix:
    r"""Approximate the specular Jacobian of
    :math:`f:\mathbb R^n\to\mathbb R^m`.

    The result has shape ``(m, n)``.
    """
    x_array, h_array = _point_and_step(x, h, point_ndim=1)
    values = _coordinate_derivatives(
        f,
        x_array,
        h_array,
        output_ndim=1,
    )
    return jnp.swapaxes(values, 0, 1)
