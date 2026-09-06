"""Internal mathematical primitives for the JAX backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

from jax import Array, custom_jvp, lax
from jax.custom_derivatives import SymbolicZero
from jax.extend.core import concrete_or_error
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

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
    high_value = jnp.where(a_radius_high, alpha, beta)
    low_value = jnp.where(a_radius_high, beta, alpha)
    radius_high = jnp.where(a_radius_high, radius_a, radius_b)
    radius_low = jnp.where(a_radius_high, radius_b, radius_a)
    result = (
        (high_value / radius_high) * (radius_low / 2) + low_value / 2
    ) / (0.5 + 0.5 * (radius_low / radius_high))
    return jnp.clip(result, jnp.minimum(alpha, beta), jnp.maximum(alpha, beta))


def _A_impl(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> Array:
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
    slope_value = _C(slope_a, slope_b)
    finite_slopes = jnp.isfinite(slope_a) & jnp.isfinite(slope_b)

    valid = (
        jnp.isfinite(a_array)
        & jnp.isfinite(b_array)
        & jnp.isfinite(c_array)
        & (c_array > 0)
    )
    diagonal = valid & (a_array == b_array)
    antidiagonal = valid & ~diagonal & (a_array == -b_array)
    same_nonfinite = (
        valid
        & ~(diagonal | antidiagonal)
        & same_sign
        & ~finite_slopes
    )
    a_high = jnp.abs(a_array) >= jnp.abs(b_array)
    raw_high = jnp.where(a_high, a_array, b_array)
    raw_low = jnp.where(a_high, b_array, a_array)
    high = jnp.where(same_nonfinite, raw_high, jnp.ones_like(raw_high))
    low = jnp.where(same_nonfinite, raw_low, jnp.zeros_like(raw_low))
    scale = jnp.where(same_nonfinite, c_array, jnp.ones_like(c_array))
    w = ((low / scale) - (scale / high)) / (1 + low / high)
    same_nonfinite_value = w + jnp.copysign(
        jnp.hypot(jnp.ones_like(w), w),
        high,
    )

    radius_high = jnp.maximum(radius_a, radius_b)
    radius_low = jnp.minimum(radius_a, radius_b)
    rationalized = (
        ((a_array + b_array) / radius_high) * (c_array / radius_low)
    ) / (1 + inverse_a * inverse_b - unit_a * unit_b)
    result = jnp.where(
        finite_slopes,
        slope_value,
        jnp.where(same_nonfinite, same_nonfinite_value, rationalized),
    )

    result = jnp.where(diagonal, a_array / c_array, result)
    result = jnp.where(antidiagonal, jnp.zeros_like(result), result)
    return jnp.where(valid, result, jnp.full_like(result, jnp.nan))


def _C_impl(alpha: ArrayLike, beta: ArrayLike) -> Array:
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


def _radius_parts(value: Array, scale: Array) -> tuple[Array, Array]:
    """Represent hypot(value, scale) without overflowing the radius."""
    magnitude = jnp.maximum(jnp.abs(value), scale)
    unit = jnp.hypot(value / magnitude, scale / magnitude)
    return magnitude, unit


def _linear_tangent(
    partials: tuple[Array, ...],
    tangents: tuple[Array | SymbolicZero, ...],
    value: Array,
    valid: Array,
) -> Array:
    """Apply the finite-domain differential without multiplying absent tangents."""
    result = jnp.zeros_like(value)
    for partial, tangent in zip(partials, tangents):
        if not isinstance(tangent, SymbolicZero):
            coefficient = jnp.where(valid, partial, jnp.full_like(partial, jnp.nan))
            result = result + coefficient * tangent
    return result


@custom_jvp
def _C_differentiable(alpha: Array, beta: Array) -> Array:
    return _C_impl(alpha, beta)


def _C_jvp(primals, tangents):
    alpha, beta = primals
    value = _C_differentiable(alpha, beta)
    radius = jnp.hypot(jnp.ones_like(value), value)
    partial_a = 0.5 * (radius / jnp.hypot(jnp.ones_like(alpha), alpha)) ** 2
    partial_b = 0.5 * (radius / jnp.hypot(jnp.ones_like(beta), beta)) ** 2
    valid = jnp.isfinite(alpha) & jnp.isfinite(beta) & jnp.isfinite(value)
    return value, _linear_tangent((partial_a, partial_b), tangents, value, valid)


_C_differentiable.defjvp(_C_jvp, symbolic_zeros=True)


def _C(alpha: ArrayLike, beta: ArrayLike) -> Array:
    """Evaluate C with its smooth finite-input differential under JAX transforms.

    Extended-real values retain their value semantics, but their differential
    is undefined and returns NaN.
    """
    return _C_differentiable(*_broadcast_real(alpha, beta))


@custom_jvp
def _A_differentiable(a: Array, b: Array, c: Array) -> Array:
    return _A_impl(a, b, c)


def _A_jvp(primals, tangents):
    a, b, c = primals
    value = _A_differentiable(a, b, c)
    radius = jnp.hypot(jnp.ones_like(value), value)
    radius_mantissa, radius_exponent = jnp.frexp(radius)
    c_mantissa, c_exponent = jnp.frexp(c)

    def partial(increment: Array) -> Array:
        magnitude, unit = _radius_parts(increment, c)
        mantissa, exponent = jnp.frexp(magnitude)
        return jnp.ldexp(
            0.5 * radius_mantissa**2 * c_mantissa / (mantissa * unit)**2,
            2 * radius_exponent + c_exponent - 2 * exponent,
        )

    partial_a = partial(a)
    partial_b = partial(b)
    # The scale is normally static. Avoid computing its potentially unbounded
    # differential when no scale tangent is present.
    if isinstance(tangents[2], SymbolicZero):
        partial_c = jnp.zeros_like(value)
    else:
        partial_c = -(partial_a * (a / c) + partial_b * (b / c))
    valid = (
        jnp.isfinite(a) & jnp.isfinite(b) & jnp.isfinite(c)
        & (c > 0) & jnp.isfinite(value)
    )
    return value, _linear_tangent(
        (partial_a, partial_b, partial_c), tangents, value, valid
    )


_A_differentiable.defjvp(_A_jvp, symbolic_zeros=True)


def _A(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> Array:
    """Evaluate the increment kernel with its finite-domain differential."""
    return _A_differentiable(*_broadcast_real(a, b, c))


def _scaled_mean_impl(alpha: Array, beta: Array, sigma: Array) -> Array:
    """Evaluate the scaled mean without forming potentially overflowing slopes."""
    magnitude_a, unit_a = _radius_parts(alpha, sigma)
    magnitude_b, unit_b = _radius_parts(beta, sigma)
    a_high = jnp.abs(alpha) >= jnp.abs(beta)
    high = jnp.where(a_high, alpha, beta)
    low = jnp.where(a_high, beta, alpha)
    magnitude_high = jnp.where(a_high, magnitude_a, magnitude_b)
    magnitude_low = jnp.where(a_high, magnitude_b, magnitude_a)
    unit_high = jnp.where(a_high, unit_a, unit_b)
    unit_low = jnp.where(a_high, unit_b, unit_a)
    unit_ratio = unit_low / unit_high
    radius_ratio = (magnitude_low / magnitude_high) * unit_ratio
    # If the high magnitude is set by its slope, high/magnitude_high is +/-1.
    # Keep the low radius before multiplying: its ratio to the high radius
    # may underflow even though its contribution to the mean is representable.
    high_term = jnp.where(
        sigma >= jnp.abs(high),
        high,
        jnp.copysign(magnitude_low, high),
    ) * unit_ratio
    high_term = jnp.clip(high_term, -jnp.abs(high), jnp.abs(high))
    large_sum = jnp.maximum(jnp.abs(high_term), jnp.abs(low)) > (
        jnp.finfo(high_term.dtype).max / 2
    )
    factor = jnp.where(large_sum, 0.5, 1.0)
    same_value = (factor * high_term + factor * low) / (
        factor + factor * radius_ratio
    )
    same_value = jnp.clip(
        same_value, jnp.minimum(alpha, beta), jnp.maximum(alpha, beta)
    )

    normalized_a = (alpha / magnitude_a) / unit_a
    normalized_b = (beta / magnitude_b) / unit_b
    inverse_a = (sigma / magnitude_a) / unit_a
    inverse_b = (sigma / magnitude_b) / unit_b
    denominator = 1 + inverse_a * inverse_b - normalized_a * normalized_b
    sum_mantissa, sum_exponent = jnp.frexp(alpha + beta)
    sigma_mantissa, sigma_exponent = jnp.frexp(sigma)
    a_mantissa, a_exponent = jnp.frexp(magnitude_a)
    b_mantissa, b_exponent = jnp.frexp(magnitude_b)
    opposite_value = jnp.ldexp(
        sum_mantissa * sigma_mantissa**2
        / (a_mantissa * b_mantissa * unit_a * unit_b * denominator),
        sum_exponent + 2 * sigma_exponent - a_exponent - b_exponent,
    )
    value = jnp.where(
        jnp.signbit(alpha) == jnp.signbit(beta), same_value, opposite_value
    )

    def one_infinite(x: Array, sign: Array) -> Array:
        magnitude, unit = _radius_parts(x, sigma)
        magnitude_mantissa, magnitude_exponent = jnp.frexp(magnitude)
        rationalized = jnp.ldexp(
            sign * sigma_mantissa**2
            / (magnitude_mantissa * (unit - sign * x / magnitude)),
            2 * sigma_exponent - magnitude_exponent,
        )
        direct = x + sign * (magnitude * unit)
        return jnp.where(sign * x >= 0, direct, rationalized)

    value = jnp.where(
        jnp.isinf(alpha) & jnp.isfinite(beta),
        one_infinite(beta, jnp.sign(alpha)), value,
    )
    value = jnp.where(
        jnp.isfinite(alpha) & jnp.isinf(beta),
        one_infinite(alpha, jnp.sign(beta)), value,
    )
    valid = jnp.isfinite(sigma) & (sigma > 0)
    value = jnp.where(valid, value, jnp.full_like(value, jnp.nan))
    # The facade has already validated sigma. Preserve these identities even
    # when XLA flushes a positive subnormal scale to zero during arithmetic.
    value = jnp.where(alpha == beta, alpha, value)
    return jnp.where(alpha == -beta, jnp.zeros_like(value), value)


@custom_jvp
def _scaled_mean_differentiable(alpha: Array, beta: Array, sigma: Array) -> Array:
    return _scaled_mean_impl(alpha, beta, sigma)


def _scaled_mean_jvp(primals, tangents):
    alpha, beta, sigma = primals
    value = _scaled_mean_differentiable(alpha, beta, sigma)
    magnitude, unit = _radius_parts(value, sigma)

    def partial(argument: Array) -> Array:
        argument_magnitude, argument_unit = _radius_parts(argument, sigma)
        return 0.5 * (
            (magnitude / argument_magnitude) * (unit / argument_unit)
        )**2

    partial_a, partial_b = partial(alpha), partial(beta)
    if isinstance(tangents[2], SymbolicZero):
        partial_sigma = jnp.zeros_like(value)
    else:
        partial_sigma = (
            value / sigma - partial_a * (alpha / sigma)
            - partial_b * (beta / sigma)
        )
        partial_sigma = jnp.where(
            alpha == beta, jnp.zeros_like(value), partial_sigma
        )
    valid = (
        jnp.isfinite(alpha) & jnp.isfinite(beta) & jnp.isfinite(sigma)
        & (sigma > 0) & jnp.isfinite(value)
    )
    return value, _linear_tangent(
        (partial_a, partial_b, partial_sigma), tangents, value, valid
    )


_scaled_mean_differentiable.defjvp(_scaled_mean_jvp, symbolic_zeros=True)


def _scaled_mean(alpha: ArrayLike, beta: ArrayLike, sigma: ArrayLike) -> Array:
    """Promote inputs before applying the scaled angular mean."""
    return _scaled_mean_differentiable(*_broadcast_real(alpha, beta, sigma))


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
    opposite_sign = jnp.signbit(alpha_array) != jnp.signbit(beta_array)
    result = jnp.where(opposite_sign, _C(alpha_array, beta_array), result)

    result = jnp.where(alpha_array == beta_array, alpha_array, result)
    return jnp.where(
        alpha_array == -beta_array,
        jnp.zeros_like(result),
        result,
    )


def _point_and_step(
    x: ArrayLike,
    h: ArrayLike | None,
    *,
    point_ndim: int,
) -> tuple[Array, Array]:
    """Convert a point and return explicit or dtype-adaptive steps."""
    x_array = jnp.asarray(x)

    if x_array.ndim != point_ndim:
        kind = "scalar" if point_ndim == 0 else "vector"
        raise TypeError(
            f"Input 'x' must be a {kind}; got shape {x_array.shape}."
        )
    if point_ndim == 1 and x_array.shape[0] == 0:
        raise ValueError("Input 'x' must be a nonempty vector.")

    if h is None:
        dtype = _real_dtype(x_array)
        point = jnp.asarray(x_array, dtype=dtype)
        base = jnp.cbrt(jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
        step = base * jnp.maximum(jnp.ones_like(point), jnp.abs(point))
    else:
        raw_h = jnp.asarray(h)
        if raw_h.ndim != 0:
            raise TypeError(
                f"Step size 'h' must be a scalar; got shape {raw_h.shape}."
            )
        if jnp.issubdtype(raw_h.dtype, jnp.bool_):
            raise TypeError("h must be a concrete real scalar")

        dtype = _real_dtype(x_array, raw_h)
        try:
            concrete_h = concrete_or_error(
                float,
                h,
                "h must be concrete under JAX transformations",
            )
        except TypeError as exc:
            raise TypeError(
                "h must be a concrete real scalar; close over it or pass it "
                "as a static argument under JAX transformations"
            ) from exc

        effective_h = np.asarray(concrete_h, dtype=np.dtype(dtype)).item()
        if not np.isfinite(effective_h) or effective_h <= 0.0:
            raise ValueError("h must be finite and greater than zero")

        point = jnp.asarray(x_array, dtype=dtype)
        step = jnp.asarray(effective_h, dtype=dtype)

    right_sample = point + step
    left_sample = point - step
    effective = (
        jnp.isfinite(right_sample)
        & jnp.isfinite(left_sample)
        & (right_sample != point)
        & (left_sample != point)
    )
    step = jnp.where(effective, step, jnp.full_like(step, jnp.nan))

    try:
        point_host = np.asarray(point)
        step_host = np.asarray(step)
    except TypeError:
        # A traced point cannot be value-inspected. The concrete step is still
        # validated above, and JAX evaluates the displacement at runtime.
        pass
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            right = point_host + step_host
            left = point_host - step_host
        if (
            np.any(~np.isfinite(right))
            or np.any(~np.isfinite(left))
            or np.any(right == point_host)
            or np.any(left == point_host)
        ):
            raise ValueError("h is too small or too large to perturb x")

    return point, step


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

    return _sample_derivative(right_value, value, left_value, h)


def _sample_derivative(right: Array, center: Array, left: Array, h: Array) -> Array:
    """Form finite increments without overflowing finite callback samples."""
    finite = jnp.isfinite(right) & jnp.isfinite(center) & jnp.isfinite(left)
    overflow = finite & (
        ~jnp.isfinite(right - center) | ~jnp.isfinite(center - left)
    )
    factor = jnp.where(overflow, 0.5, 1.0)
    step = h * factor
    value = _A(
        factor * right - factor * center,
        factor * center - factor * left,
        step,
    )
    # Some XLA devices flush a halved minimum-normal step to zero. Symmetry
    # still determines the value even when that rescaling is unrepresentable.
    return jnp.where(
        overflow & (step == 0) & (right == left), jnp.zeros_like(value), value
    )


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

    steps = jnp.broadcast_to(h, x.shape)
    def sample(coordinate: Array) -> Array:
        step = steps[coordinate]
        right = _function_value(f, x.at[coordinate].add(step))
        left = _function_value(f, x.at[coordinate].add(-step))
        if right.shape != value.shape or left.shape != value.shape:
            raise ValueError(
                "Function 'f' must return the same shape at x and x +/- h e_i."
            )
        return _sample_derivative(right, value, left, step)

    # A fixed batch bounds working storage by O(n) for scalar outputs instead
    # of constructing an n-by-n identity matrix and all displaced inputs.
    return lax.map(
        sample,
        jnp.arange(x.shape[0]),
        batch_size=min(32, x.shape[0]),
    )


@overload
def derivative(
    f: ScalarToScalarFunc,
    x: ArrayLike,
    h: ArrayLike | None = None,
) -> Scalar: ...


@overload
def derivative(
    f: ScalarToVectorFunc,
    x: ArrayLike,
    h: ArrayLike | None = None,
) -> Vector: ...


def derivative(
    f: ScalarToScalarFunc | ScalarToVectorFunc,
    x: ArrayLike,
    h: ArrayLike | None = None,
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
    h: ArrayLike | None = None,
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
    h: ArrayLike | None = None,
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
