"""Internal mathematical primitives for the NumPy backend."""

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt

from ._types import (
    Matrix,
    Scalar,
    ScalarToScalarFunc,
    ScalarToVectorFunc,
    Vector,
    VectorToScalarFunc,
    VectorToVectorFunc,
)


type RealInput = npt.ArrayLike
type RealArray = npt.NDArray[np.float64]
type RealResult = float | RealArray

__all__ = ["derivative", "gradient", "jacobian"]


def _broadcast_real(*values: RealInput) -> tuple[RealArray, ...]:
    """Convert real inputs to broadcast-compatible float64 arrays."""

    arrays = tuple(np.asarray(value) for value in values)
    if any(np.iscomplexobj(array) for array in arrays):
        raise TypeError("the specular kernels accept real inputs only")

    try:
        return tuple(
            np.asarray(array, dtype=np.float64)
            for array in np.broadcast_arrays(*arrays)
        )
    except ValueError as exc:
        raise ValueError("inputs are not broadcast-compatible") from exc


def _finish(value: RealArray) -> RealResult:
    """Return a Python float for scalar input and an array otherwise."""

    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        return float(result)
    return result


def _one_infinite(x: RealArray, sign: RealArray) -> RealArray:
    """Evaluate C(x, sign * infinity) without subtractive cancellation."""

    radius = np.hypot(1.0, x)
    direct = sign * x >= 0.0
    result = np.empty_like(x)
    with np.errstate(over="ignore"):
        result[direct] = x[direct] + sign[direct] * radius[direct]
    result[~direct] = (sign[~direct] / radius[~direct]) / (
        1.0 - sign[~direct] * x[~direct] / radius[~direct]
    )
    return result


def _divide_by_product(
    numerator: RealArray,
    factor_a: RealArray,
    factor_b: RealArray,
    denominator: RealArray,
) -> RealArray:
    """Evaluate numerator / (factor_a * factor_b * denominator) by exponent."""

    factor_high = np.maximum(factor_a, factor_b)
    factor_low = np.minimum(factor_a, factor_b)
    scaled_numerator = numerator / factor_high
    factor_low_mantissa, factor_low_exponent = np.frexp(factor_low)
    with np.errstate(under="ignore"):
        return np.ldexp(
            scaled_numerator / (factor_low_mantissa * denominator),
            -factor_low_exponent,
        )


def _same_sign_C(
    alpha: RealArray,
    beta: RealArray,
    radius_a: RealArray,
    radius_b: RealArray,
) -> RealArray:
    """Evaluate the same-sign mean as a stable convex combination."""

    result = np.empty_like(alpha)
    a_radius_high = radius_a >= radius_b
    if np.any(a_radius_high):
        ratio = radius_b[a_radius_high] / radius_a[a_radius_high]
        weight_a = ratio / (1.0 + ratio)
        av = alpha[a_radius_high]
        bv = beta[a_radius_high]
        result[a_radius_high] = bv + (av - bv) * weight_a

    b_radius_high = ~a_radius_high
    if np.any(b_radius_high):
        ratio = radius_a[b_radius_high] / radius_b[b_radius_high]
        weight_b = ratio / (1.0 + ratio)
        av = alpha[b_radius_high]
        bv = beta[b_radius_high]
        result[b_radius_high] = av + (bv - av) * weight_b

    return np.clip(result, np.minimum(alpha, beta), np.maximum(alpha, beta))


def _finite_C(alpha: RealArray, beta: RealArray) -> RealArray:
    """Evaluate C on finite flattened inputs using stable sign branches."""

    result = np.empty_like(alpha)
    same_sign = np.signbit(alpha) == np.signbit(beta)

    if np.any(same_sign):
        a = alpha[same_sign]
        b = beta[same_sign]
        radius_a = np.hypot(1.0, a)
        radius_b = np.hypot(1.0, b)
        result[same_sign] = _same_sign_C(a, b, radius_a, radius_b)

    opposite_sign = ~same_sign
    if np.any(opposite_sign):
        a = alpha[opposite_sign]
        b = beta[opposite_sign]
        radius_a = np.hypot(1.0, a)
        radius_b = np.hypot(1.0, b)
        unit_a = a / radius_a
        unit_b = b / radius_b
        inverse_a = 1.0 / radius_a
        inverse_b = 1.0 / radius_b
        denominator = 1.0 + inverse_a * inverse_b - unit_a * unit_b
        result[opposite_sign] = _divide_by_product(
            a + b, radius_a, radius_b, denominator
        )

    return result


def _A(a: RealInput, b: RealInput, c: RealInput) -> RealResult:
    r"""Evaluate the defining secant kernel
    :math:`\mathcal A:\mathbb R^2\times(0,\infty)\to\mathbb R` elementwise.

    For :math:`c>0`,
    :math:`\mathcal A(a,b,c)=\mathcal B(a/c,b/c)=\mathcal C(a/c,b/c)`.
    Inputs are converted to broadcast-compatible NumPy arrays.
    Entries outside the mathematical domain produce ``NaN``.
    """

    a_array, b_array, c_array = _broadcast_real(a, b, c)
    shape = a_array.shape
    a_flat = a_array.reshape(-1)
    b_flat = b_array.reshape(-1)
    c_flat = c_array.reshape(-1)
    result = np.empty_like(a_flat)

    valid = (
        np.isfinite(a_flat)
        & np.isfinite(b_flat)
        & np.isfinite(c_flat)
        & (c_flat > 0.0)
    )
    result[~valid] = np.nan

    diagonal = valid & (a_flat == b_flat)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result[diagonal] = a_flat[diagonal] / c_flat[diagonal]

    antidiagonal = valid & ~diagonal & (a_flat == -b_flat)
    result[antidiagonal] = 0.0

    unresolved = valid & ~(diagonal | antidiagonal)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        slope_a = a_flat / c_flat
        slope_b = b_flat / c_flat

    finite_slopes = unresolved & np.isfinite(slope_a) & np.isfinite(slope_b)
    if np.any(finite_slopes):
        result[finite_slopes] = _finite_C(
            slope_a[finite_slopes],
            slope_b[finite_slopes],
        )

    nonfinite_slopes = unresolved & ~finite_slopes
    same_sign = nonfinite_slopes & (
        np.signbit(a_flat) == np.signbit(b_flat)
    )
    if np.any(same_sign):
        av = a_flat[same_sign]
        bv = b_flat[same_sign]
        cv = c_flat[same_sign]
        a_high = np.abs(av) >= np.abs(bv)
        high = np.where(a_high, av, bv)
        low = np.where(a_high, bv, av)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            w = ((low / cv) - (cv / high)) / (1.0 + low / high)
            result[same_sign] = w + np.copysign(np.hypot(1.0, w), high)

    opposite_sign = nonfinite_slopes & ~same_sign
    if np.any(opposite_sign):
        av = a_flat[opposite_sign]
        bv = b_flat[opposite_sign]
        cv = c_flat[opposite_sign]
        radius_a = np.hypot(av, cv)
        radius_b = np.hypot(bv, cv)
        unit_a = av / radius_a
        unit_b = bv / radius_b
        inverse_a = cv / radius_a
        inverse_b = cv / radius_b
        denominator = 1.0 + inverse_a * inverse_b - unit_a * unit_b
        radius_high = np.maximum(radius_a, radius_b)
        radius_low = np.minimum(radius_a, radius_b)
        result[opposite_sign] = (
            ((av + bv) / radius_high)
            * (cv / radius_low)
            / denominator
        )

    return _finish(result.reshape(shape))


def _B(alpha: RealInput, beta: RealInput) -> RealResult:
    r"""Evaluate the angular slope mean
    :math:`\mathcal B:\overline{\mathbb R}^2\to\overline{\mathbb R}`
    elementwise.

    It is the angle-based representation
    :math:`\tan((\arctan\alpha+\arctan\beta)/2)` and equals
    :math:`\mathcal C`. Inputs are converted to broadcast-compatible NumPy
    arrays.
    """

    alpha_array, beta_array = _broadcast_real(alpha, beta)
    shape = alpha_array.shape
    alpha_flat = alpha_array.reshape(-1)
    beta_flat = beta_array.reshape(-1)
    result = np.empty_like(alpha_flat)

    nan_entries = np.isnan(alpha_flat) | np.isnan(beta_flat)
    result[nan_entries] = np.nan

    diagonal = ~nan_entries & (alpha_flat == beta_flat)
    result[diagonal] = alpha_flat[diagonal]

    antidiagonal = ~diagonal & (alpha_flat == -beta_flat)
    result[antidiagonal] = 0.0

    unresolved = ~(nan_entries | diagonal | antidiagonal)
    alpha_infinite = np.isinf(alpha_flat)
    beta_infinite = np.isinf(beta_flat)

    alpha_only = unresolved & alpha_infinite & ~beta_infinite
    if np.any(alpha_only):
        result[alpha_only] = _one_infinite(
            beta_flat[alpha_only], np.sign(alpha_flat[alpha_only])
        )

    beta_only = unresolved & ~alpha_infinite & beta_infinite
    if np.any(beta_only):
        result[beta_only] = _one_infinite(
            alpha_flat[beta_only], np.sign(beta_flat[beta_only])
        )

    finite = unresolved & ~alpha_infinite & ~beta_infinite
    if np.any(finite):
        av = alpha_flat[finite]
        bv = beta_flat[finite]
        opposite_sign = np.signbit(av) != np.signbit(bv)
        values = np.empty_like(av)
        values[opposite_sign] = _finite_C(
            av[opposite_sign], bv[opposite_sign]
        )

        same_sign = ~opposite_sign
        angle = 0.5 * (
            np.arctan(av[same_sign]) + np.arctan(bv[same_sign])
        )
        same_sign_values = np.tan(angle)

        # arctan can round a very large finite slope to pi/2.
        # Fall back to the algebraic representation before tan loses its magnitude.
        angle_limit = np.nextafter(np.pi / 2.0, 0.0)
        saturated = np.abs(angle) >= angle_limit
        if np.any(saturated):
            same_sign_values[saturated] = _finite_C(
                av[same_sign][saturated],
                bv[same_sign][saturated],
            )
        values[same_sign] = same_sign_values
        result[finite] = values

    return _finish(result.reshape(shape))


def _C(alpha: RealInput, beta: RealInput) -> RealResult:
    r"""Evaluate the algebraic slope mean
    :math:`\mathcal C:\overline{\mathbb R}^2\to\overline{\mathbb R}`
    elementwise.

    It equals :math:`\mathcal B`; its restriction to :math:`\mathbb R^2` is
    smooth and is the computational representation used for finite slopes.
    Inputs are converted to broadcast-compatible NumPy arrays.
    """

    alpha_array, beta_array = _broadcast_real(alpha, beta)
    shape = alpha_array.shape
    alpha_flat = alpha_array.reshape(-1)
    beta_flat = beta_array.reshape(-1)
    result = np.empty_like(alpha_flat)

    nan_entries = np.isnan(alpha_flat) | np.isnan(beta_flat)
    result[nan_entries] = np.nan

    diagonal = ~nan_entries & (alpha_flat == beta_flat)
    result[diagonal] = alpha_flat[diagonal]

    antidiagonal = ~diagonal & (alpha_flat == -beta_flat)
    result[antidiagonal] = 0.0

    unresolved = ~(nan_entries | diagonal | antidiagonal)
    alpha_infinite = np.isinf(alpha_flat)
    beta_infinite = np.isinf(beta_flat)

    alpha_only = unresolved & alpha_infinite & ~beta_infinite
    if np.any(alpha_only):
        result[alpha_only] = _one_infinite(
            beta_flat[alpha_only], np.sign(alpha_flat[alpha_only])
        )

    beta_only = unresolved & ~alpha_infinite & beta_infinite
    if np.any(beta_only):
        result[beta_only] = _one_infinite(
            alpha_flat[beta_only], np.sign(beta_flat[beta_only])
        )

    finite = unresolved & ~alpha_infinite & ~beta_infinite
    if np.any(finite):
        result[finite] = _finite_C(alpha_flat[finite], beta_flat[finite])

    return _finish(result.reshape(shape))


def _real_scalar(value: object, *, name: str) -> float:
    """Normalize a scalar argument without imposing a value-domain policy."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real")
    if array.ndim != 0:
        raise TypeError(f"{name} must be a scalar; got shape {array.shape}")
    try:
        return float(array)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc


def _positive_step(value: object) -> float:
    """Require a finite, positive scalar step before evaluating a callback."""

    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.ndim != 0:
        raise TypeError("h must be a concrete real scalar")
    step = float(array)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("h must be finite and greater than zero")
    return step


def _step_values(x: RealArray, h: Scalar | None) -> RealArray:
    """Return explicit or dtype-adaptive steps matching the point shape."""

    if h is None:
        base = np.cbrt(np.finfo(np.float64).eps)
        steps = base * np.maximum(1.0, np.abs(x))
    else:
        steps = np.full_like(x, _positive_step(h), dtype=np.float64)
    return np.asarray(steps, dtype=np.float64)


def _require_distinct_samples(x: RealArray, h: RealArray) -> None:
    """Reject steps that cannot form finite, distinct floating-point samples."""

    with np.errstate(over="ignore", invalid="ignore"):
        right = x + h
        left = x - h
    if (
        np.any(~np.isfinite(right))
        or np.any(~np.isfinite(left))
        or np.any(right == x)
        or np.any(left == x)
    ):
        raise ValueError("h is too small or too large to perturb x")


def _real_vector(value: object, *, name: str) -> Vector:
    """Normalize a vector argument to a private float64 copy."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real")
    if array.ndim != 1:
        raise TypeError(f"{name} must be a vector; got shape {array.shape}")
    try:
        return np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real values") from exc


def _real_output(value: object) -> RealArray:
    """Normalize a scalar or vector callback result to a private copy."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError("f must return real values")
    if array.ndim not in (0, 1):
        raise TypeError(
            "f must return a scalar or vector; "
            f"got an array with shape {array.shape}"
        )
    if array.ndim == 1 and array.size == 0:
        raise ValueError("f must return a nonempty vector")
    try:
        return np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("f must return real values") from exc


def _matching_output(value: object, shape: tuple[int, ...]) -> RealArray:
    """Normalize a callback result and require its output shape to be stable."""

    output = _real_output(value)
    if output.shape != shape:
        raise ValueError(
            "f returned inconsistent shapes: "
            f"expected {shape}, got {output.shape}"
        )
    return output


def _line_derivative(
    f: ScalarToScalarFunc | ScalarToVectorFunc,
    x: Scalar,
    h: Scalar | None,
) -> Scalar | Vector:
    """Differentiate a scalar-domain map with scalar or vector codomain."""

    x_value = _real_scalar(x, name="x")
    x_array = np.asarray(x_value, dtype=np.float64)
    h_value = _step_values(x_array, h)
    _require_distinct_samples(x_array, h_value)
    step = float(h_value)
    f_value = _real_output(f(x_value))
    f_right = _matching_output(f(x_value + step), f_value.shape)
    f_left = _matching_output(f(x_value - step), f_value.shape)
    return _A(f_right - f_value, f_value - f_left, step)


def _coordinate_derivatives(
    f: VectorToScalarFunc | VectorToVectorFunc,
    x: Vector,
    h: Scalar | None,
    *,
    output_ndim: int,
) -> RealArray:
    """Differentiate a vector-domain map along every coordinate direction."""

    x_array = _real_vector(x, name="x")
    if x_array.size == 0:
        raise ValueError("x must be a nonempty vector")

    h_values = _step_values(x_array, h)
    _require_distinct_samples(x_array, h_values)
    f_value = _real_output(f(x_array.copy()))
    if f_value.ndim != output_ndim:
        output_name = "a scalar" if output_ndim == 0 else "a vector"
        raise TypeError(f"f must return {output_name}")

    sample_shape = (x_array.size, *f_value.shape)
    right_values = np.empty(sample_shape, dtype=np.float64)
    left_values = np.empty(sample_shape, dtype=np.float64)

    for index in range(x_array.size):
        x_right = x_array.copy()
        x_left = x_array.copy()
        x_right[index] += h_values[index]
        x_left[index] -= h_values[index]
        right_values[index] = _matching_output(f(x_right), f_value.shape)
        left_values[index] = _matching_output(f(x_left), f_value.shape)

    increments_right = right_values - f_value
    increments_left = f_value - left_values
    step_shape = (x_array.size,) + (1,) * f_value.ndim
    values = np.asarray(
        _A(
            increments_right,
            increments_left,
            h_values.reshape(step_shape),
        ),
        dtype=np.float64,
    )
    return values


@overload
def derivative(
    f: ScalarToScalarFunc,
    x: Scalar,
    h: Scalar | None = None,
) -> Scalar: ...


@overload
def derivative(
    f: ScalarToVectorFunc,
    x: Scalar,
    h: Scalar | None = None,
) -> Vector: ...


def derivative(
    f: ScalarToScalarFunc | ScalarToVectorFunc,
    x: Scalar,
    h: Scalar | None = None,
) -> Scalar | Vector:
    r"""Approximate the specular derivative of a map from
    :math:`\mathbb R` to :math:`\mathbb R` or :math:`\mathbb R^m`.

    The center value is evaluated once, and the defining increment kernel
    :math:`\mathcal A` is applied directly without first forming quotients.
    """

    return _line_derivative(f, x, h)


def gradient(
    f: VectorToScalarFunc,
    x: Vector,
    h: Scalar | None = None,
) -> Vector:
    r"""Approximate the specular gradient of
    :math:`f:\mathbb R^n\to\mathbb R`.

    The result has shape ``(n,)``.
    """

    return _coordinate_derivatives(f, x, h, output_ndim=0)


def jacobian(
    f: VectorToVectorFunc,
    x: Vector,
    h: Scalar | None = None,
) -> Matrix:
    r"""Approximate the specular Jacobian of
    :math:`f:\mathbb R^n\to\mathbb R^m`.

    The result has shape ``(m, n)``.
    """

    values = _coordinate_derivatives(f, x, h, output_ndim=1)
    return np.moveaxis(values, 0, -1)
