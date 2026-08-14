"""Internal mathematical primitives for the Numba backend."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from types import BuiltinFunctionType, FunctionType
from typing import Any, overload

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.core.errors import NumbaError
from numba.core.registry import CPUDispatcher

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

_ANGLE_LIMIT = float(np.nextafter(np.pi / 2.0, 0.0))


@njit(cache=True, error_model="numpy")
def _one_infinite_scalar(x: float, sign: float) -> float:
    """Evaluate C(x, sign * infinity) without cancellation."""

    radius = math.hypot(1.0, x)
    if sign * x >= 0.0:
        return x + sign * radius
    return (sign / radius) / (1.0 - sign * x / radius)


@njit(cache=True, error_model="numpy")
def _divide_by_product_scalar(
    numerator: float,
    factor_a: float,
    factor_b: float,
    denominator: float,
) -> float:
    """Evaluate a product quotient by separating the smaller exponent."""

    factor_high = max(factor_a, factor_b)
    factor_low = min(factor_a, factor_b)
    factor_low_mantissa, factor_low_exponent = math.frexp(factor_low)
    return math.ldexp(
        (numerator / factor_high) / (factor_low_mantissa * denominator),
        -factor_low_exponent,
    )


@njit(cache=True, error_model="numpy")
def _finite_C_scalar(alpha: float, beta: float) -> float:
    """Evaluate C for finite, nonexceptional scalar inputs."""

    same_sign = math.copysign(1.0, alpha) == math.copysign(1.0, beta)
    radius_a = math.hypot(1.0, alpha)
    radius_b = math.hypot(1.0, beta)

    if same_sign:
        # C is a convex combination on each same-sign quadrant.  Writing it
        # this way keeps the result between alpha and beta even when both are
        # so close to max_float that the normalized quotient rounds upward to
        # infinity.
        if radius_a >= radius_b:
            ratio = radius_b / radius_a
            inverse_weight_sum = 1.0 / (1.0 + ratio)
            return (
                alpha * (ratio * inverse_weight_sum)
                + beta * inverse_weight_sum
            )
        ratio = radius_a / radius_b
        inverse_weight_sum = 1.0 / (1.0 + ratio)
        return (
            alpha * inverse_weight_sum
            + beta * (ratio * inverse_weight_sum)
        )

    unit_a = alpha / radius_a
    unit_b = beta / radius_b
    inverse_a = 1.0 / radius_a
    inverse_b = 1.0 / radius_b
    denominator = 1.0 + inverse_a * inverse_b - unit_a * unit_b
    return _divide_by_product_scalar(
        alpha + beta,
        radius_a,
        radius_b,
        denominator,
    )


@njit(cache=True, error_model="numpy")
def _A_scalar(a: float, b: float, c: float) -> float:
    """Compiled scalar implementation of the defining secant kernel."""

    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return math.nan
    if c <= 0.0:
        return math.nan
    if a == b:
        return a / c
    if a == -b:
        return 0.0

    radius_a = math.hypot(a, c)
    radius_b = math.hypot(b, c)
    same_sign = math.copysign(1.0, a) == math.copysign(1.0, b)

    if same_sign:
        slope_a = a / c
        slope_b = b / c
        if math.isfinite(slope_a) and math.isfinite(slope_b):
            return _finite_C_scalar(slope_a, slope_b)
        return (a / radius_a + b / radius_b) / (
            c / radius_a + c / radius_b
        )

    unit_a = a / radius_a
    unit_b = b / radius_b
    inverse_a = c / radius_a
    inverse_b = c / radius_b
    denominator = 1.0 + inverse_a * inverse_b - unit_a * unit_b
    radius_high = max(radius_a, radius_b)
    radius_low = min(radius_a, radius_b)
    return ((a + b) / radius_high) * (c / radius_low) / denominator


@njit(cache=True, error_model="numpy")
def _B_scalar(alpha: float, beta: float) -> float:
    """Compiled scalar implementation of the angular slope mean."""

    if math.isnan(alpha) or math.isnan(beta):
        return math.nan
    if alpha == beta:
        return alpha
    if alpha == -beta:
        return 0.0

    alpha_infinite = math.isinf(alpha)
    beta_infinite = math.isinf(beta)
    if alpha_infinite:
        return _one_infinite_scalar(beta, math.copysign(1.0, alpha))
    if beta_infinite:
        return _one_infinite_scalar(alpha, math.copysign(1.0, beta))

    angle = 0.5 * (math.atan(alpha) + math.atan(beta))
    if abs(angle) >= _ANGLE_LIMIT:
        return _finite_C_scalar(alpha, beta)
    return math.tan(angle)


@njit(cache=True, error_model="numpy")
def _C_scalar(alpha: float, beta: float) -> float:
    """Compiled scalar implementation of the algebraic slope mean."""

    if math.isnan(alpha) or math.isnan(beta):
        return math.nan
    if alpha == beta:
        return alpha
    if alpha == -beta:
        return 0.0

    alpha_infinite = math.isinf(alpha)
    beta_infinite = math.isinf(beta)
    if alpha_infinite:
        return _one_infinite_scalar(beta, math.copysign(1.0, alpha))
    if beta_infinite:
        return _one_infinite_scalar(alpha, math.copysign(1.0, beta))
    return _finite_C_scalar(alpha, beta)


@njit(cache=True)
def _A_flat(a: RealArray, b: RealArray, c: RealArray) -> RealArray:
    result = np.empty(a.size, dtype=np.float64)
    for index in range(a.size):
        result[index] = _A_scalar(a[index], b[index], c[index])
    return result


@njit(cache=True)
def _B_flat(alpha: RealArray, beta: RealArray) -> RealArray:
    result = np.empty(alpha.size, dtype=np.float64)
    for index in range(alpha.size):
        result[index] = _B_scalar(alpha[index], beta[index])
    return result


@njit(cache=True)
def _C_flat(alpha: RealArray, beta: RealArray) -> RealArray:
    result = np.empty(alpha.size, dtype=np.float64)
    for index in range(alpha.size):
        result[index] = _C_scalar(alpha[index], beta[index])
    return result


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


def _A(a: RealInput, b: RealInput, c: RealInput) -> RealResult:
    r"""Evaluate :math:`\mathcal A(a,b,c)` elementwise with compiled kernels.

    Entries outside :math:`\mathbb R^2\times(0,\infty)` produce ``NaN``.
    """

    a_array, b_array, c_array = _broadcast_real(a, b, c)
    shape = a_array.shape
    result = _A_flat(
        np.ravel(a_array).copy(),
        np.ravel(b_array).copy(),
        np.ravel(c_array).copy(),
    )
    return _finish(result.reshape(shape))


def _B(alpha: RealInput, beta: RealInput) -> RealResult:
    r"""Evaluate the angular mean :math:`\mathcal B` elementwise."""

    alpha_array, beta_array = _broadcast_real(alpha, beta)
    shape = alpha_array.shape
    result = _B_flat(
        np.ravel(alpha_array).copy(),
        np.ravel(beta_array).copy(),
    )
    return _finish(result.reshape(shape))


def _C(alpha: RealInput, beta: RealInput) -> RealResult:
    r"""Evaluate the stable algebraic mean :math:`\mathcal C` elementwise."""

    alpha_array, beta_array = _broadcast_real(alpha, beta)
    shape = alpha_array.shape
    result = _C_flat(
        np.ravel(alpha_array).copy(),
        np.ravel(beta_array).copy(),
    )
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


@lru_cache(maxsize=128)
def _compile_python_callback(f: Callable[..., Any]) -> CPUDispatcher:
    """Compile and retain a bounded set of ordinary callback functions."""

    try:
        return njit(f)
    except TypeError as exc:
        # Numba does not decorate builtins and NumPy ufuncs directly, even
        # though calls to them are valid nopython operations.  A one-argument
        # closure gives these callbacks the same contract as Python functions.
        def wrapped(value: Any) -> Any:
            return f(value)

        try:
            return njit(wrapped)
        except TypeError:
            raise TypeError("f must be a Numba-compilable callable") from exc


def _compile_callback(f: Callable[..., Any]) -> CPUDispatcher:
    """Return a reusable nopython dispatcher for a supported callback."""

    if isinstance(f, CPUDispatcher):
        return f
    if not isinstance(f, (FunctionType, BuiltinFunctionType, np.ufunc)):
        raise TypeError(
            "f must be a Python function, builtin, NumPy ufunc, "
            "or Numba CPUDispatcher"
        )
    return _compile_python_callback(f)


def _evaluate_center(
    compiled_f: CPUDispatcher,
    argument: float | RealArray,
) -> RealArray:
    """Compile/evaluate a callback once and require nopython execution."""

    try:
        value = compiled_f(argument)
    except NumbaError as exc:
        raise TypeError("f must be Numba-compilable in nopython mode") from exc
    if not compiled_f.nopython_signatures:
        raise TypeError("f must be Numba-compilable in nopython mode")
    return _real_output(value)


def _invalid_h(h: float) -> bool:
    return not np.isfinite(h) or h <= 0.0


@njit
def _line_scalar_loop(
    f: Callable[[float], Any],
    x: float,
    h: float,
    center: float,
) -> float:
    right = np.asarray(f(x + h)).item()
    left = np.asarray(f(x - h)).item()
    return _A_scalar(right - center, center - left, h)


@njit
def _line_vector_loop(
    f: Callable[[float], RealArray],
    x: float,
    h: float,
    center: RealArray,
) -> RealArray:
    right = np.asarray(f(x + h)).copy()
    left = np.asarray(f(x - h)).copy()
    if right.shape != center.shape or left.shape != center.shape:
        raise ValueError("f returned inconsistent shapes")

    result = np.empty(center.size, dtype=np.float64)
    for index in range(center.size):
        result[index] = _A_scalar(
            right[index] - center[index],
            center[index] - left[index],
            h,
        )
    return result


@njit
def _coordinate_scalar_loop(
    f: Callable[[RealArray], Any],
    x: RealArray,
    h: float,
    center: float,
) -> RealArray:
    result = np.empty(x.size, dtype=np.float64)
    for coordinate in range(x.size):
        x_right = x.copy()
        x_left = x.copy()
        x_right[coordinate] += h
        x_left[coordinate] -= h
        right = np.asarray(f(x_right)).item()
        left = np.asarray(f(x_left)).item()
        result[coordinate] = _A_scalar(
            right - center,
            center - left,
            h,
        )
    return result


@njit
def _coordinate_vector_loop(
    f: Callable[[RealArray], RealArray],
    x: RealArray,
    h: float,
    center: RealArray,
) -> RealArray:
    result = np.empty((center.size, x.size), dtype=np.float64)
    for coordinate in range(x.size):
        x_right = x.copy()
        x_left = x.copy()
        x_right[coordinate] += h
        x_left[coordinate] -= h
        right = np.asarray(f(x_right)).copy()
        left = np.asarray(f(x_left)).copy()
        if right.shape != center.shape or left.shape != center.shape:
            raise ValueError("f returned inconsistent shapes")

        for output in range(center.size):
            result[output, coordinate] = _A_scalar(
                right[output] - center[output],
                center[output] - left[output],
                h,
            )
    return result


@overload
def derivative(
    f: ScalarToScalarFunc,
    x: Scalar,
    h: Scalar = 1e-6,
) -> Scalar: ...


@overload
def derivative(
    f: ScalarToVectorFunc,
    x: Scalar,
    h: Scalar = 1e-6,
) -> Vector: ...


def derivative(
    f: ScalarToScalarFunc | ScalarToVectorFunc,
    x: Scalar,
    h: Scalar = 1e-6,
) -> Scalar | Vector:
    r"""Approximate the Numba-compiled specular derivative on :math:`\mathbb R`."""

    x_value = _real_scalar(x, name="x")
    h_value = _real_scalar(h, name="h")
    compiled_f = _compile_callback(f)
    center = _evaluate_center(compiled_f, x_value)

    if center.ndim == 0:
        if _invalid_h(h_value):
            return math.nan
        return _line_scalar_loop(compiled_f, x_value, h_value, center.item())

    if _invalid_h(h_value):
        return np.full(center.shape, np.nan, dtype=np.float64)
    return _line_vector_loop(compiled_f, x_value, h_value, center)


def gradient(
    f: VectorToScalarFunc,
    x: Vector,
    h: Scalar = 1e-6,
) -> Vector:
    r"""Approximate the Numba-compiled gradient; result shape is ``(n,)``."""

    x_array = _real_vector(x, name="x")
    if x_array.size == 0:
        raise ValueError("x must be a nonempty vector")
    h_value = _real_scalar(h, name="h")
    compiled_f = _compile_callback(f)
    center = _evaluate_center(compiled_f, x_array.copy())
    if center.ndim != 0:
        raise TypeError("f must return a scalar")

    if _invalid_h(h_value):
        return np.full(x_array.shape, np.nan, dtype=np.float64)
    return _coordinate_scalar_loop(
        compiled_f,
        x_array,
        h_value,
        center.item(),
    )


def jacobian(
    f: VectorToVectorFunc,
    x: Vector,
    h: Scalar = 1e-6,
) -> Matrix:
    r"""Approximate the Numba-compiled Jacobian; result shape is ``(m, n)``."""

    x_array = _real_vector(x, name="x")
    if x_array.size == 0:
        raise ValueError("x must be a nonempty vector")
    h_value = _real_scalar(h, name="h")
    compiled_f = _compile_callback(f)
    center = _evaluate_center(compiled_f, x_array.copy())
    if center.ndim != 1:
        raise TypeError("f must return a vector")

    if _invalid_h(h_value):
        return np.full((center.size, x_array.size), np.nan, dtype=np.float64)
    return _coordinate_vector_loop(compiled_f, x_array, h_value, center)
