"""Shared scale-safe evaluation for the float64 calculation backends."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


type FloatArray = npt.NDArray[np.float64]


def _radius_parts(
    value: FloatArray, sigma: float,
) -> tuple[FloatArray, npt.NDArray[np.int32], FloatArray, FloatArray]:
    """Represent hypot(value, sigma) without overflowing its magnitude."""

    scale = np.maximum(np.abs(value), sigma)
    unit = value / scale
    inverse = sigma / scale
    radius = np.hypot(unit, inverse)
    mantissa, exponent = np.frexp(scale)
    mantissa, adjustment = np.frexp(mantissa * radius)
    return mantissa, exponent + adjustment, unit / radius, inverse / radius


def _finite_scaled_mean(
    alpha: FloatArray, beta: FloatArray, sigma: float,
) -> FloatArray:
    """Evaluate finite inputs without forming either input divided by sigma."""

    ma, ea, ua, ia = _radius_parts(alpha, sigma)
    mb, eb, ub, ib = _radius_parts(beta, sigma)
    result = np.empty_like(alpha)
    same_sign = np.signbit(alpha) == np.signbit(beta)
    if np.any(same_sign):
        a_high = (ea > eb) | ((ea == eb) & (ma >= mb))
        high_m = np.where(a_high, ma, mb)[same_sign]
        low_m = np.where(a_high, mb, ma)[same_sign]
        high_e = np.where(a_high, ea, eb)[same_sign]
        low_e = np.where(a_high, eb, ea)[same_sign]
        ratio = np.ldexp(low_m / high_m, low_e - high_e)
        high = np.where(a_high, alpha, beta)[same_sign]
        low = np.where(a_high, beta, alpha)[same_sign]
        difference, exponent = np.frexp(high - low)
        correction = np.ldexp(
            difference * (low_m / high_m) / (1.0 + ratio),
            exponent + low_e - high_e,
        )
        result[same_sign] = np.clip(
            low + correction, np.minimum(high, low), np.maximum(high, low),
        )

    opposite_sign = ~same_sign
    if np.any(opposite_sign):
        # Rationalization avoids cancellation between the two weighted slopes.
        # Carry the powers of two separately so a tiny intermediate mean can
        # still be rescaled to a representable final value.
        numerator, exponent = np.frexp((alpha + beta)[opposite_sign])
        scale_m, scale_e = np.frexp(sigma)
        denominator = (
            1.0 + ia[opposite_sign] * ib[opposite_sign]
            - ua[opposite_sign] * ub[opposite_sign]
        )
        mantissa = (numerator * scale_m * scale_m) / (
            ma[opposite_sign] * mb[opposite_sign] * denominator
        )
        result[opposite_sign] = np.ldexp(
            mantissa,
            exponent + 2 * scale_e - ea[opposite_sign] - eb[opposite_sign],
        )
    return result


def _one_infinite_scaled(
    value: FloatArray, sign: FloatArray, sigma: float,
) -> FloatArray:
    """Retain the finite slope when the other input is a genuine infinity."""

    result = np.empty_like(value)
    direct = sign * value >= 0.0
    result[direct] = value[direct] + sign[direct] * np.hypot(value[direct], sigma)
    if np.any(~direct):
        mantissa, exponent, unit, _ = _radius_parts(value[~direct], sigma)
        scale_m, scale_e = np.frexp(sigma)
        result[~direct] = sign[~direct] * np.ldexp(
            scale_m * scale_m / (mantissa * (1.0 + np.abs(unit))),
            2 * scale_e - exponent,
        )
    return result


def scaled_mean_float64(
    alpha: object,
    beta: object,
    sigma: float,
    kernel: Callable[..., object],
) -> float | FloatArray:
    """Use the selected slope kernel, recovering under/overflowed rescaling."""

    arrays = tuple(np.asarray(value) for value in (alpha, beta))
    if any(np.iscomplexobj(value) for value in arrays):
        raise TypeError("the specular kernels accept real inputs only")
    alpha_array, beta_array = (
        np.asarray(value, dtype=np.float64)
        for value in np.broadcast_arrays(*arrays)
    )
    shape = alpha_array.shape
    a = alpha_array.reshape(-1)
    b = beta_array.reshape(-1)
    with np.errstate(all="ignore"):
        scaled_a, scaled_b = a / sigma, b / sigma
        mean = np.asarray(kernel(scaled_a, scaled_b), dtype=np.float64)
        result = sigma * mean
        tiny = np.finfo(np.float64).tiny
        finite = np.isfinite(a) & np.isfinite(b)
        needs_recovery = finite & (
            ~np.isfinite(scaled_a) | ~np.isfinite(scaled_b)
            | ((a != 0.0) & (np.abs(scaled_a) < tiny))
            | ((b != 0.0) & (np.abs(scaled_b) < tiny))
            | ~np.isfinite(result) | (np.abs(mean) < tiny)
        )
        if np.any(needs_recovery):
            result[needs_recovery] = _finite_scaled_mean(
                a[needs_recovery], b[needs_recovery], sigma,
            )
        a_infinite = np.isinf(a) & np.isfinite(b)
        b_infinite = np.isfinite(a) & np.isinf(b)
        result[a_infinite] = _one_infinite_scaled(
            b[a_infinite], np.sign(a[a_infinite]), sigma,
        )
        result[b_infinite] = _one_infinite_scaled(
            a[b_infinite], np.sign(b[b_infinite]), sigma,
        )
        diagonal = a == b
        antidiagonal = ~diagonal & (a == -b)
        result = np.where(diagonal, a, result)
        result = np.where(antidiagonal, 0.0, result)
    result = result.reshape(shape)
    return float(result) if result.ndim == 0 else result
