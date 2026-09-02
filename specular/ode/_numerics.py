"""Stable scalar numerical primitives for the ODE methods."""

from __future__ import annotations

import math


type _Dyadic = tuple[int, int]
type _Magnitude = tuple[float, int]


def _normalize_dyadic(integer: int, exponent: int) -> _Dyadic:
    """Normalize an exact dyadic number ``integer * 2**exponent``."""

    if integer == 0:
        return 0, 0

    magnitude = abs(integer)
    shift = (magnitude & -magnitude).bit_length() - 1
    return integer >> shift, exponent + shift


def _dyadic(value: float) -> _Dyadic:
    """Return the exact dyadic representation of a finite float."""

    numerator, denominator = float(value).as_integer_ratio()
    exponent = -(denominator.bit_length() - 1)
    return _normalize_dyadic(numerator, exponent)


def _dyadic_negate(value: _Dyadic) -> _Dyadic:
    """Return the exact additive inverse of a dyadic number."""

    return (-value[0], value[1]) if value[0] else value


def _dyadic_sum(*values: _Dyadic) -> _Dyadic:
    """Add dyadic numbers exactly."""

    nonzero = [value for value in values if value[0]]
    if not nonzero:
        return 0, 0

    common_exponent = min(exponent for _, exponent in nonzero)
    integer = sum(
        significand << (exponent - common_exponent)
        for significand, exponent in nonzero
    )
    return _normalize_dyadic(integer, common_exponent)


def _dyadic_product(*values: _Dyadic) -> _Dyadic:
    """Multiply dyadic numbers exactly."""

    integer = 1
    exponent = 0
    for significand, value_exponent in values:
        if significand == 0:
            return 0, 0
        integer *= significand
        exponent += value_exponent
    return _normalize_dyadic(integer, exponent)


def _magnitude(value: _Dyadic) -> _Magnitude:
    """Round a nonzero dyadic magnitude to 53 binary digits."""

    integer = abs(value[0])
    bit_count = integer.bit_length()
    exponent = value[1] + bit_count

    if bit_count <= 53:
        significand = integer << (53 - bit_count)
    else:
        shift = bit_count - 53
        significand, remainder = divmod(integer, 1 << shift)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (
            remainder == halfway and significand & 1
        ):
            significand += 1
        if significand == 1 << 53:
            significand >>= 1
            exponent += 1

    return math.ldexp(float(significand), -53), exponent


def _normalize_magnitude(
    mantissa: float,
    exponent: int,
) -> _Magnitude:
    """Normalize a positive binary magnitude."""

    normalized, shift = math.frexp(mantissa)
    return normalized, exponent + shift


def _magnitude_add(
    first: _Magnitude,
    second: _Magnitude,
) -> _Magnitude:
    """Add two positive binary magnitudes without exponent overflow."""

    if first[1] < second[1]:
        first, second = second, first
    return _normalize_magnitude(
        first[0] + math.ldexp(second[0], second[1] - first[1]),
        first[1],
    )


def _magnitude_multiply(
    first: _Magnitude,
    second: _Magnitude,
) -> _Magnitude:
    """Multiply two positive binary magnitudes."""

    return _normalize_magnitude(
        first[0] * second[0],
        first[1] + second[1],
    )


def _magnitude_divide(
    numerator: _Magnitude,
    denominator: _Magnitude,
) -> _Magnitude:
    """Divide two positive binary magnitudes."""

    return _normalize_magnitude(
        numerator[0] / denominator[0],
        numerator[1] - denominator[1],
    )


def _magnitude_sqrt(value: _Magnitude) -> _Magnitude:
    """Take the square root of a positive binary magnitude."""

    mantissa, exponent = value
    if exponent & 1:
        mantissa *= 2.0
        exponent -= 1
    return _normalize_magnitude(math.sqrt(mantissa), exponent // 2)


def _magnitude_float(value: _Magnitude) -> float:
    """Convert a binary magnitude to float, returning infinity on overflow."""

    try:
        return math.ldexp(value[0], value[1])
    except OverflowError:
        return math.inf


def _dyadic_ratio_float(
    numerator: _Dyadic,
    denominator: _Dyadic,
) -> float:
    """Round an exact dyadic ratio to float without intermediate overflow."""

    if numerator[0] == 0:
        return 0.0
    magnitude = _magnitude_divide(
        _magnitude(numerator),
        _magnitude(denominator),
    )
    return math.copysign(_magnitude_float(magnitude), numerator[0])


def _relative_dyadic_sum(*terms: _Dyadic) -> float:
    """Return an exact sum's magnitude relative to its largest term."""

    residual = _dyadic_sum(*terms)
    if residual[0] == 0:
        return 0.0
    term_magnitudes = [_magnitude(term) for term in terms if term[0]]
    scale = max(term_magnitudes, key=lambda value: (value[1], value[0]))
    return _magnitude_float(
        _magnitude_divide(_magnitude(residual), scale)
    )
