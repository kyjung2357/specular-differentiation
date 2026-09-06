"""Regression tests for overflow between finite function samples."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

import specular


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    with specular.use_backend(request.param):
        yield request.param


def _large_scalar(value):
    return 1e308 * (2.0 * value * value - 1.0)


def _large_vector(value):
    return np.array([1e308 * (2.0 * value * value - 1.0), 2.0 * value])


def _large_field(value):
    return 1e308 * (2.0 * np.sum(value * value) - 1.0)


def _large_vector_field(value):
    return np.array(
        [
            1e308 * (2.0 * np.sum(value * value) - 1.0),
            2.0 * value[0] - 3.0 * value[1],
        ]
    )


@pytest.mark.parametrize(
    "operator, callback, point, expected",
    [
        ("derivative", _large_scalar, 0.0, 0.0),
        ("derivative", _large_vector, 0.0, [0.0, 2.0]),
        ("gradient", _large_field, [0.0, 0.0], [0.0, 0.0]),
        (
            "jacobian",
            _large_vector_field,
            [0.0, 0.0],
            [[0.0, 0.0], [2.0, -3.0]],
        ),
    ],
)
def test_finite_samples_preserve_symmetric_cancellation_in_all_map_shapes(
    backend, operator, callback, point, expected
):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = getattr(specular, operator)(callback, point, h=1.0)
    np.testing.assert_array_equal(actual, expected)


def _decimal_sample_mean(right, center, left, step):
    """Independent high-range reference from the exact binary sample values."""

    with localcontext() as context:
        context.prec = 100
        right, center, left, step = map(
            Decimal.from_float, (right, center, left, step)
        )
        a = (right - center) / step
        b = (center - left) / step
        radius_a = (1 + a * a).sqrt()
        radius_b = (1 + b * b).sqrt()
        if a * b < 0:
            value = (a + b) / (radius_a * radius_b + 1 - a * b)
        else:
            value = (a * radius_b + b * radius_a) / (radius_a + radius_b)
        return float(value)


@pytest.mark.parametrize(
    "samples",
    [
        (1.6e308, -1.2e308, 0.8e308),
        (1.6e308, -1.2e308, -1.6e308),
        (-1.6e308, 1.2e308, 1.6e308),
        (1.6e308, -1.2e308, -1.2e308),
        (-1.6e308, 1.2e308, 1.2e308),
    ],
)
@pytest.mark.parametrize("step", [1.0, 1e308, np.finfo(np.float64).smallest_subnormal])
def test_asymmetric_finite_sample_overflow_matches_high_range_reference(
    backend, samples, step
):
    right, center, left = samples

    def callback(value):
        if value > 0.0:
            return right
        if value < 0.0:
            return left
        return center

    expected = _decimal_sample_mean(right, center, left, float(step))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = specular.derivative(callback, 0.0, h=step)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=8 * np.finfo(np.float64).eps,
        atol=2 * np.finfo(np.float64).smallest_subnormal,
    )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("sample_position", [-1.0, 0.0, 1.0])
def test_nonfinite_original_sample_remains_nan(backend, bad_value, sample_position):
    def callback(value):
        if value == sample_position:
            return bad_value
        return 1e308 if value != 0.0 else -1e308

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = specular.derivative(callback, 0.0, h=1.0)
    assert np.isnan(actual)
