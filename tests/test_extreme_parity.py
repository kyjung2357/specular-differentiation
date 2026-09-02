"""Numerical regression tests shared by calculation backends."""

from __future__ import annotations

import math

import numpy as np
import pytest

from specular.backends.numpy import calculation as numpy_backend


def _numba_backend():
    pytest.importorskip("numba")
    from specular.backends.numba import calculation

    return calculation


def _jax_backend():
    jax = pytest.importorskip("jax")
    from specular.backends.jax import calculation

    return jax, calculation


@pytest.mark.parametrize("kernel_name", ["_B", "_C"])
def test_cpu_slope_kernels_match_on_extreme_values(kernel_name: str) -> None:
    numba_backend = _numba_backend()
    maximum = np.finfo(np.float64).max
    previous = np.nextafter(maximum, 0.0)
    near_antidiagonal = np.nextafter(1.0, 0.0)
    alpha = np.array(
        [
            maximum,
            maximum,
            1.0,
            1e300,
            1e200,
            np.inf,
            -np.inf,
            np.inf,
            np.inf,
        ]
    )
    beta = np.array(
        [
            1.0,
            previous,
            -near_antidiagonal,
            -1e100,
            -np.nextafter(1e200, 0.0),
            1.0,
            1.0,
            np.inf,
            -np.inf,
        ]
    )

    expected = np.asarray(getattr(numpy_backend, kernel_name)(alpha, beta))
    actual = np.asarray(getattr(numba_backend, kernel_name)(alpha, beta))

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=8 * np.finfo(np.float64).eps,
        atol=np.finfo(np.float64).tiny,
    )
    assert expected[0] == pytest.approx(1.0 + math.sqrt(2.0))
    assert expected[1] == previous
    assert expected[2] == 2.7755575615628914e-17
    assert expected[-2] == np.inf
    assert expected[-1] == 0.0


@pytest.mark.parametrize("kernel_name", ["_B", "_C"])
def test_numpy_infinite_slope_overflow_is_quiet(kernel_name: str) -> None:
    maximum = np.finfo(np.float64).max

    with np.errstate(over="raise"):
        result = getattr(numpy_backend, kernel_name)(maximum, np.inf)

    assert result == np.inf


def test_cpu_increment_kernel_preserves_scale_invariance_at_subnormal_scale(
) -> None:
    numba_backend = _numba_backend()
    tiny = np.nextafter(0.0, 1.0)
    expected = numpy_backend._C(1.0, -2.0)

    assert numpy_backend._A(tiny, -2.0 * tiny, tiny) == expected
    assert numba_backend._A(tiny, -2.0 * tiny, tiny) == expected


def test_cpu_increment_kernel_handles_overflowed_opposite_slopes() -> None:
    numba_backend = _numba_backend()
    expected = 2.7777777777778e-310

    assert numpy_backend._A(2.0, -1.8, 1e-308) == expected
    assert numba_backend._A(2.0, -1.8, 1e-308) == expected


def test_increment_kernel_keeps_finite_same_sign_overflow_boundary() -> None:
    numba_backend = _numba_backend()
    maximum = np.finfo(np.float64).max
    minimum_normal = np.finfo(np.float64).tiny
    base = minimum_normal * maximum
    up = np.nextafter(base, np.inf)
    down = np.nextafter(base, 0.0)
    previous_maximum = np.nextafter(maximum, 0.0)

    a = np.array([up, up, base, -up, 8.0, 8.0])
    b = np.array([base, down, down, -base, minimum_normal, 0.0])
    c = np.array(
        [
            minimum_normal,
            minimum_normal,
            minimum_normal,
            minimum_normal,
            minimum_normal,
            minimum_normal,
        ]
    )
    expected = np.array(
        [
            maximum,
            maximum,
            previous_maximum,
            -maximum,
            1.0 + math.sqrt(2.0),
            1.0,
        ]
    )

    numpy_result = np.asarray(numpy_backend._A(a, b, c))
    numba_result = np.asarray(numba_backend._A(a, b, c))
    np.testing.assert_allclose(numpy_result, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(numba_result, expected, rtol=0.0, atol=0.0)

    jax, jax_backend = _jax_backend()
    with jax.enable_x64(True):
        jax_result = np.asarray(jax_backend._A(a, b, c))
    np.testing.assert_allclose(
        jax_result,
        expected,
        rtol=8 * np.finfo(np.float64).eps,
        atol=np.finfo(np.float64).tiny,
    )


@pytest.mark.parametrize("kernel_name", ["_B", "_C"])
def test_jax_x64_slope_kernels_match_numpy_extremes(
    kernel_name: str,
) -> None:
    jax, jax_backend = _jax_backend()
    maximum = np.finfo(np.float64).max
    alpha = np.array(
        [maximum, maximum, 1.0, 1e300, np.inf, -np.inf, np.inf]
    )
    beta = np.array(
        [
            1.0,
            maximum / 2.0,
            -np.nextafter(1.0, 0.0),
            -1e100,
            1.0,
            1.0,
            -np.inf,
        ]
    )
    expected = np.asarray(getattr(numpy_backend, kernel_name)(alpha, beta))

    with jax.enable_x64(True):
        actual = np.asarray(getattr(jax_backend, kernel_name)(alpha, beta))

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=8 * np.finfo(np.float64).eps,
        atol=np.finfo(np.float64).tiny,
    )


def test_jax_x64_increment_kernel_matches_numpy_at_normal_scales() -> None:
    jax, jax_backend = _jax_backend()
    minimum_normal = np.finfo(np.float64).tiny
    a = np.array(
        [minimum_normal, 1e200, 1e200, np.finfo(np.float64).max]
    )
    b = np.array(
        [2.0 * minimum_normal, 1e-200, -1e-200, 1.0]
    )
    c = np.array([minimum_normal, 1e-200, 1e-200, 1.0])
    expected = np.asarray(numpy_backend._A(a, b, c))

    with jax.enable_x64(True):
        actual = np.asarray(jax_backend._A(a, b, c))

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=8 * np.finfo(np.float64).eps,
        atol=np.finfo(np.float64).tiny,
    )


def test_all_backends_preserve_broadcast_shape() -> None:
    alpha = np.array([[1.0], [-2.0], [3.0]])
    beta = np.array([[0.5, -1.0, 2.0, 4.0]])
    expected = np.asarray(numpy_backend._C(alpha, beta))
    assert expected.shape == (3, 4)

    numba_backend = _numba_backend()
    np.testing.assert_array_equal(numba_backend._C(alpha, beta), expected)

    jax, jax_backend = _jax_backend()
    with jax.enable_x64(True):
        actual = np.asarray(jax_backend._C(alpha, beta))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=8 * np.finfo(np.float64).eps,
        atol=np.finfo(np.float64).tiny,
    )
