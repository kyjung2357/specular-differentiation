"""Independent range and input-promotion checks for the public angular mean."""

from __future__ import annotations

from contextlib import nullcontext
import math

import numpy as np
import pytest

import specular


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
@pytest.mark.parametrize(
    "dtype,alpha,beta,expected",
    [(np.float16, 30000.0, 20000.0, 24000.0), (np.float32, 1e38, 5e37, 2e38 / 3)],
)
def test_scaled_mean_promotes_inputs_before_dividing(
    backend, dtype, alpha, beta, expected,
):
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is not installed")
    with specular.use_backend(backend):
        actual = specular.scaled_mean(
            np.array([alpha, -alpha], dtype=dtype),
            np.array([beta, -beta], dtype=dtype),
            sigma=0.1,
        )
    np.testing.assert_allclose(actual, [expected, -expected], rtol=2e-6)


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
@pytest.mark.parametrize(
    "alpha,beta,sigma,expected",
    [
        (1e308, 5e307, 1e-300, 6.666666666666666e307),
        (1e-300, 2e-300, 1e300, 1.5e-300),
        (1e-300, -2e-300, 1e300, -5e-301),
        (1e300, 1e-300, 1e-300, (1.0 + math.sqrt(2.0)) * 1e-300),
        (1e300, -1e-300, 1e-300, (math.sqrt(2.0) - 1.0) * 1e-300),
    ],
)
def test_scaled_mean_recovers_representable_values_after_rescaling(
    backend, alpha, beta, sigma, expected,
):
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is not installed")
    if backend == "jax":
        import jax

        precision = jax.enable_x64(True)
    else:
        precision = nullcontext()
    with specular.use_backend(backend), precision, np.errstate(all="raise"):
        actual = specular.scaled_mean(alpha, beta, sigma)
        reversed_actual = specular.scaled_mean(beta, alpha, sigma)
        negative_actual = specular.scaled_mean(-alpha, -beta, sigma)
    assert float(actual) == pytest.approx(expected, rel=3e-14, abs=0.0)
    assert float(reversed_actual) == float(actual)
    assert float(negative_actual) == -float(actual)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_cpu_scaled_mean_retains_subnormal_scale_information(backend):
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is not installed")
    with specular.use_backend(backend), np.errstate(all="raise"):
        actual = specular.scaled_mean(
            [1e308, 1e308, 1e308], [5e307, 1e-308, -1e-308], 1e-308,
        )
    np.testing.assert_allclose(
        actual,
        [6.666666666666666e307, 2.414213562373095e-308, 4.14213562373095e-309],
        rtol=3e-14,
        atol=np.nextafter(0.0, 1.0),
    )


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
def test_scaled_mean_keeps_complex_input_rejection(backend):
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is not installed")
    with specular.use_backend(backend), pytest.raises(TypeError, match="real"):
        specular.scaled_mean(1.0 + 2.0j, 2.0, sigma=0.1)


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
@pytest.mark.parametrize(
    "finite,sigma,expected",
    [(1e100, 1e-300, 2e100), (-1e300, 1e100, 5e-101), (0.0, 1e-300, 1e-300)],
)
def test_scaled_mean_with_one_infinite_slope_keeps_the_finite_scale(
    backend, finite, sigma, expected,
):
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is not installed")
    if backend == "jax":
        import jax

        precision = jax.enable_x64(True)
    else:
        precision = nullcontext()
    with specular.use_backend(backend), precision, np.errstate(all="raise"):
        actual = specular.scaled_mean(np.inf, finite, sigma)
        swapped = specular.scaled_mean(finite, np.inf, sigma)
        negative = specular.scaled_mean(-np.inf, -finite, sigma)
    assert float(actual) == pytest.approx(expected, rel=3e-14, abs=0.0)
    assert float(swapped) == float(actual)
    assert float(negative) == -float(actual)
