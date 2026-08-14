"""Tests for the backend-neutral calculation facade."""

from __future__ import annotations

import numpy as np
import pytest

import specular
from specular.calculation import _A, _B, _C


def _square(value):
    return value * value


def test_numpy_facade_calculates_all_map_shapes() -> None:
    specular.set_backend("numpy")

    derivative = specular.derivative(_square, 2.0)
    gradient = specular.gradient(
        lambda value: np.sum(value * value),
        np.array([1.0, 2.0, 3.0]),
    )
    jacobian = specular.jacobian(
        lambda value: np.array(
            [value[0] + 2.0 * value[1], 3.0 * value[0] - value[1]]
        ),
        np.array([1.0, 2.0]),
    )

    assert derivative == pytest.approx(4.0)
    np.testing.assert_allclose(gradient, [2.0, 4.0, 6.0])
    np.testing.assert_allclose(jacobian, [[1.0, 2.0], [3.0, -1.0]])


def test_internal_kernels_follow_the_selected_backend() -> None:
    specular.set_backend("numpy")

    assert _A(2.0, 2.0, 1.0) == pytest.approx(2.0)
    assert _B(1.0, -1.0) == pytest.approx(0.0)
    assert _C(1.0, -1.0) == pytest.approx(0.0)


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
def test_scalar_derivative_dispatches_to_each_available_backend(
    backend: str,
) -> None:
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is not installed")

    specular.set_backend(backend)
    result = specular.derivative(_square, 2.0, h=1e-3)

    assert float(result) == pytest.approx(4.0, abs=2e-3)
