"""Behavioral tests for independent optimization directions."""

from __future__ import annotations

import numpy as np
import pytest

import specular
from specular.optimization.direction import _as_point, _norm, make_direction


def _square(x):
    return x * x


def _quadratic(x):
    return (x * x).sum()


@pytest.mark.parametrize("method", ["speg", "specular_gradient"])
@pytest.mark.parametrize(
    "gradient, expected", [(2.0, -1.0), (-4.0, 1.0), (0.0, 0.0)]
)
def test_scalar_direction_has_negative_gradient_sign(method, gradient, expected):
    direction = make_direction(method, gradient=lambda x: gradient)
    assert direction(1, 3.0) == expected


@pytest.mark.parametrize(
    "magnitude", [np.finfo(float).max, 1e-300, np.nextafter(0.0, 1.0)]
)
def test_direction_normalizes_extreme_gradients(magnitude):
    direction = make_direction(gradient=lambda x: [magnitude, -magnitude])
    with np.errstate(all="raise"):
        result = direction(1, [1.0, 2.0])
    np.testing.assert_allclose(result, [-np.sqrt(0.5), np.sqrt(0.5)], rtol=1e-15)
    assert _norm(result) == pytest.approx(1.0)


def test_direction_retains_representable_components_across_scales():
    direction = make_direction(gradient=lambda x: [1e308, 1.0])
    with np.errstate(all="raise"):
        result = direction(1, [0.0, 0.0])
    assert result[0] == -1.0
    assert result[1] == pytest.approx(-1e-308, rel=1e-15, abs=0.0)


def test_vector_zero_and_near_zero_are_distinct():
    direction = make_direction(gradient=lambda x: [0.0, 0.0])
    np.testing.assert_array_equal(direction(1, [1.0, 2.0]), [0.0, 0.0])
    np.testing.assert_array_equal(
        direction(1, [1.0, 2.0], gradient_value=[1e-300, 0.0]), [-1.0, 0.0]
    )


def test_cached_gradient_bypasses_gradient_callback():
    def fail(x):
        pytest.fail("the cached gradient must avoid another callback evaluation")

    direction = make_direction(gradient=fail)
    np.testing.assert_array_equal(
        direction(7, [1.0, 2.0], gradient_value=[0.0, -3.0]), [0.0, 1.0]
    )


@pytest.mark.parametrize(
    "point, gradient", [(1.0, [1.0]), ([1.0], 1.0), ([1.0, 2.0], [1.0])]
)
def test_direction_rejects_gradient_shape_broadcasting(point, gradient):
    direction = make_direction(gradient=lambda x: gradient)
    with pytest.raises(ValueError, match="same shape"):
        direction(1, point)


@pytest.mark.parametrize(
    "value", [True, [True], [1.0, True], 1j, [1.0, 1j], "1", ["1"]]
)
def test_direction_rejects_nonreal_points_and_gradients(value):
    direction = make_direction(gradient=lambda x: value)
    with pytest.raises(TypeError, match="real"):
        direction(1, value)
    with pytest.raises(TypeError, match="real"):
        direction(1, 1.0)


@pytest.mark.parametrize("value", [np.nan, np.inf, [1.0, np.nan], [], [[1.0]]])
def test_direction_rejects_invalid_points_before_callback(value):
    def fail(x):
        pytest.fail("invalid points must be rejected before callback evaluation")

    with pytest.raises(ValueError):
        make_direction(gradient=fail)(1, value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_direction_rejects_nonfinite_gradients(value):
    with pytest.raises(ValueError, match="gradient must be finite"):
        make_direction(gradient=lambda x: value)(1, 1.0)


@pytest.mark.parametrize(
    "iteration, error",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.0, TypeError)],
)
def test_direction_requires_a_positive_integer_iteration(iteration, error):
    with pytest.raises(error, match="n must"):
        make_direction(gradient=lambda x: x)(iteration, 1.0)


@pytest.mark.parametrize("h", [0.0, -1.0, np.nan, np.inf])
def test_direction_rejects_invalid_differentiation_interval(h):
    with pytest.raises(ValueError, match="h must"):
        make_direction(f=_square, h=h)


@pytest.mark.parametrize("h", [True, 1j, [1.0]])
def test_direction_rejects_nonreal_or_nonscalar_interval(h):
    with pytest.raises(TypeError, match="h must"):
        make_direction(f=_square, h=h)


def test_custom_direction_has_private_input_and_output_and_is_not_normalized():
    point = np.array([1.0, 2.0])
    output = np.array([3.0, 4.0])

    def custom(n, x):
        assert n == 3
        x[:] = 99.0
        return output

    result = make_direction(custom)(3, point)
    np.testing.assert_array_equal(point, [1.0, 2.0])
    np.testing.assert_array_equal(result, output)
    result[:] = 0.0
    np.testing.assert_array_equal(output, [3.0, 4.0])


def test_gradient_callback_cannot_mutate_the_callers_point():
    point = np.array([1.0, 2.0])

    def gradient(x):
        x[:] = [0.0, 3.0]
        return x

    np.testing.assert_array_equal(
        make_direction(gradient=gradient)(1, point), [0.0, -1.0]
    )
    np.testing.assert_array_equal(point, [1.0, 2.0])


@pytest.mark.parametrize(
    "output, error",
    [(np.inf, ValueError), ([1.0], ValueError), (True, TypeError)],
)
def test_custom_direction_output_is_validated(output, error):
    with pytest.raises(error):
        make_direction(lambda n, x: output)(1, 1.0)


def test_direction_factory_validates_configuration():
    with pytest.raises(ValueError, match="requires f"):
        make_direction()
    with pytest.raises(ValueError, match="unknown direction"):
        make_direction("bfgs", f=_square)
    with pytest.raises(TypeError, match="method must"):
        make_direction(42, f=_square)
    with pytest.raises(TypeError, match="f must"):
        make_direction(f=42)
    with pytest.raises(TypeError, match="gradient must"):
        make_direction(gradient=42)


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
@pytest.mark.parametrize("h", [None, 1e-3])
def test_numerical_direction_uses_selected_backend(backend, h):
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is not installed")
    scalar_direction = make_direction(f=_square, h=h)
    vector_direction = make_direction(f=_quadratic, h=h)
    with specular.use_backend(backend):
        assert scalar_direction(1, 2.0) == -1.0
        np.testing.assert_allclose(
            vector_direction(1, np.array([3.0, 4.0])), [-0.6, -0.8], atol=2e-4
        )


def test_point_helper_copies_and_norm_handles_extremes():
    point = np.array([3.0, 4.0])
    result = _as_point(point)
    result[:] = 0.0
    np.testing.assert_array_equal(point, [3.0, 4.0])
    assert _norm(np.array([1e300, 1e300])) == pytest.approx(np.sqrt(2.0) * 1e300)
    assert _norm(np.array([1e-300, 1e-300])) == pytest.approx(
        np.sqrt(2.0) * 1e-300, rel=1e-15, abs=0.0
    )
