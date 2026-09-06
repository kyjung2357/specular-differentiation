"""JAX transformation and bounded-memory calculation regressions."""

from __future__ import annotations

import numpy as np
import pytest

import specular

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from specular.backends.jax import calculation as backend


@pytest.fixture(autouse=True)
def _jax_x64_backend():
    with jax.enable_x64(True), specular.use_backend("jax"):
        yield


@pytest.mark.parametrize("point", [(0., 0.), (1., 1.), (1., -1.), (-2., 3.)])
def test_slope_kernel_has_angular_differential_at_equalities(point) -> None:
    reference = lambda a, b: jnp.tan((jnp.arctan(a) + jnp.arctan(b)) / 2)
    gradient = jax.grad(backend._C, argnums=(0, 1))
    expected = jax.grad(reference, argnums=(0, 1))(*point)

    np.testing.assert_allclose(gradient(*point), expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        jax.jit(gradient)(*point), expected, rtol=2e-14, atol=2e-14
    )
    actual_jvp = jax.jvp(backend._C, point, (2., -3.))[1]
    expected_jvp = jax.jvp(reference, point, (2., -3.))[1]
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("point", [(0., 0., 2.), (1., 1., 2.), (1., -1., 2.)])
def test_increment_kernel_has_the_full_scale_differential(point) -> None:
    reference = lambda a, b, c: jnp.tan(
        (jnp.arctan(a / c) + jnp.arctan(b / c)) / 2
    )
    actual = jax.jit(jax.grad(backend._A, argnums=(0, 1, 2)))(*point)
    expected = jax.grad(reference, argnums=(0, 1, 2))(*point)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    actual_hessian = jax.hessian(lambda values: backend._A(*values))(
        jnp.asarray(point)
    )
    expected_hessian = jax.hessian(lambda values: reference(*values))(
        jnp.asarray(point)
    )
    np.testing.assert_allclose(
        actual_hessian, expected_hessian, rtol=2e-14, atol=2e-14
    )


@pytest.mark.parametrize("point", [(0., 0.), (2., 2.), (2., -2.), (-2., 3.)])
def test_public_scaled_mean_supports_first_and_second_derivatives(point) -> None:
    sigma = 2.
    actual = lambda pair: specular.scaled_mean(pair[0], pair[1], sigma=sigma)
    reference = lambda pair: sigma * jnp.tan(
        (jnp.arctan(pair[0] / sigma) + jnp.arctan(pair[1] / sigma)) / 2
    )
    value = jnp.asarray(point)
    np.testing.assert_allclose(
        jax.jit(jax.grad(actual))(value), jax.grad(reference)(value),
        rtol=2e-14, atol=2e-14,
    )
    np.testing.assert_allclose(
        jax.jacfwd(jax.grad(actual))(value),
        jax.jacfwd(jax.grad(reference))(value), rtol=2e-14, atol=2e-14,
    )


def test_scaled_mean_broadcast_gradient_reduces_to_input_shape() -> None:
    beta = jnp.array([1., -1., 2.])
    reference = lambda a: jnp.sum(
        2 * jnp.tan((jnp.arctan(a / 2) + jnp.arctan(beta / 2)) / 2)
    )
    actual = lambda a: jnp.sum(specular.scaled_mean(a, beta, sigma=2.))
    np.testing.assert_allclose(jax.grad(actual)(1.), jax.grad(reference)(1.))


def test_derivative_of_quadratic_derivative_is_correct_at_origin() -> None:
    h = 1e-3
    numerical_derivative = lambda x: specular.derivative(lambda v: v*v, x, h=h)
    expected = 2 / (1 + h*h)
    assert float(jax.grad(numerical_derivative)(0.)) == pytest.approx(expected)
    assert float(jax.jit(jax.grad(numerical_derivative))(0.)) == pytest.approx(
        expected
    )


def test_gradient_remains_differentiable_through_coordinate_batches() -> None:
    h = 1e-3
    gradient = lambda x: specular.gradient(lambda v: jnp.sum(v*v), x, h=h)
    actual = jax.jit(jax.jacfwd(gradient))(jnp.zeros(3))
    np.testing.assert_allclose(actual, np.eye(3) * (2 / (1 + h*h)))


def test_extended_real_slope_has_explicitly_undefined_differential() -> None:
    assert jnp.isnan(jax.grad(lambda a: backend._C(a, 1.))(jnp.inf))
    assert jnp.isnan(jax.jvp(lambda a: backend._C(a, 1.), (jnp.inf,), (1.,))[1])


def test_finite_samples_do_not_overflow_before_increment_kernel() -> None:
    f = lambda x: 1e308 * (2*x*x - 1)
    scalar = jax.jit(lambda x: specular.derivative(f, x, h=1.))
    assert float(scalar(0.)) == 0.
    np.testing.assert_array_equal(
        specular.gradient(lambda x: f(x[0]), jnp.zeros(1), h=1.), [0.]
    )
    np.testing.assert_array_equal(
        specular.jacobian(lambda x: jnp.array([f(x[0])]), jnp.zeros(1), h=1.),
        [[0.]],
    )


def test_large_gradient_uses_bounded_coordinate_batches(monkeypatch) -> None:
    def no_dense_eye(*args, **kwargs):
        raise AssertionError("a gradient must not allocate a dense identity matrix")

    monkeypatch.setattr(jnp, "eye", no_dense_eye)
    gradient = jax.jit(
        lambda x: specular.gradient(lambda v: jnp.sum(v), x, h=0.125)
    )
    np.testing.assert_array_equal(gradient(jnp.zeros(2048)), np.ones(2048))
