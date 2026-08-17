"""Tests for the backend-neutral calculation facade."""

from __future__ import annotations

import numpy as np
import pytest

import specular


@pytest.fixture(autouse=True)
def _use_numpy_backend():
    specular.set_backend("numpy")
    yield
    specular.set_backend("numpy")


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


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
def test_scalar_derivative_dispatches_to_each_available_backend(
    backend: str,
) -> None:
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is not installed")

    specular.set_backend(backend)
    result = specular.derivative(_square, 2.0, h=1e-3)

    assert float(result) == pytest.approx(4.0, abs=2e-3)


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
def test_automatic_coordinate_steps_work_on_each_available_backend(
    backend: str,
) -> None:
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is not installed")

    x = np.array([1.0, 2.0, 3.0])
    specular.set_backend(backend)
    gradient = np.asarray(
        specular.gradient(lambda value: (value * value).sum(), x)
    )
    jacobian = np.asarray(specular.jacobian(lambda value: value * value, x))

    np.testing.assert_allclose(gradient, 2.0 * x, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(
        jacobian,
        np.diag(2.0 * x),
        rtol=3e-4,
        atol=3e-4,
    )


def test_numba_accepts_callables_that_are_compilable_through_a_wrapper() -> None:
    if "numba" not in specular.available_backends():
        pytest.skip("numba is not installed")

    specular.set_backend("numba")
    derivative = specular.derivative(np.float64, 2.0, h=1e-4)
    gradient_sum = specular.gradient(np.sum, np.array([1.0, 2.0]), h=1e-4)
    gradient_norm = specular.gradient(
        np.linalg.norm,
        np.array([3.0, 4.0]),
        h=1e-4,
    )

    assert derivative == pytest.approx(1.0)
    np.testing.assert_allclose(gradient_sum, [1.0, 1.0], rtol=1e-8)
    np.testing.assert_allclose(gradient_norm, [0.6, 0.8], rtol=1e-8)


def test_numba_callback_driver_cache_evicts_old_callbacks() -> None:
    if "numba" not in specular.available_backends():
        pytest.skip("numba is not installed")

    import gc
    import weakref

    from specular.backends.numba import calculation as numba_backend

    cache = numba_backend._cached_compiled_callback
    cache.cache_clear()
    first_callback = lambda value: value
    first_compiled = numba_backend._compile_callback(first_callback)
    center = numba_backend._evaluate_center(first_compiled.dispatcher, 1.0)
    first_compiled.line_scalar(1.0, 1e-4, center.item())
    dispatcher_ref = weakref.ref(first_compiled.dispatcher)
    driver_ref = weakref.ref(first_compiled.line_scalar)
    del first_compiled, center

    try:
        for offset in range(numba_backend._CALLBACK_CACHE_SIZE):
            callback = lambda value, shift=offset: value + shift
            numba_backend._compile_callback(callback)

        cache_info = cache.cache_info()
        assert cache_info.maxsize == numba_backend._CALLBACK_CACHE_SIZE
        assert cache_info.currsize == numba_backend._CALLBACK_CACHE_SIZE
        gc.collect()
        assert dispatcher_ref() is None
        assert driver_ref() is None
    finally:
        cache.cache_clear()


@pytest.mark.parametrize(
    "operation, x",
    [
        (specular.derivative, 0.0),
        (specular.gradient, np.array([0.0])),
        (specular.jacobian, np.array([0.0])),
    ],
)
@pytest.mark.parametrize("h", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_step_value_is_rejected_before_callback(
    operation,
    x,
    h,
) -> None:
    callback_calls = 0

    def callback(value):
        nonlocal callback_calls
        callback_calls += 1
        return value

    with pytest.raises(ValueError, match="finite and greater than zero"):
        operation(callback, x, h=h)

    assert callback_calls == 0


@pytest.mark.parametrize("h", [[1e-6], 1e-6 + 1e-6j, "1e-6", True])
def test_invalid_step_type_is_rejected_before_callback(h) -> None:
    callback_calls = 0

    def callback(value):
        nonlocal callback_calls
        callback_calls += 1
        return value

    with pytest.raises(TypeError, match="concrete real scalar"):
        specular.derivative(callback, 0.0, h=h)

    assert callback_calls == 0


def test_numpy_scalar_step_is_accepted() -> None:
    result = specular.derivative(_square, 2.0, h=np.float64(1e-4))

    assert result == pytest.approx(4.0)


def test_none_selects_an_automatic_step() -> None:
    result = specular.derivative(_square, 2.0, h=None)

    assert result == pytest.approx(4.0)


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
@pytest.mark.parametrize("x, h", [(1.0, 1e-20), (1e20, 1e-6)])
def test_ineffective_step_is_rejected_before_callback(backend, x, h) -> None:
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is not installed")

    callback_calls = 0

    def callback(value):
        nonlocal callback_calls
        callback_calls += 1
        return value

    specular.set_backend(backend)
    with pytest.raises(ValueError, match="perturb x"):
        specular.derivative(callback, x, h=h)

    assert callback_calls == 0


def test_jax_rejects_step_that_underflows_in_effective_dtype() -> None:
    if "jax" not in specular.available_backends():
        pytest.skip("jax is not installed")

    callback_calls = 0

    def callback(value):
        nonlocal callback_calls
        callback_calls += 1
        return value

    specular.set_backend("jax")
    with pytest.raises(ValueError, match="finite and greater than zero"):
        specular.derivative(callback, 0.0, h=1e-50)

    assert callback_calls == 0


def test_jax_static_step_remains_jittable() -> None:
    if "jax" not in specular.available_backends():
        pytest.skip("jax is not installed")

    import jax

    specular.set_backend("jax")
    compiled = jax.jit(
        lambda x: specular.derivative(lambda value: value * value, x, h=1e-2)
    )

    assert float(compiled(2.0)) == pytest.approx(4.0, abs=2e-3)


def test_jax_automatic_step_is_accurate_in_default_float32() -> None:
    if "jax" not in specular.available_backends():
        pytest.skip("jax is not installed")

    specular.set_backend("jax")
    result = specular.derivative(lambda value: value * value, 2.0)

    assert float(result) == pytest.approx(4.0, abs=1e-3)


def test_jax_automatic_step_remains_jittable() -> None:
    if "jax" not in specular.available_backends():
        pytest.skip("jax is not installed")

    import jax

    specular.set_backend("jax")
    compiled = jax.jit(
        lambda x: specular.derivative(lambda value: value * value, x)
    )

    assert float(compiled(2.0)) == pytest.approx(4.0, abs=1e-3)


def test_jax_dynamic_step_is_rejected_at_trace_time() -> None:
    if "jax" not in specular.available_backends():
        pytest.skip("jax is not installed")

    import jax

    specular.set_backend("jax")
    compiled = jax.jit(
        lambda x, h: specular.derivative(lambda value: value * value, x, h=h)
    )

    with pytest.raises(TypeError, match="concrete real scalar"):
        compiled(2.0, 1e-2)


def test_jax_static_ineffective_step_becomes_nan_under_jit() -> None:
    if "jax" not in specular.available_backends():
        pytest.skip("jax is not installed")

    import jax
    import jax.numpy as jnp

    specular.set_backend("jax")
    compiled = jax.jit(
        lambda x: specular.derivative(lambda value: value * value, x, h=1e-20)
    )

    assert bool(jnp.isnan(compiled(1.0)))
