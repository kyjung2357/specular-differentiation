"""End-to-end contracts for the composed optimization loop."""

from __future__ import annotations

from contextlib import nullcontext
import subprocess
import sys

import numpy as np
import pytest

import specular
from specular.optimization import make_direction, make_step_size, minimize, specular_gradient


@pytest.fixture(autouse=True)
def _numpy_backend():
    with specular.use_backend("numpy"):
        yield


def _square(x):
    return x * x


def _quadratic(x):
    return (x * x).sum()


def test_speg_records_only_accepted_updates_and_final_step_convergence():
    result = specular_gradient(_square, 3.0, a=1.0, max_iter=3)
    assert result.success
    assert result.iteration == 3
    assert result.solution == 0.0
    assert result.func_val == 0.0
    np.testing.assert_array_equal(result.history["variables"], [3, 2, 1, 0])
    np.testing.assert_array_equal(result.history["values"], [9, 4, 1, 0])
    np.testing.assert_array_equal(result.history["step_sizes"], [1, 1, 1])
    assert result.last_record() == (0.0, 0.0, result.runtime)
    assert result.get_history()[2] == result.runtime


def test_small_raw_gradient_stops_before_normalization():
    result = specular_gradient(_square, 1.0, gradient=lambda x: 1e-12, tol=1e-10)
    assert result.success
    assert result.iteration == 0
    assert result.solution == 1.0


def test_initial_stationary_point_and_zero_iteration_budget():
    assert specular_gradient(_square, 0.0).success
    result = specular_gradient(_square, 3.0, max_iter=0)
    assert not result.success
    assert result.iteration == 0
    assert result.func_val == 9.0
    np.testing.assert_array_equal(result.history["variables"], [3.0])


def test_speg_uses_unit_direction_and_counts_gradient_once_per_iterate():
    calls = []

    def raw_gradient(x):
        calls.append(x.copy())
        return np.array([3.0, 4.0])

    result = specular_gradient(_quadratic, [0.0, 0.0], gradient=raw_gradient,
                               step_size=2.0, max_iter=2, tol=0)
    np.testing.assert_allclose(result.solution, [-2.4, -3.2])
    assert len(calls) == 3  # initial, second iterate, final endpoint
    assert result.iteration == 2
    assert not result.success


def test_custom_direction_and_prebuilt_schedule_compose():
    direction = make_direction(lambda n, x: -2.0 * x)
    schedule = make_step_size(lambda n: 0.25)
    result = minimize(_quadratic, [2.0, -4.0], direction=direction,
                      step_size=schedule, max_iter=3, tol=0)
    np.testing.assert_allclose(result.solution, [0.25, -0.5])
    assert result.iteration == 3


def test_custom_callbacks_cannot_modify_initial_point_or_history():
    initial = np.array([3.0, 4.0])

    def direction(n, x):
        x[:] = -1
        return x

    def step(n, x, d):
        x[:] = 1000
        d[:] = 0
        return 1.0

    result = minimize(_quadratic, initial, direction=direction, step_size=step,
                      max_iter=1, tol=0)
    np.testing.assert_array_equal(initial, [3, 4])
    np.testing.assert_array_equal(result.history["variables"], [[3, 4], [2, 3]])
    assert not np.shares_memory(initial, result.solution)


@pytest.mark.parametrize("rule", ["Armijo", "Wolfe", "strong_Wolfe",
                                  "specular_Armijo", "specular_Wolfe", "specular_strong_Wolfe"])
def test_speg_line_search_composition(rule):
    result = specular_gradient(_square, 3.0, rule, max_iter=4)
    assert result.success
    assert result.solution == 0.0
    assert result.iteration == 3


def test_failed_line_search_does_not_commit_a_trial():
    result = minimize(_square, 1.0, direction=lambda n, x: 1.0,
                      step_size="Wolfe", line_search_gradient=lambda x: 2*x)
    assert not result.success
    assert result.iteration == 0
    assert result.solution == 1.0
    assert "line search failed" in result.stop_reason
    assert result.history["variables"].tolist() == [1.0]


def test_specular_search_reuses_explicit_raw_gradient():
    seen = []

    def raw(x):
        seen.append(x)
        return 2*x

    result = specular_gradient(_square, 2.0, "specular_Wolfe", gradient=raw,
                               max_iter=1, tol=0)
    assert result.iteration == 1
    assert 1.0 in seen  # candidate gradient is provided by the same callback


def test_ordinary_search_has_separate_gradient_callback():
    seen = []

    def search_gradient(x):
        seen.append(x)
        return 2*x

    result = specular_gradient(_square, 2.0, "Wolfe", gradient=lambda x: 100.0,
                               line_search_gradient=search_gradient, max_iter=1)
    assert result.iteration == 1
    assert 2.0 in seen and 1.0 in seen


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("rule", ["constant", "specular_Wolfe"])
def test_backend_switching_applies_to_complete_speg_runs(backend, vector, rule):
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is unavailable")
    if backend == "jax":
        import jax
        precision = jax.enable_x64(True)
    else:
        precision = nullcontext()
    with precision, specular.use_backend(backend):
        f, x0 = (_quadratic, np.array([3.0, 0.0])) if vector else (_square, 3.0)
        result = specular_gradient(f, x0, rule, max_iter=3)
        assert specular.get_backend() == backend
        assert result.success
        np.testing.assert_allclose(result.solution, np.zeros_like(x0), atol=1e-8)


@pytest.mark.parametrize("options, error", [
    ({"max_iter": True}, TypeError), ({"max_iter": 1.5}, TypeError),
    ({"max_iter": -1}, ValueError), ({"tol": np.nan}, ValueError),
    ({"tol": -1}, ValueError), ({"h": 0}, ValueError),
    ({"h": True}, TypeError), ({"record_history": 1}, TypeError),
    ({"gradient": 3}, TypeError), ({"line_search_gradient": 3}, TypeError),
])
def test_invalid_configuration_is_rejected_before_objective(options, error):
    def f(x):
        pytest.fail("invalid configuration must not evaluate objective")
    with pytest.raises(error):
        minimize(f, 1.0, **options)


@pytest.mark.parametrize("initial", [[], [[1.0]], float("inf"), 1j, True])
def test_invalid_initial_points(initial):
    with pytest.raises((TypeError, ValueError)):
        minimize(_square, initial)


def test_invalid_objective_or_gradient_shape_is_rejected():
    with pytest.raises(TypeError, match="scalar"):
        minimize(lambda x: [1.0], 1.0)
    with pytest.raises(ValueError, match="finite"):
        minimize(lambda x: np.inf, 1.0)
    with pytest.raises(ValueError, match="shape"):
        minimize(_quadratic, [1.0, 2.0], gradient=lambda x: 1.0)


def test_invalid_scheduled_candidate_keeps_last_accepted_point():
    result = specular_gradient(lambda x: np.inf if x < 0 else x, 0.5,
                               gradient=lambda x: 1.0, a=1.0)
    assert result.solution == 0.5
    assert result.iteration == 0
    assert "non-finite objective" in result.stop_reason


def test_stagnation_is_not_reported_as_convergence():
    result = specular_gradient(lambda x: x, 1e100, gradient=lambda x: 1.0, a=1e-100)
    assert result.iteration == 0
    assert not result.success
    assert "working precision" in result.stop_reason


def test_representable_affine_update_survives_intermediate_overflow():
    with np.errstate(all="raise"):
        result = minimize(lambda x: x / 1e308, 1e308, gradient=lambda x: 1.0,
                          direction=lambda n, x: -1e308, step_size=2.0,
                          max_iter=1, tol=0)
    assert result.solution == -1e308
    assert result.iteration == 1
    assert result.func_val == -1.0


def test_true_update_overflow_preserves_last_point():
    result = minimize(lambda x: x / 1e308, -1e308, gradient=lambda x: 1.0,
                      direction=lambda n, x: -1e308, step_size=2.0, max_iter=1)
    assert result.solution == -1e308
    assert result.iteration == 0
    assert "non-finite proposed point" in result.stop_reason


@pytest.mark.parametrize("backend", ["numba", "jax"])
def test_specular_rule_name_normalization_keeps_backend_callback(backend):
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is unavailable")
    with specular.use_backend(backend):
        result = minimize(_square, 3.0, step_size=" specular_Wolfe ", max_iter=3)
    assert result.success
    assert result.solution == 0.0


def test_disabled_history_preserves_result_and_vector_dimensions():
    result = specular_gradient(_quadratic, [3.0, 0.0], max_iter=2, record_history=False)
    assert result.iteration == 2
    np.testing.assert_allclose(result.solution, [1.0, 0.0])
    assert result.history["variables"].shape == (0, 2)
    assert result.history["values"].shape == (0,)
    assert result.history["step_sizes"].shape == (0,)


def test_step_parameters_cannot_silently_override_each_other():
    with pytest.raises(TypeError, match="duplicate"):
        specular_gradient(_square, 1.0, a=1.0, step_options={"a": 0.5})
    with pytest.raises(ValueError, match="step_options"):
        minimize(_square, 1.0, step_size=lambda n, x, d: 1.0, step_options={"a": 1.0})


def test_optimization_exports_are_lazy_and_do_not_import_optional_dependencies():
    code = """
import sys
import specular
assert 'specular.optimization' not in sys.modules
assert 'specular_gradient' in dir(specular)
from specular.optimization import specular_gradient, minimize, OptimizationResult
assert specular.specular_gradient is specular_gradient
assert specular.minimize is minimize
assert specular.OptimizationResult is OptimizationResult
assert 'scipy' not in sys.modules
assert 'tqdm' not in sys.modules
assert 'jax' not in sys.modules
assert 'numba' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)
