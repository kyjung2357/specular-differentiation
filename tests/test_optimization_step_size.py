"""Acceptance conditions and failure behavior for optimization steps."""

from __future__ import annotations

from contextlib import nullcontext
import math

import numpy as np
import pytest

import specular
from specular.optimization.step_size import LineSearchError, make_step_size


@pytest.mark.parametrize(
    "rule, options, expected",
    [
        (0.125, {}, 0.125),
        ("constant", {"a": 2}, 2),
        ("not_summable", {"a": 2}, 1),
        ("square_summable_not_summable", {"a": 12, "b": 2}, 2),
        ("geometric_series", {"a": 16, "r": 0.5}, 1),
        (lambda n: 1 / n, {}, 0.25),
        ("user_defined", {"user_defined_rule": lambda n: 2 / n}, 0.5),
    ],
)
def test_schedule_formulas_and_numeric_callable_rules(rule, options, expected):
    assert make_step_size(rule, **options)(4) == expected


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_schedule_rejects_invalid_iteration(n):
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        make_step_size()(n)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, [1]])
def test_schedule_rejects_invalid_constant_and_callback_output(value):
    with pytest.raises((TypeError, ValueError)):
        make_step_size(value)
    with pytest.raises((TypeError, ValueError)):
        make_step_size(lambda n: value)(1)


@pytest.mark.parametrize(
    "rule, options",
    [
        ("constant", {"b": 1}),
        ("not_summable", {"a": np.inf}),
        ("square_summable_not_summable", {"b": -1}),
        ("geometric_series", {"r": 1}),
        ("user_defined", {}),
        ("Wolfe", {"c_1": 0.9, "c_2": 0.1}),
        ("strong_Wolfe", {"c_2": 0.9, "c_3": 0.9}),
        ("Armijo", {"rho": 1}),
        ("Armijo", {"max_iter": True}),
        ("Armijo", {"h": np.inf}),
        ("exact", {"tol": 0}),
        ("Wolfe", {"zero_tol": 1e-8}),
    ],
)
def test_rule_specific_options_are_validated(rule, options):
    with pytest.raises((TypeError, ValueError)):
        make_step_size(rule, f=lambda x: x * x, **options)


def test_geometric_underflow_is_not_returned_as_a_zero_step():
    with pytest.raises(ValueError, match="scheduled step"):
        make_step_size("geometric_series", r=0.5)(2000)


def _quadratic(x):
    return 20 * (x - 1) ** 2


def _quadratic_gradient(x):
    return 40 * (x - 1)


@pytest.mark.parametrize("rule", ["Armijo", "Wolfe", "strong_Wolfe"])
def test_returned_steps_satisfy_their_raw_gradient_conditions(rule):
    step = make_step_size(
        rule, f=_quadratic, gradient=_quadratic_gradient,
        t_0=16, max_alpha=32, c_1=0.01,
        **({"c_2": 0.1} if rule != "Armijo" else {}),
    )
    alpha = step(1, 0.0, 1.0)
    assert 0 < alpha < 16
    assert _quadratic(alpha) <= _quadratic(0) + 0.01 * alpha * _quadratic_gradient(0)
    if rule == "Wolfe":
        assert _quadratic_gradient(alpha) >= 0.1 * _quadratic_gradient(0)
    if rule == "strong_Wolfe":
        assert abs(_quadratic_gradient(alpha)) <= -0.1 * _quadratic_gradient(0)


@pytest.mark.parametrize("rule", ["Wolfe", "strong_Wolfe"])
def test_wolfe_expands_small_initial_steps(rule):
    step = make_step_size(rule, f=lambda x: (x - 3) ** 2,
                          gradient=lambda x: 2 * (x - 3), t_0=0.01,
                          max_alpha=10, c_2=0.1)
    alpha = step(1, 0.0, 1.0)
    assert alpha > 0.01
    assert (alpha - 3) ** 2 <= 9 - 1e-4 * alpha * 6
    if rule == "strong_Wolfe":
        assert abs(2 * (alpha - 3)) <= 0.6
    else:
        assert 2 * (alpha - 3) >= -0.6


def test_strong_wolfe_handles_a_reversed_zoom_bracket():
    f = lambda x: (x - 0.6) ** 2
    g = lambda x: 2 * (x - 0.6)
    alpha = make_step_size("strong_Wolfe", f=f, gradient=g, c_3=0.1)(1, 0.0, 1.0)
    assert f(alpha) <= f(0) + 1e-4 * alpha * g(0)
    assert abs(g(alpha)) <= -0.1 * g(0)


@pytest.mark.parametrize("rule", ["Armijo", "Wolfe", "strong_Wolfe"])
def test_line_search_rejects_non_descent_without_taking_a_step(rule):
    step = make_step_size(rule, f=_quadratic, gradient=_quadratic_gradient)
    with pytest.raises(LineSearchError, match="descent"):
        step(1, 0.0, -1.0)


@pytest.mark.parametrize("rule", ["Armijo", "Wolfe", "strong_Wolfe"])
def test_search_contracts_away_from_nonfinite_objective_trials(rule):
    def objective(x):
        return (x - 0.5) ** 2 if x < 0.75 else np.inf

    step = make_step_size(rule, f=objective, gradient=lambda x: 2 * (x - 0.5), t_0=2)
    alpha = step(1, 0.0, 1.0)
    assert 0 < alpha < 0.75
    assert objective(alpha) <= objective(0) - 1e-4 * alpha


def test_wolfe_contracts_away_from_nonfinite_trial_gradient():
    f = lambda x: (x - 0.6) ** 2
    g = lambda x: 2 * (x - 0.6) if x < 0.75 else np.nan
    alpha = make_step_size("strong_Wolfe", f=f, gradient=g)(1, 0.0, 1.0)
    assert 0 < alpha < 0.75
    assert abs(g(alpha)) <= -0.9 * g(0)


@pytest.mark.parametrize("rule", ["Armijo", "Wolfe", "strong_Wolfe", "exact"])
def test_nonfinite_initial_data_raise_search_error(rule):
    step = make_step_size(rule, f=lambda x: np.inf, gradient=lambda x: -1)
    with pytest.raises(LineSearchError, match="finite"):
        step(1, 0.0, 1.0)


@pytest.mark.parametrize("rule", ["Armijo", "Wolfe", "strong_Wolfe"])
def test_no_unchecked_step_is_returned_after_budget_exhaustion(rule):
    step = make_step_size(rule, f=lambda x: 0.0 if x == 0 else np.inf,
                          gradient=lambda x: -1.0, max_iter=3)
    with pytest.raises(LineSearchError):
        step(1, 0.0, 1.0)


def test_wolfe_fails_if_bound_prevents_acceptable_curvature():
    step = make_step_size("Wolfe", f=lambda x: (x - 10) ** 2,
                          gradient=lambda x: 2 * (x - 10), max_alpha=0.01, c_2=0.1)
    with pytest.raises(LineSearchError, match="max_alpha"):
        step(1, 0.0, 1.0)


def test_wolfe_stagnation_fails_with_no_movement():
    step = make_step_size("strong_Wolfe", f=lambda x: -x,
                          gradient=lambda x: -1.0, max_iter=200)
    with pytest.raises(LineSearchError, match="stagnated"):
        step(1, 1e300, 1.0)


def test_cached_values_avoid_recomputing_initial_data():
    objective_points, gradient_points = [], []

    def f(x):
        objective_points.append(x)
        return (x - 1) ** 2

    def g(x):
        gradient_points.append(x)
        return 2 * (x - 1)

    alpha = make_step_size("Wolfe", f=f, gradient=g)(
        1, 0.0, 1.0, gradient_value=-2.0, f_value=1.0,
    )
    assert alpha == 1
    assert objective_points == [1.0]
    assert gradient_points == [1.0]


def test_exact_finds_an_interior_minimum_within_its_evaluation_budget():
    evaluated = []

    def f(x):
        evaluated.append(x)
        return (x - 2.345) ** 2

    alpha = make_step_size("exact", f=f, gradient=lambda x: 2 * (x - 2.345),
                           max_alpha=10, tol=1e-9, max_iter=80)(1, 0.0, 1.0)
    assert alpha == pytest.approx(2.345, abs=1e-8)
    assert len(evaluated) <= 80


def test_exact_can_accept_a_bounded_endpoint():
    step = make_step_size("exact", f=lambda x: (x - 3) ** 2,
                          gradient=lambda x: 2 * (x - 3), max_alpha=1)
    assert step(1, 0.0, 1.0) == 1


def test_exact_does_not_return_an_unconverged_budget_result():
    step = make_step_size("exact", f=_quadratic, gradient=_quadratic_gradient,
                          max_alpha=10, max_iter=5, tol=1e-12)
    with pytest.raises(LineSearchError, match="budget"):
        step(1, 0.0, 1.0)


def test_exact_budget_includes_initial_value_and_never_computes_a_gradient():
    evaluated = []

    def objective(x):
        evaluated.append(x)
        return (x - 1) ** 2

    def forbidden_gradient(x):
        raise AssertionError("Bounded minimization must be derivative free")

    step = make_step_size("exact", f=objective, gradient=forbidden_gradient,
                          max_iter=1, max_alpha=2)
    with pytest.raises(LineSearchError, match="budget"):
        step(1, 0.0, 1.0)
    assert evaluated == [0.0]


def test_exact_without_gradient_obeys_the_total_objective_budget():
    evaluated = []

    def objective(x):
        evaluated.append(x)
        return (x - 1) ** 2

    step = make_step_size("exact", f=objective, max_iter=1, max_alpha=2)
    with pytest.raises(LineSearchError, match="budget"):
        step(1, 0.0, 1.0)
    assert evaluated == [0.0]


def test_search_recovers_from_arithmetic_overflow_in_trial_objective():
    objective = lambda x: (x - 2) ** 2 + math.exp(x)
    gradient = lambda x: 2 * (x - 2) + math.exp(x)
    alpha = make_step_size("strong_Wolfe", f=objective, gradient=gradient,
                           t_0=1024)(1, 0.0, 1.0)
    assert objective(alpha) <= objective(0) + 1e-4 * alpha * gradient(0)
    assert abs(gradient(alpha)) <= -0.9 * gradient(0)


def test_ordinary_default_differentiates_a_vector_objective():
    f = lambda x: np.sum((x - np.array([1.0, 2.0])) ** 2)
    point, direction = np.zeros(2), np.array([1.0, 2.0])
    alpha = make_step_size("strong_Wolfe", f=f)(1, point, direction)
    assert f(point + alpha * direction) < f(point)
    np.testing.assert_array_equal(point, [0, 0])
    np.testing.assert_array_equal(direction, [1, 2])


def test_line_search_preserves_finite_affine_cancellation():
    visited = []

    def objective(x):
        visited.append(x)
        return x / 1e308

    alpha = make_step_size("Armijo", f=objective, gradient=lambda x: 1e-308,
                           t_0=2)(1, 1e308, -1e308)
    assert alpha == 2
    assert visited == [1e308, -1e308]


def test_ordinary_difference_handles_overflowing_sample_differences():
    step = make_step_size("Armijo", f=lambda x: x, h=1e308)
    assert step(1, 0.0, -1.0) == 1


def _backend_scalar_objective(x):
    return (x - 1) ** 2


def _backend_vector_objective(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("rule", ["specular_Armijo", "specular_Wolfe", "specular_strong_Wolfe"])
def test_standalone_specular_search_honors_the_selected_backend(backend, vector, rule):
    if backend not in specular.available_backends():
        pytest.skip(f"{backend} is not installed")
    objective = _backend_vector_objective if vector else _backend_scalar_objective
    point = np.zeros(2) if vector else 0.0
    direction = np.array([1.0, 2.0]) if vector else 1.0
    context = nullcontext()
    if backend == "jax":
        import jax

        context = jax.enable_x64(True)
    with specular.use_backend(backend), context:
        alpha = make_step_size(rule, f=objective)(1, point, direction)
        assert alpha > 0
        assert objective(point + alpha * direction) < objective(point)
