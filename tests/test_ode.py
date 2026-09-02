"""Tests for the scalar specular ODE methods."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import fields

import numpy as np
import pytest

import specular
from specular.ode import (
    ODEResult,
    ellipse_scheme,
    euler_scheme_1,
    euler_scheme_2,
    euler_scheme_5,
)
from specular.ode import solver as ode_solver
from specular.ode._ellipse import (
    _defect_minimizing_mean,
    _defect_minimizing_scale,
    _fourth_order_scale,
    _numeric_derivatives_of_F,
    _third_order_scale,
)


def test_ode_api_is_available_from_the_top_level_package() -> None:
    assert specular.ODEResult is ODEResult
    assert specular.ellipse_scheme is ellipse_scheme
    assert specular.euler_scheme_1 is euler_scheme_1
    assert specular.euler_scheme_2 is euler_scheme_2
    assert specular.euler_scheme_5 is euler_scheme_5
    assert ode_solver.ellipse_scheme is ellipse_scheme
    assert ode_solver.euler_scheme_1 is euler_scheme_1
    assert ode_solver.euler_scheme_2 is euler_scheme_2
    assert ode_solver.euler_scheme_5 is euler_scheme_5
    assert "euler_scheme_1" in specular.__all__
    assert "euler_scheme_2" in specular.__all__
    assert "euler_scheme_5" in specular.__all__
    assert "ellipse_scheme_3rd_order" not in specular.__all__
    assert "ellipse_scheme_4th_order" not in specular.__all__
    assert not hasattr(specular, "ellipse_scheme_3rd_order")
    assert not hasattr(specular, "ellipse_scheme_4th_order")
    assert not hasattr(sys.modules["specular.ode"], "ellipse_scheme_3rd_order")
    assert not hasattr(sys.modules["specular.ode"], "ellipse_scheme_4th_order")
    assert [field.name for field in fields(ODEResult)] == [
        "t",
        "u",
        "sigma",
        "number_of_field_evaluations",
    ]


def test_importing_specular_does_not_import_heavy_example_dependencies() -> None:
    code = (
        "import sys; import specular; "
        "assert 'specular.ode' not in sys.modules; "
        "assert specular.euler_scheme_1.__name__ == 'euler_scheme_1'; "
        "assert specular.euler_scheme_2.__name__ == 'euler_scheme_2'; "
        "assert specular.euler_scheme_5.__name__ == 'euler_scheme_5'; "
        "assert specular.ellipse_scheme.__name__ == 'ellipse_scheme'; "
        "assert 'specular.ode' in sys.modules; "
        "assert 'jax' not in sys.modules; "
        "assert 'jaxlib' not in sys.modules; "
        "assert 'numba' not in sys.modules; "
        "assert 'scipy' not in sys.modules; "
        "assert 'pandas' not in sys.modules; "
        "assert 'matplotlib' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_constant_equation_is_exact_and_result_has_minimal_shapes() -> None:
    calls = 0

    def F(t: float, u: float) -> float:
        nonlocal calls
        calls += 1
        return 2.5

    result = ellipse_scheme(
        F,
        1.0,
        2.5,
        -3.0,
        n_steps=6,
        sigma_n=0.75,
    )
    expected_t = np.linspace(1.0, 2.5, 7)

    assert isinstance(result, ODEResult)
    assert result.t.shape == (7,)
    assert result.u.shape == (7,)
    assert result.sigma.shape == (6,)
    np.testing.assert_array_equal(result.t, expected_t)
    np.testing.assert_allclose(
        result.u,
        -3.0 + 2.5 * (expected_t - expected_t[0]),
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_array_equal(result.sigma, np.full(6, 0.75))
    assert result.number_of_field_evaluations == calls


def test_ellipse_scheme_avoids_intermediate_overflow_in_finite_update(
) -> None:
    result = ellipse_scheme(
        lambda t, u: 1e308,
        0.0,
        2.0,
        -1e308,
        n_steps=1,
        sigma_n=1.0,
    )

    np.testing.assert_array_equal(result.u, [-1e308, 1e308])


def test_coupled_ellipse_scheme_avoids_intermediate_overflow(
) -> None:
    result = ellipse_scheme(
        lambda t, u: 1e308,
        0.0,
        2.0,
        -1e308,
        n_steps=1,
        fourth_order=True,
        derivatives_of_F=lambda point: np.array([0.0, 0.0]),
    )

    np.testing.assert_array_equal(result.u, [-1e308, 1e308])
    np.testing.assert_array_equal(result.sigma, [1.0])


@pytest.mark.parametrize(
    "mode",
    ["third_order", "fourth_order", "minimize_defect"],
)
def test_automatic_modes_count_every_internal_field_evaluation(
    mode: str,
) -> None:
    calls = 0

    def F(t: float, u: float) -> float:
        nonlocal calls
        calls += 1
        return -u

    result = ellipse_scheme(
        F,
        0.0,
        0.1,
        1.0,
        n_steps=1,
        atol=1e-13,
        rtol=1e-13,
        **{mode: True},
    )

    assert result.number_of_field_evaluations == calls


def test_uniform_mesh_reaches_both_endpoints_exactly() -> None:
    result = ellipse_scheme(
        lambda t, u: 0.0,
        -0.75,
        1.25,
        4.0,
        n_steps=8,
        sigma_n=1.0,
    )

    assert result.t[0] == -0.75
    assert result.t[-1] == 1.25
    np.testing.assert_allclose(np.diff(result.t), 0.25, rtol=0.0, atol=0.0)


def test_uniform_mesh_avoids_endpoint_subtraction_overflow() -> None:
    result = ellipse_scheme(
        lambda t, u: 0.0,
        -1e308,
        1e308,
        0.0,
        n_steps=2,
        sigma_n=1.0,
    )

    np.testing.assert_array_equal(result.t, [-1e308, 0.0, 1e308])
    np.testing.assert_array_equal(result.u, np.zeros(3))


def test_solver_uses_each_represented_time_interval_as_h_n() -> None:
    calls: list[tuple[int, float, float, float]] = []

    def sigma_n(n: int, t_n: float, u_n: float, h_n: float) -> float:
        calls.append((n, t_n, u_n, h_n))
        return h_n

    t_0 = 1e16
    T = t_0 + 10.0
    result = ellipse_scheme(
        lambda t, u: 1.0,
        t_0,
        T,
        7.0,
        n_steps=3,
        sigma_n=sigma_n,
    )
    represented_steps = np.diff(result.t)

    np.testing.assert_array_equal(represented_steps, [4.0, 2.0, 4.0])
    assert not np.all(represented_steps == (T - t_0) / 3.0)
    np.testing.assert_array_equal(result.u, 7.0 + (result.t - t_0))
    np.testing.assert_array_equal(result.sigma, represented_steps)
    assert [call[0] for call in calls] == [0, 1, 2]
    np.testing.assert_array_equal([call[1] for call in calls], result.t[:-1])
    np.testing.assert_array_equal([call[2] for call in calls], result.u[:-1])
    np.testing.assert_array_equal([call[3] for call in calls], represented_steps)


def _reference_unscaled_C(alpha: float, beta: float) -> float:
    """Evaluate the defining unscaled specular mean on moderate data."""

    slope_sum = alpha + beta
    if slope_sum == 0.0:
        return 0.0
    w = (alpha * beta - 1.0) / slope_sum
    return w + math.copysign(math.sqrt(1.0 + w * w), slope_sum)


@pytest.mark.parametrize(
    ("method", "uses_field_history"),
    [(euler_scheme_1, True), (euler_scheme_2, False)],
)
def test_two_step_euler_methods_follow_their_defining_recurrence(
    method,
    uses_field_history: bool,
) -> None:
    def F(t: float, u: float) -> float:
        return 1.0 + t + 0.2 * u

    result = method(F, 0.0, 1.5, 0.5, 0.9, n_steps=3)
    expected = np.empty(4)
    expected[:2] = [0.5, 0.9]

    for step in range(1, 3):
        current = F(float(result.t[step]), float(expected[step]))
        if uses_field_history:
            second_slope = F(
                float(result.t[step - 1]),
                float(expected[step - 1]),
            )
        else:
            second_slope = (
                float(expected[step]) - float(expected[step - 1])
            ) / float(result.t[step] - result.t[step - 1])
        expected[step + 1] = expected[step] + (
            float(result.t[step + 1] - result.t[step])
            * _reference_unscaled_C(current, second_slope)
        )

    np.testing.assert_allclose(
        result.u,
        expected,
        rtol=2e-15,
        atol=2e-15,
    )
    np.testing.assert_array_equal(result.sigma, np.ones(3))


@pytest.mark.parametrize("method", [euler_scheme_1, euler_scheme_2])
def test_two_step_euler_methods_are_exact_for_a_constant_field(
    method,
) -> None:
    t_0 = -0.5
    T = 1.5
    n_steps = 8
    u_0 = -3.0
    slope = 2.5
    h = (T - t_0) / n_steps

    result = method(
        lambda t, u: slope,
        t_0,
        T,
        u_0,
        u_0 + h * slope,
        n_steps=n_steps,
    )

    np.testing.assert_allclose(
        result.u,
        u_0 + slope * (result.t - t_0),
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(result.sigma, np.ones(n_steps))


@pytest.mark.parametrize("method", [euler_scheme_1, euler_scheme_2])
def test_two_step_euler_methods_require_an_external_u_1(method) -> None:
    with pytest.raises(TypeError, match="u_1"):
        method(  # type: ignore[call-arg]
            lambda t, u: 0.0,
            0.0,
            1.0,
            0.0,
            n_steps=1,
        )


@pytest.mark.parametrize("method", [euler_scheme_1, euler_scheme_2])
def test_one_interval_two_step_call_returns_the_starter_without_F(
    method,
) -> None:
    calls = 0

    def F(t: float, u: float) -> float:
        nonlocal calls
        calls += 1
        raise AssertionError("F must not be evaluated for n_steps=1")

    result = method(F, 2.0, 5.0, -1.0, 7.0, n_steps=1)

    assert calls == 0
    np.testing.assert_array_equal(result.t, [2.0, 5.0])
    np.testing.assert_array_equal(result.u, [-1.0, 7.0])
    np.testing.assert_array_equal(result.sigma, [1.0])
    assert result.number_of_field_evaluations == 0


@pytest.mark.parametrize(
    ("method", "expected_indices"),
    [
        (euler_scheme_1, [0, 1, 2, 3]),
        (euler_scheme_2, [1, 2, 3]),
    ],
)
def test_two_step_euler_field_evaluation_counts_and_nodes(
    method,
    expected_indices: list[int],
) -> None:
    calls: list[tuple[float, float]] = []

    def F(t: float, u: float) -> float:
        calls.append((t, u))
        return 0.25

    result = method(F, 0.0, 1.0, 2.0, 2.0625, n_steps=4)

    assert len(calls) == len(expected_indices)
    assert result.number_of_field_evaluations == len(calls)
    np.testing.assert_array_equal(
        [call[0] for call in calls],
        result.t[expected_indices],
    )
    np.testing.assert_array_equal(
        [call[1] for call in calls],
        result.u[expected_indices],
    )


@pytest.mark.parametrize("method", [euler_scheme_1, euler_scheme_2])
def test_two_step_euler_methods_have_generic_first_order_convergence(
    method,
) -> None:
    errors = []
    for n_steps in (20, 40, 80):
        h = 1.0 / n_steps
        result = method(
            lambda t, u: -u,
            0.0,
            1.0,
            1.0,
            math.exp(-h),
            n_steps=n_steps,
        )
        errors.append(abs(result.u[-1] - math.exp(-1.0)))

    assert errors[0] > errors[1] > errors[2] > 0.0
    observed_orders = [
        math.log2(errors[index] / errors[index + 1]) for index in (0, 1)
    ]
    assert min(observed_orders) > 0.97


def test_euler_scheme_5_is_the_unit_scale_ellipse_scheme() -> None:
    def F(t: float, u: float) -> float:
        return 1.0 - 0.5 * u + 0.1 * math.sin(t)

    kwargs = {
        "n_steps": 16,
        "atol": 1e-13,
        "rtol": 1e-13,
        "max_iter": 200,
    }
    actual = euler_scheme_5(F, 0.0, 1.0, -1.0, **kwargs)
    expected = ellipse_scheme(F, 0.0, 1.0, -1.0, sigma_n=1.0, **kwargs)

    np.testing.assert_array_equal(actual.t, expected.t)
    np.testing.assert_array_equal(actual.u, expected.u)
    np.testing.assert_array_equal(actual.sigma, expected.sigma)
    np.testing.assert_array_equal(actual.sigma, np.ones(16))
    assert (
        actual.number_of_field_evaluations
        == expected.number_of_field_evaluations
    )


@pytest.mark.parametrize("method", [euler_scheme_1, euler_scheme_2])
def test_two_step_euler_methods_support_a_large_represented_grid(
    method,
) -> None:
    result = method(
        lambda t, u: 0.0,
        -1e308,
        1e308,
        3.0,
        3.0,
        n_steps=2,
    )

    np.testing.assert_array_equal(result.t, [-1e308, 0.0, 1e308])
    np.testing.assert_array_equal(result.u, [3.0, 3.0, 3.0])
    np.testing.assert_array_equal(result.sigma, [1.0, 1.0])


@pytest.mark.parametrize("method", [euler_scheme_1, euler_scheme_2])
def test_two_step_euler_methods_reject_nonfinite_field_values(
    method,
) -> None:
    with pytest.raises(ValueError, match=r"F\(.*must be finite"):
        method(
            lambda t, u: math.nan,
            0.0,
            1.0,
            0.0,
            0.0,
            n_steps=2,
        )


@pytest.mark.parametrize(
    ("method", "u_0", "u_1"),
    [
        (euler_scheme_1, np.finfo(np.float64).max, np.finfo(np.float64).max),
        (euler_scheme_2, 0.0, np.finfo(np.float64).max),
    ],
)
def test_two_step_euler_methods_reject_a_nonfinite_advanced_state(
    method,
    u_0: float,
    u_1: float,
) -> None:
    with pytest.raises(RuntimeError, match=r"state is non-finite at step 1"):
        method(
            lambda t, u: np.finfo(np.float64).max,
            0.0,
            2.0,
            u_0,
            u_1,
            n_steps=2,
        )


@pytest.mark.parametrize("method", [euler_scheme_1, euler_scheme_2])
@pytest.mark.parametrize("bad_u_1", [math.nan, math.inf, -math.inf])
def test_two_step_euler_methods_reject_nonfinite_u_1_before_F(
    method,
    bad_u_1: float,
) -> None:
    calls = 0

    def F(t: float, u: float) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    with pytest.raises(ValueError, match="u_1"):
        method(F, 0.0, 1.0, 0.0, bad_u_1, n_steps=2)

    assert calls == 0


@pytest.mark.parametrize("method", [euler_scheme_1, euler_scheme_2])
@pytest.mark.parametrize("bad_n_steps", [0, -1, True, 1.5])
def test_two_step_euler_methods_validate_n_steps_before_F(
    method,
    bad_n_steps,
) -> None:
    calls = 0

    def F(t: float, u: float) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    with pytest.raises((TypeError, ValueError), match="n_steps"):
        method(F, 0.0, 1.0, 0.0, 0.0, n_steps=bad_n_steps)

    assert calls == 0


@pytest.mark.parametrize("method", [euler_scheme_1, euler_scheme_2])
@pytest.mark.parametrize(
    ("t_0", "T"),
    [(0.0, 0.0), (1.0, 0.0), (0.0, math.inf), (math.nan, 1.0)],
)
def test_two_step_euler_methods_validate_the_time_interval_before_F(
    method,
    t_0: float,
    T: float,
) -> None:
    calls = 0

    def F(t: float, u: float) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    with pytest.raises((TypeError, ValueError)):
        method(F, t_0, T, 0.0, 0.0, n_steps=2)

    assert calls == 0


@pytest.mark.parametrize("method", [euler_scheme_1, euler_scheme_2])
def test_two_step_euler_methods_reject_a_noncallable_field(method) -> None:
    with pytest.raises(TypeError, match="F"):
        method(1.0, 0.0, 1.0, 0.0, 0.0, n_steps=2)


@pytest.mark.parametrize("method", [euler_scheme_1, euler_scheme_2])
def test_two_step_euler_methods_preserve_field_exceptions(method) -> None:
    class FieldError(Exception):
        pass

    def F(t: float, u: float) -> float:
        raise FieldError("field failed")

    with pytest.raises(FieldError, match="field failed"):
        method(F, 0.0, 1.0, 0.0, 0.0, n_steps=2)


def test_affine_equation_matches_its_exact_solution() -> None:
    # u' = 1 - u/2, u(0) = -1 has u(t) = 2 - 3 exp(-t/2).
    result = ellipse_scheme(
        lambda t, u: 1.0 - 0.5 * u,
        0.0,
        1.0,
        -1.0,
        n_steps=100,
        sigma_n=0.7,
        atol=1e-13,
        rtol=1e-13,
        max_iter=100,
    )
    expected = 2.0 - 3.0 * np.exp(-0.5 * result.t)

    np.testing.assert_allclose(result.u, expected, rtol=0.0, atol=2e-5)


def test_sigma_n_callback_receives_step_index_state_and_step_size_once() -> None:
    calls: list[tuple[int, float, float, float]] = []

    def sigma_n(n: int, t_n: float, u_n: float, h_n: float) -> float:
        calls.append((n, t_n, u_n, h_n))
        return (n + 1) * h_n

    result = ellipse_scheme(
        lambda t, u: 3.0,
        2.0,
        3.0,
        -1.0,
        n_steps=8,
        sigma_n=sigma_n,
    )

    assert len(calls) == 8
    assert [call[0] for call in calls] == list(range(8))
    np.testing.assert_allclose([call[1] for call in calls], result.t[:-1])
    np.testing.assert_allclose([call[2] for call in calls], result.u[:-1])
    np.testing.assert_array_equal([call[3] for call in calls], np.full(8, 0.125))
    np.testing.assert_allclose(result.sigma, 0.125 * np.arange(1.0, 9.0))


def test_mesh_dependent_scale_reproduces_fourth_order_inverse_ode_family(
) -> None:
    # For u' = 1/u and sigma_n = h, the manuscript proves order 2 + 2 = 4.
    errors = []
    for n_steps in (20, 40, 80):
        result = ellipse_scheme(
            lambda t, u: 1.0 / u,
            0.0,
            1.0,
            1.0,
            n_steps=n_steps,
            sigma_n=lambda n, t_n, u_n, h_n: h_n,
            atol=1e-14,
            rtol=1e-14,
            max_iter=100,
        )
        errors.append(abs(result.u[-1] - math.sqrt(3.0)))

    assert errors[0] > errors[1] > errors[2] > 0.0
    observed_orders = [
        math.log2(errors[index] / errors[index + 1]) for index in (0, 1)
    ]
    assert min(observed_orders) > 3.7


def _decay_derivatives(point: np.ndarray) -> np.ndarray:
    """Return ``(L_F F, L_F^2 F)`` for ``F(t, u) = -u``."""

    u = float(point[1])
    return np.array([u, -u])


def _defect_for_decay(u: float, sigma: float) -> float:
    F_value = -u
    first_derivative = u
    second_derivative = -u
    return second_derivative - (
        3.0 * F_value * first_derivative**2
        / (sigma**2 + F_value**2)
    )


def test_third_order_selects_the_left_defect_cancelling_scale() -> None:
    received: list[tuple[np.ndarray, np.ndarray]] = []
    returned: list[tuple[np.ndarray, np.ndarray]] = []

    def derivatives_of_F(point: np.ndarray) -> np.ndarray:
        received.append((point, point.copy()))
        values = _decay_derivatives(point)
        values.setflags(write=False)
        returned.append((values, values.copy()))
        return values

    result = ellipse_scheme(
        lambda t, u: -u,
        0.0,
        0.4,
        1.0,
        n_steps=4,
        third_order=True,
        derivatives_of_F=derivatives_of_F,
        atol=1e-13,
        rtol=1e-13,
    )

    # Here Q = L_F F = u and R = L_F^2 F = -u, so the unique
    # defect-cancelling scale is sigma_n = sqrt(2) u_n.
    np.testing.assert_allclose(
        result.sigma,
        math.sqrt(2.0) * result.u[:-1],
        rtol=2e-13,
        atol=2e-13,
    )
    assert received
    assert returned
    for values, original in [*received, *returned]:
        np.testing.assert_array_equal(values, original)


def test_third_order_default_numerical_derivatives_select_expected_scale(
) -> None:
    result = ellipse_scheme(
        lambda t, u: -u,
        0.0,
        0.2,
        1.0,
        n_steps=2,
        third_order=True,
        atol=1e-13,
        rtol=1e-13,
    )

    np.testing.assert_allclose(
        result.sigma,
        math.sqrt(2.0) * result.u[:-1],
        rtol=2e-5,
        atol=2e-5,
    )


def test_numerical_field_derivatives_avoid_intermediate_overflow() -> None:
    def large_radius_field(t: float, u: float) -> float:
        return 1.0 + (1e-150 * t) ** 2

    Q, R = _numeric_derivatives_of_F(
        large_radius_field,
        0.0,
        0.0,
        1.0,
        1e200,
        step=0,
    )
    assert Q == 0.0
    assert R == pytest.approx(2e-300, rel=5e-15, abs=0.0)

    constant_result = ellipse_scheme(
        lambda t, u: 1e308,
        0.0,
        1e-308,
        0.0,
        n_steps=1,
        third_order=True,
        atol=0.0,
        rtol=1e-13,
    )
    np.testing.assert_array_equal(constant_result.u, [0.0, 0.9999999999999999])
    np.testing.assert_array_equal(constant_result.sigma, [1.0])


@pytest.mark.parametrize(
    ("mode", "expected_scale_ratio"),
    [
        ("third_order", math.sqrt(2.0)),
        ("fourth_order", 1.3497316999446598),
        ("minimize_defect", 1.3497316999446598),
    ],
)
@pytest.mark.parametrize("amplitude", [1e-200, 1.0, 1e200])
def test_automatic_scale_selection_is_amplitude_stable(
    mode: str,
    expected_scale_ratio: float,
    amplitude: float,
) -> None:
    result = ellipse_scheme(
        lambda t, u: -u,
        0.0,
        0.1,
        amplitude,
        n_steps=1,
        derivatives_of_F=_decay_derivatives,
        atol=0.0,
        rtol=1e-13,
        max_iter=200,
        **{mode: True},
    )

    assert result.sigma[0] / amplitude == pytest.approx(
        expected_scale_ratio,
        rel=2e-13,
    )
    assert result.u[-1] / amplitude == pytest.approx(
        0.9048325141612076
        if mode == "third_order"
        else 0.904837117132141,
        rel=2e-13,
    )


def test_fourth_order_scale_allows_cancelling_large_endpoint_terms() -> None:
    Q_left = 2e154
    Q_right = math.nextafter(math.sqrt(1.25) * Q_left, 0.0)
    left = (1.0, Q_left, 0.0)
    right = (-2.0, Q_right, 2.8145366785447223e292)

    assert _fourth_order_scale(left, right, step=0) == 1.0


@pytest.mark.parametrize(
    ("case", "left", "right", "expected"),
    [
        ("E5a", (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), math.sqrt(2.0)),
        (
            "E5b",
            (1.0, 1.0, 0.0),
            (-2.0, 1.0, 1.0),
            math.sqrt(3.0 * math.sqrt(2.0) - 4.0),
        ),
    ],
)
def test_fourth_order_scale_selects_sigma_in_e5a_and_e5b(
    case: str,
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    expected: float,
) -> None:
    actual = _fourth_order_scale(left, right, step=0)
    assert actual == pytest.approx(
        expected,
        rel=3e-15,
        abs=0.0,
    ), case


@pytest.mark.parametrize(
    ("case", "left", "right"),
    [
        ("E1", (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        ("E2", (1.0, 1.0, 0.0), (-2.0, 1.0, 0.0)),
        ("E3a", (1.0, 1.0, 0.0), (-4.0, 2.0, 0.0)),
        ("E3b", (1.0, 1.0, 0.0), (2.0, 1.0, 0.0)),
        ("E4", (-4.0, 2.0, 0.0), (1.0, 1.0, -1.0)),
        ("E5c", (-4.0, 3.0, 0.0), (1.0, 1.0, -5.0)),
        ("E6a", (1.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ("E6b(i)", (-4.0, 0.0, 0.0), (-4.0, 2.0, -3.0)),
        ("E6b(ii)", (0.0, 0.0, 3.0), (1.0, 1.0, 0.0)),
        (
            "E6c (interior)",
            (-4.0, 1.0, 0.0),
            (1.0, 1.0, -20.0),
        ),
        (
            "E6c (zero boundary)",
            (1.0, 1.0, 10.0),
            (1.0, 1.0, 0.0),
        ),
        (
            "E6c (infinite boundary)",
            (-4.0, 0.0, 0.0),
            (-4.0, 1.0, 1.0),
        ),
    ],
)
def test_fourth_order_scale_uses_one_outside_e5a_and_e5b(
    case: str,
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> None:
    actual = _fourth_order_scale(left, right, step=0)
    assert actual == 1.0, case


@pytest.mark.parametrize(
    ("case", "left", "right", "expected"),
    [
        ("E1", (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), 1.0),
        ("E2", (1.0, 1.0, 0.0), (-2.0, 1.0, 0.0), math.sqrt(2.0)),
        ("E3a", (1.0, 1.0, 0.0), (-4.0, 2.0, 0.0), 0.0),
        ("E3b", (1.0, 1.0, 0.0), (2.0, 1.0, 0.0), math.inf),
        (
            "E4",
            (-4.0, 2.0, 0.0),
            (1.0, 1.0, -1.0),
            math.sqrt(14.0 - 6.0 * math.sqrt(5.0)),
        ),
        ("E5a", (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), math.sqrt(2.0)),
        (
            "E5b",
            (1.0, 1.0, 0.0),
            (-2.0, 1.0, 1.0),
            math.sqrt(3.0 * math.sqrt(2.0) - 4.0),
        ),
        ("E5c", (-4.0, 3.0, 0.0), (1.0, 1.0, -5.0), math.sqrt(2.0)),
        ("E6a", (1.0, 0.0, 1.0), (1.0, 0.0, 0.0), 1.0),
        ("E6b(i)", (-4.0, 0.0, 0.0), (-4.0, 2.0, -3.0), 0.0),
        ("E6b(ii)", (0.0, 0.0, 3.0), (1.0, 1.0, 0.0), 0.0),
        (
            "E6c (interior)",
            (-4.0, 1.0, 0.0),
            (1.0, 1.0, -20.0),
            math.sqrt(14.0),
        ),
        (
            "E6c (zero boundary)",
            (1.0, 1.0, 10.0),
            (1.0, 1.0, 0.0),
            0.0,
        ),
        (
            "E6c (infinite boundary)",
            (-4.0, 0.0, 0.0),
            (-4.0, 1.0, 1.0),
            math.inf,
        ),
    ],
)
def test_defect_minimizing_scale_follows_the_current_classification(
    case: str,
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    expected: float,
) -> None:
    actual = _defect_minimizing_scale(left, right, step=0)
    if math.isinf(expected):
        assert actual == expected, case
    else:
        assert actual == pytest.approx(
            expected,
            rel=3e-15,
            abs=0.0,
        ), case


@pytest.mark.parametrize(
    ("alpha", "beta", "expected"),
    [
        (2.0, 4.0, 8.0 / 3.0),
        (-2.0, -4.0, -8.0 / 3.0),
        (2.0, -4.0, 0.0),
        (0.0, 4.0, 0.0),
        (
            np.finfo(np.float64).max,
            np.finfo(np.float64).max,
            np.finfo(np.float64).max,
        ),
    ],
)
def test_defect_minimizing_zero_scale_uses_the_harmonic_limit(
    alpha: float,
    beta: float,
    expected: float,
) -> None:
    assert _defect_minimizing_mean(alpha, beta, 0.0) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("alpha", "beta", "expected"),
    [
        (2.0, 4.0, 3.0),
        (-2.0, -4.0, -3.0),
        (
            np.finfo(np.float64).max,
            np.finfo(np.float64).max,
            np.finfo(np.float64).max,
        ),
        (np.finfo(np.float64).max, -np.finfo(np.float64).max, 0.0),
        (
            np.nextafter(0.0, 1.0),
            np.nextafter(0.0, 1.0),
            np.nextafter(0.0, 1.0),
        ),
        (
            np.nextafter(0.0, 1.0),
            2.0 * np.nextafter(0.0, 1.0),
            2.0 * np.nextafter(0.0, 1.0),
        ),
    ],
)
def test_defect_minimizing_infinite_scale_uses_crank_nicolson(
    alpha: float,
    beta: float,
    expected: float,
) -> None:
    assert _defect_minimizing_mean(alpha, beta, math.inf) == expected


def test_third_order_scale_avoids_max_float_residual_overflow() -> None:
    R = np.finfo(np.float64).max
    Q = math.sqrt((2.0 / 3.0) * R)

    assert _third_order_scale((1.0, Q, R), None, step=0) == pytest.approx(
        1.0,
        abs=3e-16,
    )


def test_fourth_order_balances_the_two_endpoint_defects() -> None:
    result = ellipse_scheme(
        lambda t, u: -u,
        0.0,
        0.4,
        1.0,
        n_steps=4,
        fourth_order=True,
        derivatives_of_F=_decay_derivatives,
        atol=1e-13,
        rtol=1e-13,
    )

    for u_n, u_next, sigma in zip(
        result.u[:-1],
        result.u[1:],
        result.sigma,
        strict=True,
    ):
        defect_sum = _defect_for_decay(float(u_n), float(sigma))
        defect_sum += _defect_for_decay(float(u_next), float(sigma))
        assert defect_sum == pytest.approx(0.0, rel=0.0, abs=5e-11)


def test_fourth_order_default_numerical_derivatives_balance_defects() -> None:
    result = ellipse_scheme(
        lambda t, u: -u,
        0.0,
        0.1,
        1.0,
        n_steps=1,
        fourth_order=True,
        atol=1e-13,
        rtol=1e-13,
    )

    defect_sum = _defect_for_decay(float(result.u[0]), result.sigma[0])
    defect_sum += _defect_for_decay(float(result.u[1]), result.sigma[0])
    assert defect_sum == pytest.approx(0.0, rel=0.0, abs=2e-5)


@pytest.mark.parametrize(
    "mode",
    ["third_order", "fourth_order", "minimize_defect"],
)
def test_automatic_modes_use_one_for_an_all_scale_cancelling_case(
    mode: str,
) -> None:
    result = ellipse_scheme(
        lambda t, u: 2.0,
        0.0,
        0.3,
        -1.0,
        n_steps=3,
        **{mode: True},
    )

    np.testing.assert_allclose(result.u, -1.0 + 2.0 * result.t)
    np.testing.assert_array_equal(result.sigma, np.ones(3))


@pytest.mark.parametrize(
    "mode",
    ["third_order", "fourth_order", "minimize_defect"],
)
def test_default_derivative_radius_is_not_limited_by_a_tiny_ode_step(
    mode: str,
) -> None:
    result = ellipse_scheme(
        lambda t, u: -u,
        0.0,
        1e-8,
        1.0,
        n_steps=1,
        **{mode: True},
    )

    assert result.sigma[0] == pytest.approx(math.sqrt(2.0), rel=2e-5)
    assert result.u[1] == pytest.approx(math.exp(-1e-8), rel=0.0, abs=2e-16)


@pytest.mark.parametrize(
    ("mode", "minimum_order"),
    [("third_order", 2.9), ("fourth_order", 3.9)],
)
def test_default_derivative_radius_is_time_translation_invariant(
    mode: str,
    minimum_order: float,
) -> None:
    shift = 1000.0

    def F(t: float, u: float) -> float:
        local_t = t - shift
        return 3.0 - math.sin(local_t) - math.cos(local_t)

    exact = -2.0 + 0.3 + math.cos(0.1) - math.sin(0.1)
    errors = []
    for n_steps in (5, 10, 20):
        result = ellipse_scheme(
            F,
            shift,
            shift + 0.1,
            -1.0,
            n_steps=n_steps,
            atol=1e-14,
            rtol=1e-14,
            max_iter=200,
            **{mode: True},
        )
        errors.append(abs(result.u[-1] - exact))

    observed_orders = [
        math.log2(errors[index] / errors[index + 1]) for index in (0, 1)
    ]
    assert min(observed_orders) > minimum_order


def test_fourth_order_recovers_when_euler_endpoint_is_outside_e5a_and_e5b(
) -> None:
    result = ellipse_scheme(
        lambda t, u: -u,
        0.0,
        1.0,
        1.0,
        n_steps=1,
        fourth_order=True,
        derivatives_of_F=_decay_derivatives,
        atol=1e-13,
        rtol=1e-13,
        max_iter=200,
    )

    assert result.u[1] == pytest.approx(0.357783068978, abs=2e-12)
    assert result.sigma[0] == pytest.approx(1.17029793038, abs=2e-12)
    defect_sum = _defect_for_decay(float(result.u[0]), result.sigma[0])
    defect_sum += _defect_for_decay(float(result.u[1]), result.sigma[0])
    assert defect_sum == pytest.approx(0.0, abs=2e-12)


@pytest.mark.parametrize(
    ("mode", "minimum_order"),
    [("third_order", 2.9), ("fourth_order", 3.9)],
)
def test_default_numerical_derivatives_recover_the_requested_order(
    mode: str,
    minimum_order: float,
) -> None:
    errors = []
    for n_steps in (10, 20, 40):
        result = ellipse_scheme(
            lambda t, u: -u,
            0.0,
            1.0,
            1.0,
            n_steps=n_steps,
            atol=1e-14,
            rtol=1e-14,
            max_iter=200,
            **{mode: True},
        )
        errors.append(abs(result.u[-1] - math.exp(-1.0)))

    assert errors[0] > errors[1] > errors[2] > 0.0
    observed_orders = [
        math.log2(errors[index] / errors[index + 1]) for index in (0, 1)
    ]
    assert min(observed_orders) > minimum_order


def test_third_order_reports_when_no_positive_cancelling_scale_exists(
) -> None:
    def F(t: float, u: float) -> float:
        return u**0.25

    def derivatives_of_F(point: np.ndarray) -> np.ndarray:
        u = float(point[1])
        return np.array([0.25 * u**-0.5, -0.125 * u**-1.25])

    with pytest.raises(
        RuntimeError,
        match=r"no unique positive defect-cancelling scale.*step 0",
    ):
        ellipse_scheme(
            F,
            0.0,
            0.1,
            1.0,
            n_steps=1,
            third_order=True,
            derivatives_of_F=derivatives_of_F,
        )


def test_fourth_order_e6c_infinite_boundary_matches_unit_scale_method(
) -> None:
    def F(t: float, u: float) -> float:
        return u**0.25

    def derivatives_of_F(point: np.ndarray) -> np.ndarray:
        u = float(point[1])
        return np.array([0.25 * u**-0.5, -0.125 * u**-1.25])

    result = ellipse_scheme(
        F,
        0.0,
        0.1,
        1.0,
        n_steps=1,
        fourth_order=True,
        derivatives_of_F=derivatives_of_F,
        atol=1e-13,
        rtol=1e-13,
    )
    unit_scale = ellipse_scheme(
        F,
        0.0,
        0.1,
        1.0,
        n_steps=1,
        sigma_n=1.0,
        atol=1e-13,
        rtol=1e-13,
    )

    np.testing.assert_array_equal(result.sigma, [1.0])
    np.testing.assert_allclose(
        result.u,
        unit_scale.u,
        rtol=5e-15,
        atol=5e-16,
    )


def test_fourth_order_e3a_matches_the_unit_scale_method() -> None:
    # Endpoint data are (F, L_F F, L_F^2 F) = (1, 1, -39) and
    # (-4, 2, 39), which is case E3a and therefore falls back to unit scale.
    def F(t: float, u: float) -> float:
        del u
        return 1.0 + t - 19.5 * t**2 + 14.0 * t**3 - 0.5 * t**4

    def derivatives_of_F(point: np.ndarray) -> np.ndarray:
        t = float(point[0])
        first = 1.0 - 39.0 * t + 42.0 * t**2 - 2.0 * t**3
        second = -39.0 + 84.0 * t - 6.0 * t**2
        return np.array([first, second])

    result = ellipse_scheme(
        F,
        0.0,
        1.0,
        0.0,
        n_steps=1,
        fourth_order=True,
        derivatives_of_F=derivatives_of_F,
    )
    unit_scale = ellipse_scheme(
        F,
        0.0,
        1.0,
        0.0,
        n_steps=1,
        sigma_n=1.0,
    )

    np.testing.assert_array_equal(result.sigma, [1.0])
    np.testing.assert_allclose(
        result.u,
        unit_scale.u,
        rtol=5e-15,
        atol=5e-16,
    )


def test_fourth_order_e4_matches_the_unit_scale_method() -> None:
    # This quintic has endpoint data F=(1, -2), L_F F=(1, 1), and
    # L_F^2 F=(-0.05, -0.05).  The resulting scale equation has two
    # distinct positive roots, so E4 falls back to unit scale.
    def F(t: float, u: float) -> float:
        return (
            1.0
            + t
            - 0.025 * t**2
            - 39.95 * t**3
            + 59.975 * t**4
            - 24.0 * t**5
        )

    def derivatives_of_F(point: np.ndarray) -> np.ndarray:
        t = float(point[0])
        first = (
            1.0
            - 0.05 * t
            - 119.85 * t**2
            + 239.9 * t**3
            - 120.0 * t**4
        )
        second = -0.05 - 239.7 * t + 719.7 * t**2 - 480.0 * t**3
        return np.array([first, second])

    result = ellipse_scheme(
        F,
        0.0,
        1.0,
        0.0,
        n_steps=1,
        fourth_order=True,
        derivatives_of_F=derivatives_of_F,
    )
    unit_scale = ellipse_scheme(
        F,
        0.0,
        1.0,
        0.0,
        n_steps=1,
        sigma_n=1.0,
    )

    np.testing.assert_array_equal(result.sigma, [1.0])
    np.testing.assert_allclose(
        result.u,
        unit_scale.u,
        rtol=5e-15,
        atol=5e-16,
    )


def test_minimize_defect_e6c_infinite_boundary_uses_crank_nicolson(
) -> None:
    def F(t: float, u: float) -> float:
        return u**0.25

    def derivatives_of_F(point: np.ndarray) -> np.ndarray:
        u = float(point[1])
        return np.array([0.25 * u**-0.5, -0.125 * u**-1.25])

    result = ellipse_scheme(
        F,
        0.0,
        0.1,
        1.0,
        n_steps=1,
        minimize_defect=True,
        derivatives_of_F=derivatives_of_F,
        atol=1e-13,
        rtol=1e-13,
    )

    assert result.sigma[0] == math.inf
    assert result.u[1] == pytest.approx(
        1.0 + 0.05 * (F(0.0, 1.0) + F(0.1, result.u[1])),
        rel=1e-11,
        abs=1e-12,
    )


def test_minimize_defect_e3a_uses_the_zero_scale_limit() -> None:
    # Endpoint data are (F, L_F F, L_F^2 F) = (1, 1, -39) and
    # (-4, 2, 39), which is E3a. Opposite endpoint slopes give a zero mean.
    def F(t: float, u: float) -> float:
        del u
        return 1.0 + t - 19.5 * t**2 + 14.0 * t**3 - 0.5 * t**4

    def derivatives_of_F(point: np.ndarray) -> np.ndarray:
        t = float(point[0])
        first = 1.0 - 39.0 * t + 42.0 * t**2 - 2.0 * t**3
        second = -39.0 + 84.0 * t - 6.0 * t**2
        return np.array([first, second])

    result = ellipse_scheme(
        F,
        0.0,
        1.0,
        0.0,
        n_steps=1,
        minimize_defect=True,
        derivatives_of_F=derivatives_of_F,
    )

    np.testing.assert_array_equal(result.u, [0.0, 0.0])
    np.testing.assert_array_equal(result.sigma, [0.0])


def test_minimize_defect_e4_uses_the_smaller_positive_scale() -> None:
    def F(t: float, u: float) -> float:
        return (
            1.0
            + t
            - 0.025 * t**2
            - 39.95 * t**3
            + 59.975 * t**4
            - 24.0 * t**5
        )

    def derivatives_of_F(point: np.ndarray) -> np.ndarray:
        t = float(point[0])
        first = (
            1.0
            - 0.05 * t
            - 119.85 * t**2
            + 239.9 * t**3
            - 120.0 * t**4
        )
        second = -0.05 - 239.7 * t + 719.7 * t**2 - 480.0 * t**3
        return np.array([first, second])

    result = ellipse_scheme(
        F,
        0.0,
        1.0,
        0.0,
        n_steps=1,
        minimize_defect=True,
        derivatives_of_F=derivatives_of_F,
    )

    expected = math.sqrt(12.8 / (2.5 + math.sqrt(3.69)))
    assert result.sigma[0] == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize(
    ("first_mode", "second_mode"),
    [
        ("third_order", "fourth_order"),
        ("third_order", "minimize_defect"),
        ("fourth_order", "minimize_defect"),
    ],
)
def test_automatic_modes_are_mutually_exclusive(
    first_mode: str,
    second_mode: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="mutually exclusive|cannot.*True|only one",
    ):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            1.0,
            1.0,
            n_steps=1,
            **{first_mode: True, second_mode: True},
        )


@pytest.mark.parametrize(
    "mode",
    ["third_order", "fourth_order", "minimize_defect"],
)
@pytest.mark.parametrize("bad_value", [0, 1, 1.0, None, "true"])
def test_automatic_mode_flags_require_booleans(
    mode: str,
    bad_value,
) -> None:
    with pytest.raises(TypeError, match=mode):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            1.0,
            1.0,
            n_steps=1,
            sigma_n=1.0,
            **{mode: bad_value},
        )


def test_base_mode_requires_sigma_n() -> None:
    with pytest.raises(ValueError, match="sigma_n"):
        ellipse_scheme(lambda t, u: -u, 0.0, 1.0, 1.0, n_steps=1)


@pytest.mark.parametrize(
    "mode",
    ["third_order", "fourth_order", "minimize_defect"],
)
def test_automatic_modes_reject_sigma_n(mode: str) -> None:
    with pytest.raises(ValueError, match="sigma_n"):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            1.0,
            1.0,
            n_steps=1,
            sigma_n=1.0,
            **{mode: True},
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"derivatives_of_F": _decay_derivatives},
        {"derivative_step": 1e-4},
    ],
)
def test_base_mode_rejects_automatic_derivative_options(extra: dict) -> None:
    with pytest.raises(
        ValueError,
        match="automatic|third_order|fourth_order|minimize_defect",
    ):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            1.0,
            1.0,
            n_steps=1,
            sigma_n=1.0,
            **extra,
        )


@pytest.mark.parametrize(
    "mode",
    ["third_order", "fourth_order", "minimize_defect"],
)
def test_exact_and_numerical_derivative_options_cannot_be_combined(
    mode: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="derivatives_of_F.*derivative_step|derivative_step.*derivatives_of_F",
    ):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            1.0,
            1.0,
            n_steps=1,
            derivatives_of_F=_decay_derivatives,
            derivative_step=1e-4,
            **{mode: True},
        )


@pytest.mark.parametrize(
    "bad_values",
    [
        np.array(1.0),
        np.array([1.0]),
        np.array([[1.0, -1.0]]),
        np.array([1.0, -1.0, 2.0]),
    ],
)
@pytest.mark.parametrize(
    "mode",
    ["third_order", "fourth_order", "minimize_defect"],
)
def test_derivatives_of_F_requires_a_length_two_vector(
    mode: str,
    bad_values: np.ndarray,
) -> None:
    with pytest.raises((TypeError, ValueError), match="derivatives_of_F"):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            0.1,
            1.0,
            n_steps=1,
            derivatives_of_F=lambda point: bad_values,
            **{mode: True},
        )


@pytest.mark.parametrize(
    "bad_values",
    [np.array([math.nan, -1.0]), np.array([1.0, math.inf])],
)
@pytest.mark.parametrize(
    "mode",
    ["third_order", "fourth_order", "minimize_defect"],
)
def test_derivatives_of_F_requires_finite_values(
    mode: str,
    bad_values: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="derivatives_of_F"):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            0.1,
            1.0,
            n_steps=1,
            derivatives_of_F=lambda point: bad_values,
            **{mode: True},
        )


@pytest.mark.parametrize(
    "mode",
    ["third_order", "fourth_order", "minimize_defect"],
)
def test_derivatives_of_F_must_be_callable(mode: str) -> None:
    with pytest.raises(TypeError, match="derivatives_of_F"):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            0.1,
            1.0,
            n_steps=1,
            derivatives_of_F=1.0,  # type: ignore[arg-type]
            **{mode: True},
        )


@pytest.mark.parametrize(
    "bad_step",
    [0.0, -1.0, math.nan, math.inf, True, "1e-4"],
)
@pytest.mark.parametrize(
    "mode",
    ["third_order", "fourth_order", "minimize_defect"],
)
def test_invalid_derivative_step_is_rejected(mode: str, bad_step) -> None:
    with pytest.raises((TypeError, ValueError), match="derivative_step"):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            0.1,
            1.0,
            n_steps=1,
            derivative_step=bad_step,
            **{mode: True},
        )


@pytest.mark.parametrize(
    "mode",
    ["third_order", "fourth_order", "minimize_defect"],
)
def test_unrepresentable_derivative_step_is_rejected_before_F(
    mode: str,
) -> None:
    calls = 0

    def F(t: float, u: float) -> float:
        nonlocal calls
        calls += 1
        return -u

    with pytest.raises(ValueError, match="derivative_step"):
        ellipse_scheme(
            F,
            1e16,
            1e16 + 4.0,
            1.0,
            n_steps=1,
            derivative_step=1e-8,
            **{mode: True},
        )

    assert calls == 0


@pytest.mark.parametrize("mode", ["fourth_order", "minimize_defect"])
def test_overflowing_full_derivative_radius_is_rejected_before_F(
    mode: str,
) -> None:
    calls = 0

    def F(t: float, u: float) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    with pytest.raises(ValueError, match="derivative_step"):
        ellipse_scheme(
            F,
            1.6e308,
            1.7e308,
            0.0,
            n_steps=1,
            derivative_step=1e307,
            **{mode: True},
        )

    assert calls == 0


@pytest.mark.parametrize(
    "t_0, T",
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, math.inf),
        (math.nan, 1.0),
    ],
)
def test_invalid_time_interval_is_rejected(t_0: float, T: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        ellipse_scheme(
            lambda t, u: 0.0,
            t_0,
            T,
            0.0,
            n_steps=1,
            sigma_n=1.0,
        )


@pytest.mark.parametrize("bad_steps", [0, -1])
def test_nonpositive_step_count_is_rejected(bad_steps: int) -> None:
    with pytest.raises(ValueError, match="n_steps"):
        ellipse_scheme(
            lambda t, u: 0.0,
            0.0,
            1.0,
            0.0,
            n_steps=bad_steps,
            sigma_n=1.0,
        )


@pytest.mark.parametrize("bad_steps", [True, 1.5, "10"])
def test_noninteger_step_count_is_rejected(bad_steps) -> None:
    with pytest.raises(TypeError, match="n_steps"):
        ellipse_scheme(
            lambda t, u: 0.0,
            0.0,
            1.0,
            0.0,
            n_steps=bad_steps,
            sigma_n=1.0,
        )


@pytest.mark.parametrize("bad_sigma", [0.0, -1.0, math.nan, math.inf])
def test_invalid_scalar_scale_is_rejected(bad_sigma: float) -> None:
    with pytest.raises(ValueError, match="sigma"):
        ellipse_scheme(
            lambda t, u: 0.0,
            0.0,
            1.0,
            0.0,
            n_steps=1,
            sigma_n=bad_sigma,
        )


@pytest.mark.parametrize("bad_sigma", [True, "1.0", 1.0 + 0.0j, [1.0]])
def test_invalid_scale_type_is_rejected(bad_sigma) -> None:
    with pytest.raises(TypeError, match="sigma"):
        ellipse_scheme(
            lambda t, u: 0.0,
            0.0,
            1.0,
            0.0,
            n_steps=1,
            sigma_n=bad_sigma,
        )


@pytest.mark.parametrize("bad_value", [0.0, -1.0, math.nan, math.inf])
def test_invalid_scale_callback_output_is_rejected(
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match="sigma"):
        ellipse_scheme(
            lambda t, u: 0.0,
            0.0,
            1.0,
            0.0,
            n_steps=1,
            sigma_n=lambda n, t_n, u_n, h_n: bad_value,
        )


@pytest.mark.parametrize("bad_value", [True, "1.0", 1.0 + 0.0j, [1.0]])
def test_invalid_scale_callback_output_type_is_rejected(bad_value) -> None:
    with pytest.raises(TypeError, match="sigma"):
        ellipse_scheme(
            lambda t, u: 0.0,
            0.0,
            1.0,
            0.0,
            n_steps=1,
            sigma_n=lambda n, t_n, u_n, h_n: bad_value,
        )


@pytest.mark.parametrize(
    "keyword, bad_value",
    [
        ("atol", -1.0),
        ("atol", math.nan),
        ("rtol", -1.0),
        ("rtol", math.inf),
        ("max_iter", 0),
    ],
)
def test_invalid_iteration_controls_are_rejected(
    keyword: str,
    bad_value,
) -> None:
    kwargs = {
        "n_steps": 1,
        "sigma_n": 1.0,
        keyword: bad_value,
    }
    with pytest.raises((TypeError, ValueError), match=keyword):
        ellipse_scheme(lambda t, u: 0.0, 0.0, 1.0, 0.0, **kwargs)


def test_at_least_one_tolerance_must_be_positive() -> None:
    with pytest.raises(ValueError, match="atol and rtol"):
        ellipse_scheme(
            lambda t, u: 0.0,
            0.0,
            1.0,
            0.0,
            n_steps=1,
            sigma_n=1.0,
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize(
    "atol, rtol",
    [(0.0, 1e-10), (1e-12, 0.0)],
)
def test_either_individual_tolerance_may_be_zero(
    atol: float,
    rtol: float,
) -> None:
    result = ellipse_scheme(
        lambda t, u: 0.0,
        0.0,
        1.0,
        0.0,
        n_steps=1,
        sigma_n=1.0,
        atol=atol,
        rtol=rtol,
    )

    assert result.u[-1] == 0.0


@pytest.mark.parametrize("bad_u_0", [math.nan, math.inf, -math.inf])
def test_nonfinite_initial_value_is_rejected(bad_u_0: float) -> None:
    with pytest.raises(ValueError, match="u_0"):
        ellipse_scheme(
            lambda t, u: 0.0,
            0.0,
            1.0,
            bad_u_0,
            n_steps=1,
            sigma_n=1.0,
        )


def test_noncallable_field_is_rejected_before_solving() -> None:
    with pytest.raises(TypeError, match="F"):
        ellipse_scheme(  # type: ignore[arg-type]
            1.0,
            0.0,
            1.0,
            0.0,
            n_steps=1,
            sigma_n=1.0,
        )


def test_callback_errors_are_not_silently_replaced() -> None:
    class CallbackError(Exception):
        pass

    def sigma_n(n: int, t_n: float, u_n: float, h_n: float) -> float:
        raise CallbackError("user scale failed")

    with pytest.raises(CallbackError, match="user scale failed"):
        ellipse_scheme(
            lambda t, u: 0.0,
            0.0,
            1.0,
            0.0,
            n_steps=1,
            sigma_n=sigma_n,
        )


def test_fixed_point_nonconvergence_reports_the_step() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"fixed-point iteration failed to converge at step 0",
    ):
        ellipse_scheme(
            lambda t, u: u,
            0.0,
            1.0,
            1.0,
            n_steps=1,
            sigma_n=1.0,
            atol=1e-15,
            rtol=1e-15,
            max_iter=1,
        )
