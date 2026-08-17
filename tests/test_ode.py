"""Tests for the scalar specular ellipse ODE scheme."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import fields

import numpy as np
import pytest

import specular
from specular.backends.numpy.calculation import _C as _unscaled_mean
from specular.ode import ODEResult, ellipse_scheme
from specular.ode._solver import (
    _fourth_order_scale,
    _numeric_derivatives_of_F,
    _scaled_mean,
    _third_order_scale,
)


def test_ode_api_is_available_from_the_top_level_package() -> None:
    assert specular.ODEResult is ODEResult
    assert specular.ellipse_scheme is ellipse_scheme
    assert "ellipse_scheme_3rd_order" not in specular.__all__
    assert "ellipse_scheme_4th_order" not in specular.__all__
    assert not hasattr(specular, "ellipse_scheme_3rd_order")
    assert not hasattr(specular, "ellipse_scheme_4th_order")
    assert not hasattr(sys.modules["specular.ode"], "ellipse_scheme_3rd_order")
    assert not hasattr(sys.modules["specular.ode"], "ellipse_scheme_4th_order")
    assert [field.name for field in fields(ODEResult)] == ["t", "u", "sigma"]


def test_importing_specular_does_not_import_heavy_example_dependencies() -> None:
    code = (
        "import sys; import specular; "
        "assert 'specular.ode' not in sys.modules; "
        "_ = specular.ellipse_scheme; "
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


def test_scaled_mean_preserves_a_tiny_scale_beside_a_huge_slope() -> None:
    sigma = 1e-200
    result = _scaled_mean(1e200, sigma, sigma)
    reversed_result = _scaled_mean(sigma, 1e200, sigma)

    assert result == pytest.approx(
        sigma * (1.0 + math.sqrt(2.0)),
        rel=8 * np.finfo(np.float64).eps,
        abs=0.0,
    )
    assert reversed_result == result


def test_scaled_mean_keeps_tiny_same_sign_slopes_when_sigma_dominates(
) -> None:
    alpha = 1e-200
    beta = 2e-200
    sigma = 1e200

    forward = _scaled_mean(alpha, beta, sigma)
    reverse = _scaled_mean(beta, alpha, sigma)

    assert forward == pytest.approx(1.5e-200, rel=0.0, abs=5e-216)
    assert reverse == forward


def test_scaled_mean_does_not_drop_a_subnormal_same_sign_contribution(
) -> None:
    tiny = np.nextafter(0.0, 1.0)
    sigma = np.finfo(np.float64).max

    forward = _scaled_mean(tiny, 2.0 * tiny, sigma)
    reverse = _scaled_mean(2.0 * tiny, tiny, sigma)

    assert forward == 2.0 * tiny
    assert reverse == forward


def test_scaled_mean_handles_an_overflowing_same_sign_radius() -> None:
    maximum = np.finfo(np.float64).max
    expected = maximum * (math.sqrt(2.0) - 1.0)

    forward = _scaled_mean(maximum, 1.0, maximum)
    reverse = _scaled_mean(1.0, maximum, maximum)

    assert math.isfinite(forward)
    assert forward == pytest.approx(
        expected,
        rel=8 * np.finfo(np.float64).eps,
        abs=0.0,
    )
    assert reverse == forward


def test_scaled_mean_stays_finite_between_same_sign_maximal_slopes() -> None:
    alpha = np.finfo(np.float64).max
    beta = alpha / 2.0
    result = _scaled_mean(alpha, beta, 1.0)

    assert math.isfinite(result)
    assert beta < result < alpha


def test_scaled_mean_resolves_a_near_antidiagonal_residual() -> None:
    beta = -np.nextafter(1.0, 0.0)
    result = _scaled_mean(1.0, beta, 1.0)

    assert result > 0.0
    assert result == pytest.approx(
        np.finfo(np.float64).eps / 8.0,
        rel=8 * np.finfo(np.float64).eps,
        abs=0.0,
    )


def test_scaled_mean_matches_the_defining_identity_on_moderate_data() -> None:
    rng = np.random.default_rng(20260813)
    alpha = rng.uniform(-10.0, 10.0, size=256)
    beta = rng.uniform(-10.0, 10.0, size=256)
    sigma = 10.0 ** rng.uniform(-2.0, 2.0, size=256)

    actual = np.array(
        [
            _scaled_mean(float(a), float(b), float(s))
            for a, b, s in zip(alpha, beta, sigma, strict=True)
        ]
    )
    expected = sigma * np.asarray(_unscaled_mean(alpha / sigma, beta / sigma))
    swapped = np.array(
        [
            _scaled_mean(float(b), float(a), float(s))
            for a, b, s in zip(alpha, beta, sigma, strict=True)
        ]
    )

    tolerance = 64 * np.finfo(np.float64).eps
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(swapped, actual, rtol=tolerance, atol=tolerance)
    assert np.all(actual >= np.minimum(alpha, beta))
    assert np.all(actual <= np.maximum(alpha, beta))


def test_constant_equation_is_exact_and_result_has_minimal_shapes() -> None:
    result = ellipse_scheme(
        lambda t, u: 2.5,
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

    assert _fourth_order_scale(left, right, None, step=0) == 1.0


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


@pytest.mark.parametrize("mode", ["third_order", "fourth_order"])
def test_automatic_modes_continue_an_all_scale_cancelling_branch(
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


@pytest.mark.parametrize("mode", ["third_order", "fourth_order"])
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


def test_fourth_order_recovers_when_the_euler_endpoint_is_not_e5() -> None:
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


def test_fourth_order_reports_when_no_positive_balancing_scale_exists(
) -> None:
    def F(t: float, u: float) -> float:
        return u**0.25

    def derivatives_of_F(point: np.ndarray) -> np.ndarray:
        u = float(point[1])
        return np.array([0.25 * u**-0.5, -0.125 * u**-1.25])

    with pytest.raises(
        RuntimeError,
        match=r"no unique positive defect-balancing scale.*step 0",
    ):
        ellipse_scheme(
            F,
            0.0,
            0.1,
            1.0,
            n_steps=1,
            fourth_order=True,
            derivatives_of_F=derivatives_of_F,
        )


def test_fourth_order_rejects_two_positive_balancing_branches() -> None:
    # This quintic has endpoint data F=(1, -2), L_F F=(1, 1), and
    # L_F^2 F=(-0.05, -0.05).  The resulting scale equation has two
    # distinct positive roots, so no branch is determined by the method.
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

    with pytest.raises(
        RuntimeError,
        match=r"ambiguous.*two positive.*step 0",
    ):
        ellipse_scheme(
            F,
            0.0,
            1.0,
            0.0,
            n_steps=1,
            fourth_order=True,
            derivatives_of_F=derivatives_of_F,
        )


def test_high_order_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="third_order.*fourth_order"):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            1.0,
            1.0,
            n_steps=1,
            third_order=True,
            fourth_order=True,
        )


def test_base_mode_requires_sigma_n() -> None:
    with pytest.raises(ValueError, match="sigma_n"):
        ellipse_scheme(lambda t, u: -u, 0.0, 1.0, 1.0, n_steps=1)


@pytest.mark.parametrize("mode", ["third_order", "fourth_order"])
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
    with pytest.raises(ValueError, match="third_order|fourth_order"):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            1.0,
            1.0,
            n_steps=1,
            sigma_n=1.0,
            **extra,
        )


@pytest.mark.parametrize("mode", ["third_order", "fourth_order"])
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
def test_derivatives_of_F_requires_a_length_two_vector(
    bad_values: np.ndarray,
) -> None:
    with pytest.raises((TypeError, ValueError), match="derivatives_of_F"):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            0.1,
            1.0,
            n_steps=1,
            third_order=True,
            derivatives_of_F=lambda point: bad_values,
        )


@pytest.mark.parametrize(
    "bad_values",
    [np.array([math.nan, -1.0]), np.array([1.0, math.inf])],
)
def test_derivatives_of_F_requires_finite_values(
    bad_values: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="derivatives_of_F"):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            0.1,
            1.0,
            n_steps=1,
            third_order=True,
            derivatives_of_F=lambda point: bad_values,
        )


def test_derivatives_of_F_must_be_callable() -> None:
    with pytest.raises(TypeError, match="derivatives_of_F"):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            0.1,
            1.0,
            n_steps=1,
            third_order=True,
            derivatives_of_F=1.0,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "bad_step",
    [0.0, -1.0, math.nan, math.inf, True, "1e-4"],
)
def test_invalid_derivative_step_is_rejected(bad_step) -> None:
    with pytest.raises((TypeError, ValueError), match="derivative_step"):
        ellipse_scheme(
            lambda t, u: -u,
            0.0,
            0.1,
            1.0,
            n_steps=1,
            third_order=True,
            derivative_step=bad_step,
        )


def test_unrepresentable_derivative_step_is_rejected_before_F() -> None:
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
            third_order=True,
            derivative_step=1e-8,
        )

    assert calls == 0


def test_overflowing_full_derivative_radius_is_rejected_before_F() -> None:
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
            fourth_order=True,
            derivative_step=1e307,
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
