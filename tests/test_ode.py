"""Tests for the scalar specular ellipse ODE schemes."""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Callable
from dataclasses import fields

import numpy as np
import pytest

import specular
from specular.backends.numpy.calculation import _C as _unscaled_mean
from specular.ode import (
    ODEResult,
    ellipse_scheme,
    ellipse_scheme_3rd_order,
    ellipse_scheme_4th_order,
)
from specular.ode._solver import _scaled_mean


Scheme = Callable[..., ODEResult]


def test_ode_api_is_available_from_the_top_level_package() -> None:
    assert specular.ODEResult is ODEResult
    assert specular.ellipse_scheme is ellipse_scheme
    assert specular.ellipse_scheme_3rd_order is ellipse_scheme_3rd_order
    assert specular.ellipse_scheme_4th_order is ellipse_scheme_4th_order
    assert [field.name for field in fields(ODEResult)] == ["t", "y", "sigma"]


def test_importing_specular_does_not_import_heavy_example_dependencies() -> None:
    code = (
        "import sys; import specular; "
        "assert 'specular.ode' not in sys.modules; "
        "_ = specular.ellipse_scheme; "
        "assert 'specular.ode' in sys.modules; "
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

    assert result == pytest.approx(
        sigma * (1.0 + math.sqrt(2.0)),
        rel=8 * np.finfo(np.float64).eps,
        abs=0.0,
    )


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


@pytest.mark.parametrize(
    "scheme",
    [ellipse_scheme, ellipse_scheme_3rd_order, ellipse_scheme_4th_order],
)
def test_constant_equation_is_exact_and_result_has_minimal_shapes(
    scheme: Scheme,
) -> None:
    result = scheme(
        lambda t, y: 2.5,
        (1.0, 2.5),
        -3.0,
        n_steps=6,
        sigma=0.75,
    )
    expected_t = np.linspace(1.0, 2.5, 7)

    assert isinstance(result, ODEResult)
    assert result.t.shape == (7,)
    assert result.y.shape == (7,)
    assert result.sigma.shape == (6,)
    np.testing.assert_array_equal(result.t, expected_t)
    np.testing.assert_allclose(
        result.y,
        -3.0 + 2.5 * (expected_t - expected_t[0]),
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_array_equal(result.sigma, np.full(6, 0.75))


@pytest.mark.parametrize(
    "scheme",
    [ellipse_scheme, ellipse_scheme_3rd_order, ellipse_scheme_4th_order],
)
def test_uniform_mesh_reaches_both_endpoints_exactly(scheme: Scheme) -> None:
    result = scheme(
        lambda t, y: 0.0,
        (-0.75, 1.25),
        4.0,
        n_steps=8,
        sigma=1.0,
    )

    assert result.t[0] == -0.75
    assert result.t[-1] == 1.25
    np.testing.assert_allclose(np.diff(result.t), 0.25, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "scheme",
    [ellipse_scheme, ellipse_scheme_3rd_order, ellipse_scheme_4th_order],
)
def test_affine_equation_matches_its_exact_solution(scheme: Scheme) -> None:
    # y' = 1 - y/2, y(0) = -1 has y(t) = 2 - 3 exp(-t/2).
    result = scheme(
        lambda t, y: 1.0 - 0.5 * y,
        (0.0, 1.0),
        -1.0,
        n_steps=100,
        sigma=0.7,
        atol=1e-13,
        rtol=1e-13,
        max_iter=100,
    )
    expected = 2.0 - 3.0 * np.exp(-0.5 * result.t)

    np.testing.assert_allclose(result.y, expected, rtol=0.0, atol=2e-5)


@pytest.mark.parametrize(
    "scheme",
    [ellipse_scheme, ellipse_scheme_3rd_order],
)
def test_left_scale_callback_receives_step_index_state_and_step_size_once(
    scheme: Scheme,
) -> None:
    calls: list[tuple[int, float, float, float]] = []

    def scale(n: int, t: float, y: float, h: float) -> float:
        calls.append((n, t, y, h))
        return (n + 1) * h

    result = scheme(
        lambda t, y: 3.0,
        (2.0, 3.0),
        -1.0,
        n_steps=8,
        sigma=scale,
    )

    assert len(calls) == 8
    assert [call[0] for call in calls] == list(range(8))
    np.testing.assert_allclose([call[1] for call in calls], result.t[:-1])
    np.testing.assert_allclose([call[2] for call in calls], result.y[:-1])
    np.testing.assert_array_equal([call[3] for call in calls], np.full(8, 0.125))
    np.testing.assert_allclose(result.sigma, 0.125 * np.arange(1.0, 9.0))


def test_base_and_third_order_entry_points_share_the_same_update() -> None:
    def field(t: float, y: float) -> float:
        return math.cos(t) + 0.1 * y

    def scale(n: int, t: float, y: float, h: float) -> float:
        return 0.5 + 0.01 * n + 0.1 * h + 0.001 * abs(y)

    kwargs = {
        "n_steps": 20,
        "sigma": scale,
        "atol": 1e-13,
        "rtol": 1e-13,
        "max_iter": 100,
    }
    base = ellipse_scheme(field, (0.0, 1.0), 0.25, **kwargs)
    third = ellipse_scheme_3rd_order(field, (0.0, 1.0), 0.25, **kwargs)

    np.testing.assert_array_equal(third.t, base.t)
    np.testing.assert_array_equal(third.y, base.y)
    np.testing.assert_array_equal(third.sigma, base.sigma)


def test_fourth_order_scale_callback_uses_each_trial_right_state() -> None:
    calls: list[tuple[int, float, float, float, float, float]] = []

    def scale(
        n: int,
        t_left: float,
        y_left: float,
        t_right: float,
        y_trial: float,
        h: float,
    ) -> float:
        calls.append((n, t_left, y_left, t_right, y_trial, h))
        return 1.0 + 0.05 * abs(y_trial)

    result = ellipse_scheme_4th_order(
        lambda t, y: 0.25 * y,
        (0.0, 1.0),
        1.0,
        n_steps=4,
        sigma=scale,
        atol=1e-13,
        rtol=1e-13,
        max_iter=100,
    )

    assert {call[0] for call in calls} == set(range(4))
    for n, t_left, y_left, t_right, y_trial, h in calls:
        assert t_left == result.t[n]
        assert y_left == result.y[n]
        assert t_right == result.t[n + 1]
        assert math.isfinite(y_trial)
        assert h == 0.25

    # The recorded scale is the one belonging to the accepted right endpoint.
    np.testing.assert_allclose(
        result.sigma,
        1.0 + 0.05 * np.abs(result.y[1:]),
        rtol=0.0,
        atol=2e-13,
    )


def test_fourth_order_scalar_scale_degenerates_to_the_base_update() -> None:
    def field(t: float, y: float) -> float:
        return t - 0.2 * y

    kwargs = {
        "n_steps": 12,
        "sigma": 0.8,
        "atol": 1e-13,
        "rtol": 1e-13,
        "max_iter": 100,
    }
    base = ellipse_scheme(field, (-0.25, 0.75), 0.5, **kwargs)
    fourth = ellipse_scheme_4th_order(
        field,
        (-0.25, 0.75),
        0.5,
        **kwargs,
    )

    np.testing.assert_array_equal(fourth.t, base.t)
    np.testing.assert_array_equal(fourth.y, base.y)
    np.testing.assert_array_equal(fourth.sigma, base.sigma)


def test_mesh_dependent_scale_reproduces_fourth_order_inverse_ode_family(
) -> None:
    # For y' = 1/y and sigma_n = h, the manuscript proves order 2 + 2 = 4.
    errors = []
    for n_steps in (20, 40, 80):
        result = ellipse_scheme(
            lambda t, y: 1.0 / y,
            (0.0, 1.0),
            1.0,
            n_steps=n_steps,
            sigma=lambda n, t, y, h: h,
            atol=1e-14,
            rtol=1e-14,
            max_iter=100,
        )
        errors.append(abs(result.y[-1] - math.sqrt(3.0)))

    assert errors[0] > errors[1] > errors[2] > 0.0
    observed_orders = [
        math.log2(errors[index] / errors[index + 1]) for index in (0, 1)
    ]
    assert min(observed_orders) > 3.7


@pytest.mark.parametrize(
    "bad_t_span",
    [
        (0.0,),
        (0.0, 1.0, 2.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, math.inf),
        (math.nan, 1.0),
    ],
)
def test_invalid_time_span_is_rejected(bad_t_span) -> None:
    with pytest.raises((TypeError, ValueError)):
        ellipse_scheme(
            lambda t, y: 0.0,
            bad_t_span,
            0.0,
            n_steps=1,
            sigma=1.0,
        )


@pytest.mark.parametrize("bad_steps", [0, -1])
def test_nonpositive_step_count_is_rejected(bad_steps: int) -> None:
    with pytest.raises(ValueError, match="n_steps"):
        ellipse_scheme(
            lambda t, y: 0.0,
            (0.0, 1.0),
            0.0,
            n_steps=bad_steps,
            sigma=1.0,
        )


@pytest.mark.parametrize("bad_steps", [True, 1.5, "10"])
def test_noninteger_step_count_is_rejected(bad_steps) -> None:
    with pytest.raises(TypeError, match="n_steps"):
        ellipse_scheme(
            lambda t, y: 0.0,
            (0.0, 1.0),
            0.0,
            n_steps=bad_steps,
            sigma=1.0,
        )


@pytest.mark.parametrize("bad_sigma", [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize(
    "scheme",
    [ellipse_scheme, ellipse_scheme_3rd_order, ellipse_scheme_4th_order],
)
def test_invalid_scalar_scale_is_rejected(
    scheme: Scheme,
    bad_sigma: float,
) -> None:
    with pytest.raises(ValueError, match="sigma"):
        scheme(
            lambda t, y: 0.0,
            (0.0, 1.0),
            0.0,
            n_steps=1,
            sigma=bad_sigma,
        )


@pytest.mark.parametrize("bad_sigma", [True, "1.0", 1.0 + 0.0j, [1.0]])
def test_invalid_scale_type_is_rejected(bad_sigma) -> None:
    with pytest.raises(TypeError, match="sigma"):
        ellipse_scheme(
            lambda t, y: 0.0,
            (0.0, 1.0),
            0.0,
            n_steps=1,
            sigma=bad_sigma,
        )


@pytest.mark.parametrize("bad_value", [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize(
    "scheme, scale",
    [
        (ellipse_scheme, lambda bad: lambda n, t, y, h: bad),
        (ellipse_scheme_3rd_order, lambda bad: lambda n, t, y, h: bad),
        (
            ellipse_scheme_4th_order,
            lambda bad: lambda n, tl, yl, tr, yr, h: bad,
        ),
    ],
)
def test_invalid_scale_callback_output_is_rejected(
    scheme: Scheme,
    scale,
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match="sigma"):
        scheme(
            lambda t, y: 0.0,
            (0.0, 1.0),
            0.0,
            n_steps=1,
            sigma=scale(bad_value),
        )


@pytest.mark.parametrize("bad_value", [True, "1.0", 1.0 + 0.0j, [1.0]])
def test_invalid_scale_callback_output_type_is_rejected(bad_value) -> None:
    with pytest.raises(TypeError, match="sigma"):
        ellipse_scheme(
            lambda t, y: 0.0,
            (0.0, 1.0),
            0.0,
            n_steps=1,
            sigma=lambda n, t, y, h: bad_value,
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
        "sigma": 1.0,
        keyword: bad_value,
    }
    with pytest.raises((TypeError, ValueError), match=keyword):
        ellipse_scheme(lambda t, y: 0.0, (0.0, 1.0), 0.0, **kwargs)


def test_at_least_one_tolerance_must_be_positive() -> None:
    with pytest.raises(ValueError, match="atol and rtol"):
        ellipse_scheme(
            lambda t, y: 0.0,
            (0.0, 1.0),
            0.0,
            n_steps=1,
            sigma=1.0,
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
        lambda t, y: 0.0,
        (0.0, 1.0),
        0.0,
        n_steps=1,
        sigma=1.0,
        atol=atol,
        rtol=rtol,
    )

    assert result.y[-1] == 0.0


@pytest.mark.parametrize("bad_y0", [math.nan, math.inf, -math.inf])
def test_nonfinite_initial_value_is_rejected(bad_y0: float) -> None:
    with pytest.raises(ValueError, match="y0"):
        ellipse_scheme(
            lambda t, y: 0.0,
            (0.0, 1.0),
            bad_y0,
            n_steps=1,
            sigma=1.0,
        )


def test_noncallable_field_is_rejected_before_solving() -> None:
    with pytest.raises(TypeError, match="fun"):
        ellipse_scheme(  # type: ignore[arg-type]
            1.0,
            (0.0, 1.0),
            0.0,
            n_steps=1,
            sigma=1.0,
        )


def test_callback_errors_are_not_silently_replaced() -> None:
    class CallbackError(Exception):
        pass

    def scale(n: int, t: float, y: float, h: float) -> float:
        raise CallbackError("user scale failed")

    with pytest.raises(CallbackError, match="user scale failed"):
        ellipse_scheme(
            lambda t, y: 0.0,
            (0.0, 1.0),
            0.0,
            n_steps=1,
            sigma=scale,
        )


def test_fixed_point_nonconvergence_reports_the_step() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"fixed-point iteration failed to converge at step 0",
    ):
        ellipse_scheme(
            lambda t, y: y,
            (0.0, 1.0),
            1.0,
            n_steps=1,
            sigma=1.0,
            atol=1e-15,
            rtol=1e-15,
            max_iter=1,
        )


def test_time_span_input_is_not_mutated() -> None:
    t_span = np.array([-1.0, 2.0])
    original = t_span.copy()

    result = ellipse_scheme(
        lambda t, y: 0.0,
        t_span,
        1.0,
        n_steps=3,
        sigma=1.0,
    )

    np.testing.assert_array_equal(t_span, original)
    assert not np.shares_memory(result.t, t_span)
