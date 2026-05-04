import numpy as np
import pytest

from specular.optimization.line_search import LineSearch


def quadratic(x):
    x_arr = np.asarray(x, dtype=float)
    return float(np.sum(x_arr**2))


def quadratic_gradient(x):
    return 2.0 * np.asarray(x, dtype=float)


def test_line_search_requires_explicit_name():
    """Test that LineSearch does not silently default to Armijo."""
    with pytest.raises(ValueError) as exc_info:
        LineSearch()

    message = str(exc_info.value)
    assert "Line search name is required" in message
    assert "exact" in message
    assert "armijo" in message
    assert "wolfe" in message
    assert "strong_wolfe" in message


def test_line_search_rejects_invalid_names():
    with pytest.raises(ValueError, match="Invalid line search 'scipy'"):
        LineSearch("scipy")

    with pytest.raises(ValueError, match="Invalid line search 'armijio'"):
        LineSearch("armijio")


def test_line_search_normalizes_strong_wolfe_aliases():
    assert LineSearch("strong wolfe").name == "strong_wolfe"
    assert LineSearch("strong-wolfe").name == "strong_wolfe"
    assert LineSearch("STRONG_WOLFE").name == "strong_wolfe"


def test_line_search_validates_parameters():
    with pytest.raises(ValueError, match="alpha_0 must be positive"):
        LineSearch("armijo", alpha_0=0.0)

    with pytest.raises(ValueError, match="c_1 must satisfy"):
        LineSearch("armijo", c_1=0.0)

    with pytest.raises(ValueError, match="c_2 must satisfy"):
        LineSearch("armijo", c_2=1.0)

    with pytest.raises(ValueError, match=r"requires 0 < c_1 < c_2 < 1"):
        LineSearch("wolfe", c_1=0.9, c_2=0.1)

    with pytest.raises(ValueError, match="rho must satisfy"):
        LineSearch("armijo", rho=1.0)

    with pytest.raises(ValueError, match="max_iter must be positive"):
        LineSearch("armijo", max_iter=0)

    with pytest.raises(ValueError, match="max_alpha must be positive"):
        LineSearch("armijo", max_alpha=0.0)


def test_armijo_accepts_full_step_for_quadratic():
    rule = LineSearch("armijo")
    x = np.array([1.0, 0.0])
    gradient = quadratic_gradient(x)
    direction = -0.5 * gradient

    alpha = rule(
        f=quadratic,
        x=x,
        direction=direction,
        gradient_current=gradient,
    )

    assert alpha == pytest.approx(1.0)


def test_armijo_backtracks_until_condition_holds():
    rule = LineSearch("armijo", alpha_0=4.0, rho=0.5)
    x = np.array([1.0])
    gradient = quadratic_gradient(x)
    direction = -gradient

    alpha = rule(
        f=quadratic,
        x=x,
        direction=direction,
        gradient_current=gradient,
    )

    assert alpha == pytest.approx(0.5)


def test_wolfe_requires_gradient_function():
    rule = LineSearch("wolfe")
    x = np.array([1.0])
    gradient = quadratic_gradient(x)

    with pytest.raises(ValueError, match="requires gradient_f"):
        rule(
            f=quadratic,
            x=x,
            direction=-gradient,
            gradient_current=gradient,
        )


def test_wolfe_accepts_step_satisfying_conditions():
    rule = LineSearch("wolfe")
    x = np.array([1.0, 0.0])
    gradient = quadratic_gradient(x)
    direction = -0.5 * gradient

    alpha = rule(
        f=quadratic,
        x=x,
        direction=direction,
        gradient_current=gradient,
        gradient_f=quadratic_gradient,
    )

    f_current = quadratic(x)
    f_trial = quadratic(x + alpha * direction)
    initial_slope = float(np.dot(gradient, direction))
    trial_slope = float(np.dot(quadratic_gradient(x + alpha * direction), direction))

    assert rule.satisfies_wolfe(f_trial, f_current, alpha, initial_slope, trial_slope)


def test_strong_wolfe_accepts_step_satisfying_conditions():
    rule = LineSearch("strong_wolfe")
    x = np.array([1.0, 0.0])
    gradient = quadratic_gradient(x)
    direction = -0.5 * gradient

    alpha = rule(
        f=quadratic,
        x=x,
        direction=direction,
        gradient_current=gradient,
        gradient_f=quadratic_gradient,
    )

    f_current = quadratic(x)
    f_trial = quadratic(x + alpha * direction)
    initial_slope = float(np.dot(gradient, direction))
    trial_slope = float(np.dot(quadratic_gradient(x + alpha * direction), direction))

    assert rule.satisfies_strong_wolfe(
        f_trial,
        f_current,
        alpha,
        initial_slope,
        trial_slope,
    )


def test_line_search_rejects_non_descent_direction():
    rule = LineSearch("armijo")
    x = np.array([1.0])
    gradient = quadratic_gradient(x)

    with pytest.raises(ValueError, match="requires a descent direction"):
        rule(
            f=quadratic,
            x=x,
            direction=gradient,
            gradient_current=gradient,
        )


def test_line_search_rejects_shape_mismatch():
    rule = LineSearch("armijo")

    with pytest.raises(ValueError, match="Shape mismatch"):
        rule(
            f=quadratic,
            x=np.array([1.0, 2.0]),
            direction=np.array([-1.0]),
            gradient_current=np.array([2.0, 4.0]),
        )


def test_line_search_raises_when_failure_is_strict():
    rule = LineSearch("armijo", alpha_0=1.0, rho=0.5, max_iter=1, raise_on_fail=True)
    x = np.array([1.0])
    gradient = quadratic_gradient(x)
    direction = -10.0 * gradient

    with pytest.raises(RuntimeError, match="failed to satisfy"):
        rule(
            f=quadratic,
            x=x,
            direction=direction,
            gradient_current=gradient,
        )
