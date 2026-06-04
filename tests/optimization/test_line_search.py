import numpy as np
import pytest

from specular.optimization.step_size.line_search import LineSearch, LineSearchError


def quadratic(x):
    x_arr = np.asarray(x, dtype=float)
    return float(np.sum(x_arr**2))


def quadratic_gradient(x):
    return 2.0 * np.asarray(x, dtype=float)


def test_line_search_requires_explicit_name():
    """Test that LineSearch does not silently default to Armijo."""
    with pytest.raises(TypeError) as exc_info:
        LineSearch()

    message = str(exc_info.value)
    assert "missing 1 required positional argument: 'name'" in message or "Unknown line search" in message

def test_line_search_rejects_invalid_names():
    with pytest.raises(ValueError, match="Unknown line search 'scipy'"):
        LineSearch("scipy")

    with pytest.raises(ValueError, match="Unknown line search 'armijio'"):
        LineSearch("armijio")

def test_line_search_validates_parameters():
    with pytest.raises(ValueError, match="t_0 must be positive"):
        LineSearch("Armijo", t_0=0.0)

    with pytest.raises(ValueError, match="c_1 must be in"):
        LineSearch("Armijo", c_1=0.0)

    with pytest.raises(ValueError, match="rho must be in"):
        LineSearch("Armijo", rho=1.0)

    with pytest.raises(ValueError, match="max_iter must be positive"):
        LineSearch("Armijo", max_iter=0)

    with pytest.raises(ValueError, match="max_alpha must be positive"):
        LineSearch("Armijo", max_alpha=0.0)


def test_armijo_accepts_full_step_for_quadratic():
    rule = LineSearch("Armijo")
    x = np.array([1.0, 0.0])
    gradient = quadratic_gradient(x)
    direction = -0.5 * gradient

    alpha = rule(
        1,
        f=quadratic,
        x=x,
        d_k=direction,
        grad=gradient,
    )

    assert alpha == pytest.approx(1.0)


def test_armijo_backtracks_until_condition_holds():
    rule = LineSearch("Armijo", t_0=4.0, rho=0.5)
    x = np.array([1.0])
    gradient = quadratic_gradient(x)
    direction = -gradient

    alpha = rule(
        1,
        f=quadratic,
        x=x,
        d_k=direction,
        grad=gradient,
    )

    assert alpha == pytest.approx(0.5)


def test_wolfe_requires_gradient_function():
    with pytest.raises(ValueError, match="requires objective function 'f' to compute gradients"):
        LineSearch("Wolfe")

def test_wolfe_accepts_step_satisfying_conditions():
    rule = LineSearch("Wolfe", f=quadratic)
    x = np.array([1.0, 0.0])
    gradient = quadratic_gradient(x)
    direction = -0.5 * gradient

    alpha = rule(
        1,
        f=quadratic,
        x=x,
        d_k=direction,
        grad=gradient,
    )

    f_current = quadratic(x)
    f_trial = quadratic(x + alpha * direction)
    initial_slope = float(np.dot(gradient, direction))
    trial_slope = float(np.dot(quadratic_gradient(x + alpha * direction), direction))

    assert rule._satisfies_wolfe(f_trial, f_current, alpha, initial_slope, trial_slope)


def test_strong_wolfe_accepts_step_satisfying_conditions():
    rule = LineSearch("strong_Wolfe", f=quadratic)
    x = np.array([1.0, 0.0])
    gradient = quadratic_gradient(x)
    direction = -0.5 * gradient

    alpha = rule(
        1,
        f=quadratic,
        x=x,
        d_k=direction,
        grad=gradient,
    )

    f_current = quadratic(x)
    f_trial = quadratic(x + alpha * direction)
    initial_slope = float(np.dot(gradient, direction))
    trial_slope = float(np.dot(quadratic_gradient(x + alpha * direction), direction))

    assert rule._satisfies_strong_wolfe(
        f_trial,
        f_current,
        alpha,
        initial_slope,
        trial_slope,
    )


def test_line_search_rejects_non_descent_direction():
    rule = LineSearch("Armijo")
    x = np.array([1.0])
    gradient = quadratic_gradient(x)

    with pytest.raises(ValueError, match="requires a descent direction"):
        rule(
            1,
            f=quadratic,
            x=x,
            d_k=gradient,
            grad=gradient,
        )


def test_line_search_raises_when_failure_is_strict():
    rule = LineSearch("Armijo", t_0=1.0, rho=0.5, max_iter=1, skip_on_fail=False)
    x = np.array([1.0])
    gradient = quadratic_gradient(x)
    direction = -10.0 * gradient

    with pytest.raises(LineSearchError, match="failed to satisfy"):
        rule(
            1,
            f=quadratic,
            x=x,
            d_k=direction,
            grad=gradient,
        )
