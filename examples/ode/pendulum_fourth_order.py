"""Test fourth-order scale selection on normalized pendulum branches."""

from __future__ import annotations

import math

import numpy as np

import specular

from _common import maximum_error, observed_order, rk4


# The first step is case E5(i); subsequent steps are on the E5(ii) branch.
X_LEFT = 0.0
X_RIGHT = 0.8
STEP_COUNTS = (80, 160, 320)


def exact_solution(amplitude: float, x: np.ndarray) -> np.ndarray:
    left = np.sin(0.5 * amplitude * (1.0 + x))
    right = np.sin(0.5 * amplitude * (1.0 - x))
    return (2.0 / amplitude) * np.sqrt(np.maximum(left * right, 0.0))


def field(amplitude: float):
    def F(x: float, y: float) -> float:
        return -math.sin(amplitude * x) / (amplitude * y)

    return F


def field_derivatives(amplitude: float):
    def derivatives(point: np.ndarray) -> np.ndarray:
        x, y = map(float, point)
        s = math.sin(amplitude * x)
        c = math.cos(amplitude * x)
        first = -c / y - s * s / (amplitude**2 * y**3)
        second = (
            amplitude * s / y
            - 3.0 * s * c / (amplitude * y**3)
            - 3.0 * s**3 / (amplitude**3 * y**5)
        )
        return np.array([first, second])

    return derivatives


def run_se(amplitude: float, n_steps: int) -> tuple[float, float, float]:
    exact = lambda x: exact_solution(amplitude, x)
    y_0 = float(exact(np.array(X_LEFT)))
    # The discretization error reaches about 1e-13 at the finest mesh. A very
    # tight implicit tolerance keeps iteration error below that level while
    # remaining above the float64 fixed-point resolution floor.
    result = specular.ellipse_scheme(
        field(amplitude),
        X_LEFT,
        X_RIGHT,
        y_0,
        n_steps=n_steps,
        fourth_order=True,
        derivatives_of_F=field_derivatives(amplitude),
        atol=0.0,
        rtol=2e-16,
        max_iter=2000,
    )
    return (
        maximum_error(result.t, result.u, exact),
        float(np.min(result.sigma)),
        float(np.max(result.sigma)),
    )


def run_rk4(amplitude: float, n_steps: int) -> float:
    exact = lambda x: exact_solution(amplitude, x)
    y_0 = float(exact(np.array(X_LEFT)))
    t, y = rk4(field(amplitude), X_LEFT, X_RIGHT, y_0, n_steps)
    return maximum_error(t, y, exact)


def main() -> None:
    print("Normalized pendulum: fourth-order minimizing scale")
    print(
        f"{'A':>5} {'SE error':>14} {'SE order':>10} {'RK4 error':>14} "
        f"{'RK4 order':>10} {'RK4/SE':>10} {'sigma range':>23}"
    )

    for amplitude in (1.0, 0.5, 0.25, 0.1):
        se_runs = [run_se(amplitude, n_steps) for n_steps in STEP_COUNTS]
        rk_errors = [run_rk4(amplitude, n_steps) for n_steps in STEP_COUNTS]
        se_error = se_runs[-1][0]
        se_order = observed_order(se_runs[-2][0], se_error)
        rk_order = observed_order(rk_errors[-2], rk_errors[-1])
        sigma_range = f"[{se_runs[-1][1]:.6f}, {se_runs[-1][2]:.6f}]"
        print(
            f"{amplitude:5.2f} {se_error:14.6e} {se_order:10.6f} "
            f"{rk_errors[-1]:14.6e} {rk_order:10.6f} "
            f"{rk_errors[-1] / se_error:10.4f} {sigma_range:>23}"
        )


if __name__ == "__main__":
    main()
