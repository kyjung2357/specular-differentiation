"""Check exact tracing of elliptic solution graphs."""

from __future__ import annotations

import numpy as np

import specular

from _common import maximum_error


def solve_case(a: float, b: float, T: float, n_steps: int) -> None:
    """Compare the fitted scale with the unit scale on one upper arc."""

    def F(t: float, u: float) -> float:
        return -(b * b * t) / (a * a * u)

    def exact(t: np.ndarray) -> np.ndarray:
        return b * np.sqrt(1.0 - (t / a) ** 2)

    fitted = specular.ellipse_scheme(
        F,
        0.0,
        T,
        b,
        n_steps=n_steps,
        sigma_n=b / a,
        atol=1e-14,
        rtol=1e-14,
        max_iter=500,
    )
    unit_scale = specular.ellipse_scheme(
        F,
        0.0,
        T,
        b,
        n_steps=n_steps,
        sigma_n=1.0,
        atol=1e-14,
        rtol=1e-14,
        max_iter=500,
    )

    fitted_error = maximum_error(fitted.t, fitted.u, exact)
    unit_error = maximum_error(unit_scale.t, unit_scale.u, exact)
    ellipse_residual = float(
        np.max(np.abs((fitted.t / a) ** 2 + (fitted.u / b) ** 2 - 1.0))
    )

    print(
        f"a={a:g}, b={b:g}, h={T / n_steps:g}: "
        f"fitted error={fitted_error:.3e}, "
        f"unit-scale error={unit_error:.3e}, "
        f"ellipse residual={ellipse_residual:.3e}"
    )


def main() -> None:
    print("Elliptic exact-tracing calibration")
    solve_case(a=1.0, b=1.0, T=0.8, n_steps=4)
    solve_case(a=2.3, b=1.2, T=2.0, n_steps=4)


if __name__ == "__main__":
    main()
