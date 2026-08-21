"""Test mesh-dependent scales for u' = 1/u."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import specular

from _common import maximum_error, observed_order


STEP_COUNTS = (40, 80, 160, 320)
type StepScale = float | Callable[[int, float, float, float], float]


def exact(t: np.ndarray) -> np.ndarray:
    return np.sqrt(1.0 + 2.0 * t)


def mesh_scale(power: float) -> StepScale:
    def sigma_n(n: int, t_n: float, u_n: float, h_n: float) -> float:
        del n, t_n, u_n
        return h_n**power

    return sigma_n


def solve(sigma_n: StepScale, n_steps: int) -> float:

    result = specular.ellipse_scheme(
        lambda t, u: 1.0 / u,
        0.0,
        1.0,
        1.0,
        n_steps=n_steps,
        sigma_n=sigma_n,
        atol=1e-14,
        rtol=1e-14,
        max_iter=200,
    )
    return maximum_error(result.t, result.u, exact)


def main() -> None:
    methods = (
        ("sigma=1", 1.0),
        ("sigma=0.1", 0.1),
        ("sigma=0.01", 0.01),
        ("sigma=h^(1/4)", mesh_scale(0.25)),
        ("sigma=h^(1/2)", mesh_scale(0.5)),
        ("sigma=h", mesh_scale(1.0)),
    )

    print("u' = 1/u on [0, 1]")
    print(f"{'scale':<18} {'error at N=320':>18} {'observed order':>16}")
    for label, sigma_n in methods:
        errors = [solve(sigma_n, n_steps) for n_steps in STEP_COUNTS]
        order = observed_order(errors[-2], errors[-1])
        print(f"{label:<18} {errors[-1]:18.8e} {order:16.6f}")


if __name__ == "__main__":
    main()
