"""Small helpers shared by the scalar ODE examples."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


type ScalarField = Callable[[float, float], float]
type ExactSolution = Callable[[np.ndarray], np.ndarray]


def maximum_error(
    t: np.ndarray,
    u: np.ndarray,
    exact: ExactSolution,
) -> float:
    """Return the maximum nodal error."""

    exact_values = np.asarray(exact(t), dtype=np.float64)
    return float(np.max(np.abs(u - exact_values)))


def observed_order(coarse_error: float, fine_error: float) -> float:
    """Return the order observed after halving the step size."""

    return math.log2(coarse_error / fine_error)


def crank_nicolson(
    F: ScalarField,
    t_0: float,
    T: float,
    u_0: float,
    n_steps: int,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the implicit Crank--Nicolson method by fixed-point iteration."""

    t = np.linspace(t_0, T, n_steps + 1)
    u = np.empty(n_steps + 1, dtype=np.float64)
    u[0] = u_0

    for n in range(n_steps):
        h = float(t[n + 1] - t[n])
        t_n = float(t[n])
        t_next = float(t[n + 1])
        u_n = float(u[n])
        F_n = F(t_n, u_n)
        u_next = u_n + h * F_n

        for _ in range(max_iter):
            candidate = u_n + 0.5 * h * (F_n + F(t_next, u_next))
            if math.isclose(candidate, u_next, rel_tol=rtol, abs_tol=atol):
                u[n + 1] = candidate
                break
            u_next = candidate
        else:
            raise RuntimeError(
                "Crank--Nicolson iteration failed to converge "
                f"at step {n} after {max_iter} iterations"
            )

    return t, u


def rk3(
    F: ScalarField,
    t_0: float,
    T: float,
    u_0: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Kutta's explicit third-order Runge--Kutta method."""

    t = np.linspace(t_0, T, n_steps + 1)
    u = np.empty(n_steps + 1, dtype=np.float64)
    u[0] = u_0

    for n in range(n_steps):
        h = float(t[n + 1] - t[n])
        t_n = float(t[n])
        u_n = float(u[n])
        k_1 = F(t_n, u_n)
        k_2 = F(t_n + 0.5 * h, u_n + 0.5 * h * k_1)
        k_3 = F(t_n + h, u_n - h * k_1 + 2.0 * h * k_2)
        u[n + 1] = u_n + (h / 6.0) * (k_1 + 4.0 * k_2 + k_3)

    return t, u


def rk4(
    F: ScalarField,
    t_0: float,
    T: float,
    u_0: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the classical explicit fourth-order Runge--Kutta method."""

    t = np.linspace(t_0, T, n_steps + 1)
    u = np.empty(n_steps + 1, dtype=np.float64)
    u[0] = u_0

    for n in range(n_steps):
        h = float(t[n + 1] - t[n])
        t_n = float(t[n])
        u_n = float(u[n])
        k_1 = F(t_n, u_n)
        k_2 = F(t_n + 0.5 * h, u_n + 0.5 * h * k_1)
        k_3 = F(t_n + 0.5 * h, u_n + 0.5 * h * k_2)
        k_4 = F(t_n + h, u_n + h * k_3)
        u[n + 1] = u_n + (h / 6.0) * (
            k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4
        )

    return t, u
