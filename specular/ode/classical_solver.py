"""
Classical fixed-step schemes for scalar first-order ODEs.
"""

from typing import Callable

import numpy as np

from .result import ODEResult, _num_steps

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


SUPPORTED_SCHEMES = ["explicit Euler", "implicit Euler", "Crank-Nicolson"]


def explicit_Euler_scheme(
    F: Callable[[float, float], float],
    t_0: float,
    u_0: Callable[[float], float] | float,
    T: float,
    h: float = 1e-6,
) -> ODEResult:
    """
    Explicit Euler scheme.
    """
    t_curr = t_0
    u_curr = u_0(t_0) if callable(u_0) else u_0

    t_history = [t_curr]
    u_history = [u_curr]
    steps = _num_steps(t_0, T, h)

    for _ in tqdm(range(steps), desc="Running the explicit Euler scheme"):
        t_curr, u_curr = t_curr + h, u_curr + h * F(t_curr, u_curr)
        t_history.append(t_curr)
        u_history.append(u_curr)

    return ODEResult(
        scheme="explicit Euler scheme",
        h=h,
        all_history={
            "variables": np.array(t_history),
            "values": np.array(u_history),
        },
    )


def implicit_Euler_scheme(
    F: Callable[[float, float], float],
    t_0: float,
    u_0: Callable[[float], float] | float,
    T: float,
    h: float = 1e-6,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> ODEResult:
    """
    Implicit Euler scheme.
    """
    t_curr = t_0
    u_curr = u_0(t_0) if callable(u_0) else u_0

    t_history = [t_curr]
    u_history = [u_curr]
    steps = _num_steps(t_0, T, h)

    for k in tqdm(range(steps), desc="Running the implicit Euler scheme"):
        t_next = t_curr + h

        u_temp = u_curr + h * F(t_curr, u_curr)
        u_guess = u_temp

        for _ in range(max_iter):
            u_guess = u_curr + h * F(t_next, u_temp)
            if np.linalg.norm(u_guess - u_temp) < tol:
                break
            u_temp = u_guess
        else:
            print(f"Warning: step {k + 1} did not converge.")

        t_curr, u_curr = t_next, u_guess
        t_history.append(t_curr)
        u_history.append(u_curr)

    return ODEResult(
        scheme="implicit Euler scheme",
        h=h,
        all_history={
            "variables": np.array(t_history),
            "values": np.array(u_history),
        },
    )


def Crank_Nicolson_scheme(
    F: Callable[[float, float], float],
    t_0: float,
    u_0: Callable[[float], float] | float,
    T: float,
    h: float = 1e-6,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> ODEResult:
    """
    Crank Nicolson scheme
    """
    t_curr = t_0
    u_curr = u_0(t_0) if callable(u_0) else u_0

    t_history = [t_curr]
    u_history = [u_curr]
    steps = _num_steps(t_0, T, h)

    for k in tqdm(range(steps), desc="Running Crank-Nicolson scheme"):
        t_next = t_curr + h

        F_curr = F(t_curr, u_curr)
        u_temp = u_curr + h * F_curr
        u_guess = u_temp

        for _ in range(max_iter):
            f_guess = F(t_next, u_temp)
            u_guess = u_curr + 0.5 * h * (F_curr + f_guess)

            if np.linalg.norm(u_guess - u_temp) < tol:
                break

            u_temp = u_guess
        else:
            print(f"Warning: step {k + 1} did not converge.")

        t_curr, u_curr = t_next, u_guess
        t_history.append(t_curr)
        u_history.append(u_curr)

    return ODEResult(
        scheme="Crank-Nicolson scheme",
        h=h,
        all_history={
            "variables": np.array(t_history),
            "values": np.array(u_history),
        },
    )


def classical_scheme(
    F: Callable[[float, float], float],
    t_0: float,
    u_0: Callable[[float], float] | float,
    T: float,
    h: float = 1e-6,
    form: str = "explicit Euler",
    tol: float = 1e-12,
    max_iter: int = 100,
) -> ODEResult:
    """
    Solves an initial value problem (IVP) using classical numerical schemes.
    Supported forms: explicit Euler, implicit Euler, and Crank-Nicolson.

    Parameters:
        F (callable):
            The given source function ``F`` in (IVP).
            The calling signature should be ``F(t, u)`` where ``t`` and ``u`` are scalars.
        t_0 (float):
            The starting time of the simulation.
        u_0 (callable):
            The given initial condition ``u_0`` in (IVP).
        T (float):
            The end time of the simulation.
        h (float, optional):
            Mesh size used in the finite difference approximation. Must be positive.
        form (str | optional):
            The form of the numerical scheme. 
            Options: ``'explicit_Euler'``, ``'implicit_Euler'``, ``'Crank-Nicolson'``.
        tol (float | optional):
            Tolerance for fixed-point iteration.
            Used for implicit Euler and Crank-Nicolson schemes.
        max_iter (int | optional):
            Max iterations for fixed-point solver.

    Returns:
        An object containing ``(t, u)`` data and the scheme name.
    """
    if form == "explicit Euler":
        return explicit_Euler_scheme(F, t_0, u_0, T, h)

    if form == "implicit Euler":
        return implicit_Euler_scheme(F, t_0, u_0, T, h, tol, max_iter)

    if form == "Crank-Nicolson":
        return Crank_Nicolson_scheme(F, t_0, u_0, T, h, tol, max_iter)

    raise ValueError(f"Unknown form '{form}'. Supported forms: {SUPPORTED_SCHEMES}")
