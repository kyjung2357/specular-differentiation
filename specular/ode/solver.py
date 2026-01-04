"""
==============================================================
Numerical methods for solving ordinary differential equations
==============================================================

Let the source function F:[t_0, T]xR -> R be given, and the initial data u_0:R -> R be given. 
Consider the initial value problem:
(IVP)              u'(t) = F(t, u(t))
with the initial condition u(t_0) = u_0(t_0).
To solve (IVP) numerically, this module provides implementations of the specular Euler schemes, the Crank-Nicolson scheme, and the specular trigonometric scheme.
"""

import math
import numpy as np
from tqdm import tqdm 
from typing import Optional, Callable, Tuple, List
from .result import ODEResult
from .. import calculation

SUPPORTED_SCHEMES = ["explicit Euler", "implicit Euler", "Crank-Nicolson"]

def classical_scheme(
    F: Callable[[float, float], float], 
    t_0: float, 
    u_0: Callable[[float], float] | float,
    T: float, 
    h: float = 1e-6,
    scheme: str = "explicit Euler",
    u_1: Callable[[float], float] | float | bool = False,
    tol: float = 1e-6, 
    zero_tol: float = 1e-8,
    max_iter: int = 100
) -> ODEResult:
    """
    Solves an initial value problem (IVP) using classical numerical schemes.
    Supported forms: explicit Euler, implicit Euler, and Crank-Nicolson.

    Parameters
    ----------
    F : callable
        The given source function F in (IVP).
    t_0 : float
        The starting time of the simulation.
    u_0 : callable
        The given initial condition u_0 in (IVP).
    T : float
        The end time of the simulation.
    h : float, optional
        The step size. 
        Default is 1e-6.
    form : str, optional
        The form of the numerical scheme. 
        Options: "explicit_Euler", "implicit_Euler", "Crank-Nicolson".
        Default is "explicit_Euler".
    tol : float, optional
        Tolerance for fixed-point iteration (implicit-Euler/Crank-Nicolson).
    max_iter : int, optional
        Max iterations for fixed-point solver.

    Returns
    -------
    ODEResult
        An object containing (t, u) data and the scheme name.
    """

    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"Unknown form '{scheme}'. Supported forms: {SUPPORTED_SCHEMES}")
    
    t_curr = t_0
    u_curr = u_0(t_0) if callable(u_0) else u_0 

    t_history = [t_curr]
    u_history = [u_curr]

    steps = int((T - t_0) / h)
    
    if scheme == "explicit Euler":
        for _ in tqdm(range(steps), desc="Running the explicit Euler scheme"):
            t_curr, u_curr = t_curr + h, u_curr + h*F(t_curr, u_curr) # type: ignore

            t_history.append(t_curr)
            u_history.append(u_curr)

    elif scheme == "implicit Euler":
        for k in tqdm(range(steps), desc="Running the implicit Euler scheme"):
            t_next = t_curr + h

            # Initial guess: explicit Euler 
            u_temp = u_curr + h*F(t_curr, u_curr) # type: ignore
            u_guess = u_temp

            # Fixed-point iteration
            for _ in range(max_iter):
                u_guess = u_curr + h*F(t_next, u_temp)
                if np.linalg.norm(u_guess - u_temp) < tol:
                    break
                u_temp = u_guess
            else:
                print(f"Warning: step {k+1} did not converge.")

            t_curr, u_curr = t_next, u_guess  
            t_history.append(t_curr)
            u_history.append(u_curr)
    
    elif scheme == "Crank-Nicolson":
        for k in tqdm(range(steps), desc="Running Crank-Nicolson scheme"):
            t_next = t_curr + h

            F_curr = F(t_curr, u_curr)

            # Initial guess: explicit Euler
            u_temp = u_curr + h * F_curr
            u_guess = u_temp

            # Fixed-point iteration
            for _ in range(max_iter):
                f_guess = F(t_next, u_temp)
                u_guess = u_curr + 0.5 * h * (F_curr + f_guess)

                if np.linalg.norm(u_guess - u_temp) < tol:
                    break

                u_temp = u_guess
            else:
                print(f"Warning: step {k+1} did not converge.")

            t_curr, u_curr = t_next, u_guess  
            t_history.append(t_curr)
            u_history.append(u_curr)

    return ODEResult(
        time_grid=np.array(t_history), 
        numerical_sol=np.array(u_history), 
        scheme=scheme
    )

def trigonometric_scheme(
    F: Callable[[float], float], 
    t_0: float, 
    u_0: Callable[[float], float] | float,
    u_1: Callable[[float], float] | float,
    T: float, 
    h: float = 1e-6
) -> ODEResult:

    t_prev = t_0
    u_prev = u_0(t_0) if callable(u_0) else u_0 

    t_curr = t_0 + h
    u_curr = u_1(t_curr) if callable(u_1) else u_1

    t_history = [t_prev, t_curr]
    u_history = [u_prev, u_curr]

    steps = int((T - t_0) / h)

    for m in tqdm(range(steps - 1), desc="Running specular trigonometric scheme"):
        t_next = t_curr + h
        u_next = u_curr + h*math.tan(2*math.atan(F(t_curr, u_curr)) - math.atan((u_curr - u_prev) / h)) # type: ignore

        t_history.append(t_next)
        u_history.append(u_next)

        t_prev, u_prev = t_curr, u_curr
        t_curr, u_curr = t_next, u_next

    return ODEResult(
        time_grid=np.array(t_history), 
        numerical_sol=np.array(u_history), 
        scheme="specular trigonometric"
    )

