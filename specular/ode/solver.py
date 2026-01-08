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
from ..calculation import _A_scalar as A

SUPPORTED_SCHEMES = ['explicit Euler', 'implicit Euler', 'Crank-Nicolson']

def classical_scheme(
    F: Callable[[float, float], float], 
    t_0: float, 
    u_0: Callable[[float], float] | float,
    T: float, 
    h: float = 1e-6,
    form: str = 'explicit Euler',
    tol: float = 1e-6, 
    max_iter: int = 100
) -> ODEResult:
    """
    Solves an initial value problem (IVP) using classical numerical schemes.
    Supported forms: explicit Euler, implicit Euler, and Crank-Nicolson.

    Parameters
    ----------
    F : callable
        The given source function ``F`` in (IVP).
        The calling signature should be ``F(t, u)`` where ``t`` and ``u`` are scalars.
    t_0 : float
        The starting time of the simulation.
    u_0 : callable
        The given initial condition ``u_0`` in (IVP).
    T : float
        The end time of the simulation.
    h : float, optional
        Mesh size used in the finite difference approximation. Must be positive.
        Default: ``1e-6``.
    form : str | optional
        The form of the numerical scheme. 
        Options: ``'explicit_Euler'``, ``'implicit_Euler'``, ``'Crank-Nicolson'``.
        Default: ``'explicit_Euler'``.
    tol : float | optional
        Tolerance for fixed-point iteration.
        Used for implicit Euler and Crank-Nicolson schemes.
    max_iter : int | optional
        Max iterations for fixed-point solver.

    Returns
    -------
    ODEResult
        An object containing ``(t, u)`` data and the scheme name.
    """
    t_curr = t_0
    u_curr = u_0(t_0) if callable(u_0) else u_0 

    all_history = {}
    t_history = [t_curr]
    u_history = [u_curr]

    steps = int((T - t_0) / h)
    
    if form == "explicit Euler":
        for _ in tqdm(range(steps), desc="Running the explicit Euler scheme"):
            t_curr, u_curr = t_curr + h, u_curr + h*F(t_curr, u_curr) # type: ignore

            t_history.append(t_curr) 
            u_history.append(u_curr) 

    elif form == "implicit Euler":
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
    
    elif form == "Crank-Nicolson":
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
            
    else:
        raise ValueError(f"Unknown form '{form}'. Supported forms: {SUPPORTED_SCHEMES}")
    
    all_history["variables"] = np.array(t_history)
    all_history["values"] = np.array(u_history)

    return ODEResult(
        scheme= form + " scheme",
        h=h,
        all_history=all_history
    )

def Euler_scheme(
    of_Type: int | str,
    F: Callable[[float, float], float],
    t_0: float, 
    u_0: Callable[[float], float] | float,
    T: float, 
    h: float = 1e-6,
    u_1: Callable[[float], float] | float | bool = False,
    tol: float = 1e-6, 
    zero_tol: float = 1e-8,
    max_iter: int = 100
) -> ODEResult:
    """
    Solves an initial value problem (IVP) using the specular Euler scheme of Type 1, 2, 3, 4, 5, and 6.

    Parameters
    ----------
    of_Type : int | str
        The type of the specular Euler scheme.
        Options: ``1``, ``'1'``, ``2``, ``'2'``, ``3``, ``'3'``, ``4``, ``'4'``, ``5``, ``'5'``, ``6``, ``'6'``.
    F : callable
        The given source function ``F`` in (IVP).
        The calling signature should be ``F(t, u)`` where ``t`` and ``u`` are scalars.
    t_0 : float
        The starting time of the simulation.
    u_0 : callable
        The given initial condition ``u_0`` in (IVP).
    T : float
        The end time of the simulation.
    h : float, optional
        Mesh size used in the finite difference approximation. Must be positive.
        Default: ``1e-6``.
    u_1 : callable | float | bool
        The numerical solution at the time ``t_1 = t_0 + h`` for Types 1, 2, and 3.
        If a float or callable is provided, it is used as the exact value.
        If False, the explicit Euler scheme is applied.
        Default: ``False``.
    tol : float | optional
        Tolerance for fixed-point iteration
        Used for Types 3, 4, 5, and 6.
    zero_tol : float | np.floating
        A small threshold used to determine if the denominator (alpha + beta) is close to zero for numerical stability.
        Default: ``1e-6``.
    max_iter : int | optional
        Max iterations for fixed-point solver.

    Returns
    -------
    ODEResult
        An object containing ``(t, u)`` data and the scheme name.
    """
    Type = str(of_Type)

    scheme = 'specular Euler scheme of Type ' + Type
    steps = int((T - t_0) / h)

    all_history = {}
    t_history = []
    u_history = []

    if Type in ['1', '2', '3']:
        t_prev = t_0
        u_prev = u_0(t_0) if callable(u_0) else u_0

        t_history.append(t_prev)
        u_history.append(u_prev)

        t_curr = t_prev + h

        if u_1 == False:
            # explicit Euler to get u_1
            u_curr = u_prev + h * F(t_prev, u_prev)
        else:
            u_curr = u_1
        
        t_history.append(t_curr)
        u_history.append(u_curr)

        if Type == '1':
            _of_Type_1(F, h, zero_tol, steps, t_prev, t_curr, u_prev, u_curr, t_history, u_history)

        elif Type == '2':
            _of_Type_2(F, h, zero_tol, steps, t_prev, t_curr, u_prev, u_curr, t_history, u_history)

        elif Type == '3':
            _of_Type_3(F, h, tol, zero_tol, max_iter, steps, t_prev, t_curr, u_prev, u_curr, t_history, u_history)

    elif Type in ['4', '5', '6']:
        t_curr = t_0
        u_curr = u_0(t_0) if callable(u_0) else u_0  

        t_history.append(t_curr)
        u_history.append(u_curr)

        if Type == '4':
            _of_Type_4(F, h, tol, zero_tol, max_iter, steps, t_curr, u_curr, t_history, u_history)

        elif Type == '5':
            _of_Type_5(F, h, tol, zero_tol, max_iter, steps, t_curr, u_curr, t_history, u_history)

        elif Type == '6':
            _of_Type_6(F, h, tol, zero_tol, max_iter, steps, t_curr, u_curr, t_history, u_history)

    else:
        raise ValueError(f"Unknown type. Got {of_Type}. Supported types: '1', '2', '3', '4', '5', and '6'")

    all_history["variables"] = np.array(t_history)
    all_history["values"] = np.array(u_history)

    return ODEResult(
        scheme=scheme,
        h=h,
        all_history=all_history
    )

def _of_Type_1(F, h, zero_tol, steps, t_prev, t_curr, u_prev, u_curr, t_history, u_history):
    """
    Implements the specular Euler scheme of Type 1.
    """
    for _ in tqdm(range(steps - 1), desc="Running the specular Euler scheme of Type 1"):
        t_next = t_curr + h
        u_next = u_curr + h * A(F(t_curr, u_curr), F(t_prev, u_prev), zero_tol=zero_tol)

        # Update for next step
        t_prev, u_prev = t_curr, u_curr
        t_curr, u_curr = t_next, u_next

        t_history.append(t_curr)
        u_history.append(u_curr)

def _of_Type_2(F, h, zero_tol, steps, t_prev, t_curr, u_prev, u_curr, t_history, u_history):
    """
    Implements the specular Euler scheme of Type 2.
    """
    for _ in tqdm(range(steps - 1), desc="Running the specular Euler scheme of Type 2"):
        t_next = t_curr + h
        u_next = u_curr + h * A(F(t_curr, u_curr), (u_curr - u_prev)/h, zero_tol=zero_tol)

        # Update for next step
        t_prev, u_prev = t_curr, u_curr
        t_curr, u_curr = t_next, u_next

        t_history.append(t_curr)
        u_history.append(u_curr)

def _of_Type_3(F, h, tol, zero_tol, max_iter, steps, t_prev, t_curr, u_prev, u_curr, t_history, u_history):
    """
    Implements the specular Euler scheme of Type 3.
    """
    for k in tqdm(range(steps - 1), desc="Running the specular Euler scheme of Type 3"):
        t_next = t_curr + h

        # Initial guess: explicit Euler
        u_temp = u_curr + h * F(t_curr, u_curr) 
        u_guess = u_temp

        # fixed second argument
        beta = F(t_prev, u_prev)
        
        # Fixed-point iteration
        for _ in range(max_iter):
            alpha = (u_temp - u_curr) / h
            u_guess = u_curr + h * A(alpha, beta, zero_tol=zero_tol)

            if abs(u_guess - u_temp) < tol:
                break

            u_temp = u_guess
        else:
            print(f"Warning: fixed-point iteration did not converge at step {k+1}")

        # Update for next step
        t_prev, u_prev = t_curr, u_curr
        t_curr, u_curr = t_next, u_guess 

        t_history.append(t_curr)
        u_history.append(u_curr)

def _of_Type_4(F, h, tol, zero_tol, max_iter, steps, t_curr, u_curr, t_history, u_history):
    """
    Implements the specular Euler scheme of Type 4.
    """
    for k in tqdm(range(steps), desc="Running the specular Euler scheme of Type 4"):
        t_next = t_curr + h

        # Initial guess: explicit Euler
        u_temp = u_curr + h * F(t_curr, u_curr)
        u_guess = u_temp
        
        beta = F(t_curr, u_curr)  # fixed second argument

        # Fixed-point iteration
        for _ in range(max_iter):
            alpha = (u_temp - u_curr) / h
            u_guess = u_curr + h * A(alpha, beta, zero_tol=zero_tol) 

            if abs(u_guess - u_temp) < tol:
                break

            u_temp = u_guess
        else:
            print(f"Warning: fixed-point iteration did not converge at step {k+1}")

        # Update for next step
        t_curr, u_curr = t_next, u_guess  

        t_history.append(t_curr)
        u_history.append(u_curr)

def _of_Type_5(F, h, tol, zero_tol, max_iter, steps, t_curr, u_curr, t_history, u_history):
    """
    Implements the specular Euler scheme of Type 5.
    """
    for k in tqdm(range(steps), desc="Running the specular Euler scheme of Type 5"):
        beta = F(t_curr, u_curr)  # fixed second argument
        t_curr = t_curr + h

        # Initial guess: explicit Euler
        u_temp = u_curr + h * beta 
        u_guess = u_temp

        # Fixed-point iteration
        for _ in range(max_iter):
            alpha = F(t_curr, u_temp)
            u_guess = u_curr + h * A(alpha, beta, zero_tol=zero_tol)

            if abs(u_guess - u_temp) < tol:
                break

            u_temp = u_guess
        else:
            print(f"Warning: fixed-point iteration did not converge at step {k+1}")

        # Update for next step
        u_curr = u_guess    

        t_history.append(t_curr)
        u_history.append(u_curr)

def _of_Type_6(F, h, tol, zero_tol, max_iter, steps, t_curr, u_curr, t_history, u_history):
    """
    Implements the specular Euler scheme of Type 6.
    """
    for k in tqdm(range(steps), desc="Running the specular Euler scheme of Type 6"):
        t_next = t_curr + h

        # Initial guess: explicit Euler
        u_temp = u_curr + h * F(t_curr, u_curr)
        u_guess = u_temp

        # Fixed-point iteration
        for _ in range(max_iter):
            alpha = F(t_next, u_temp)
            beta = (u_temp - u_curr) / h

            u_guess = u_curr + h * A(alpha, beta, zero_tol=zero_tol)

            if abs(u_guess - u_temp) < tol:
                break
            
            u_temp = u_guess
        else:
            print(f"Warning: fixed-point iteration did not converge at step {k+1}")

        # Update for next step
        t_curr, u_curr = t_next, u_guess

        t_history.append(t_curr)
        u_history.append(u_curr)

def trigonometric_scheme(
    F: Callable[[float], float],
    t_0: float, 
    u_0: Callable[[float], float] | float,
    u_1: Callable[[float], float] | float,
    T: float, 
    h: float = 1e-6
) -> ODEResult:
    """
    Solves an initial value problem (IVP) using the specular trigonometric scheme.

    Parameters
    ----------
    F : callable
        The given source function ``F`` in (IVP).
        The calling signature should be ``F(t, u)`` where ``t`` and ``u`` are scalars.
    t_0 : float
        The starting time of the simulation.
    u_0 : callable
        The given initial condition ``u_0`` in (IVP).
    u_1 : callable | float | bool
        The numerical solution at the time ``t_1 = t_0 + h`` for Types 1, 2, and 3.
        If a float or callable is provided, it is used as the exact value.
        If False, the explicit Euler scheme is applied.
        Default: ``False``.
    T : float
        The end time of the simulation.
    h : float, optional
        Mesh size used in the finite difference approximation. Must be positive.
        Default: ``1e-6``.

    Returns
    -------
    ODEResult
        An object containing ``(t, u)`` data and the scheme name.
    """
    t_prev = t_0
    u_prev = u_0(t_0) if callable(u_0) else u_0

    t_curr = t_0 + h
    u_curr = u_1(t_curr) if callable(u_1) else u_1

    all_history = {}
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

    all_history["variables"] = np.array(t_history)
    all_history["values"] = np.array(u_history)

    return ODEResult(
        scheme="specular trigonometric scheme",
        h=h,
        all_history=all_history
    )

