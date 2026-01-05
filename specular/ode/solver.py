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
    scheme: str = 'explicit Euler',
    tol: float = 1e-6, 
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
        Options: 'explicit_Euler', 'implicit_Euler', 'Crank-Nicolson'.
        Default is 'explicit_Euler'.
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

def Euler_scheme(
    F: Callable[[float, float], float], 
    t_0: float, 
    u_0: Callable[[float], float] | float,
    T: float, 
    h: float = 1e-6,
    of_Type: str = '1',
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

    if of_Type not in ['1', '2', '3', '4', '5', '6']:
        raise ValueError(f"Unknown type '{of_Type}'. Supported forms: '1', '2', '3', '4', '5', and '6'")
    
    scheme = 'specular Euler scheme of Type ' + of_Type
    steps = int((T - t_0) / h)

    t_history = []
    u_history = []

    if of_Type in ['1', '2', '3']:
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

        if of_Type == '1':
            for _ in tqdm(range(steps - 1), desc="Running the specular Euler scheme of Type 1"):
                t_next = t_curr + h
                u_next = u_curr + h * A(F(t_curr, u_curr), F(t_prev, u_prev), zero_tol=zero_tol)    # type: ignore

                # Update for next step
                t_prev, u_prev = t_curr, u_curr
                t_curr, u_curr = t_next, u_next

                t_history.append(t_curr)
                u_history.append(u_curr)

        elif of_Type == '2':
            for _ in tqdm(range(steps - 1), desc="Running the specular Euler scheme of Type 2"):
                t_next = t_curr + h
                u_next = u_curr + h * A(F(t_curr, u_curr), (u_curr - u_prev)/h, zero_tol=zero_tol) # type: ignore

                # Update for next step
                t_prev, u_prev = t_curr, u_curr
                t_curr, u_curr = t_next, u_next

                t_history.append(t_curr)
                u_history.append(u_curr)

        elif of_Type == '3':
            for k in tqdm(range(steps - 1), desc="Running the specular Euler scheme of Type 3"):
                t_next = t_curr + h

                # Initial guess: explicit Euler
                u_temp = u_curr + h * F(t_curr, u_curr) # type: ignore
                u_guess = u_temp

                beta = F(t_prev, u_prev)  # type: ignore # fixed second argument
                
                # Fixed-point iteration
                for _ in range(max_iter):
                    alpha = (u_temp - u_curr) / h # type: ignore
                    u_guess = u_curr + h * A(alpha, beta, zero_tol=zero_tol)  # type: ignore

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

    elif of_Type in ['4', '5', '6']:
        t_curr = t_0
        u_curr = u_0(t_0) if callable(u_0) else u_0  

        t_history.append(t_curr)
        u_history.append(u_curr)

        if of_Type == '4':
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

        elif of_Type == '5':
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

        elif of_Type == '6':
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

