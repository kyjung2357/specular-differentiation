"""
==============================================================
Numerical methods for solving ordinary differential equations
==============================================================

Let the source function F:[t_0, T]xR -> R be given, and the initial data u_0:R -> R be given. 
Consider the initial value problem:
(IVP)              u'(t) = F(t, u(t))
with the initial condition u(t_0) = u_0(t_0).
To solve (IVP) numerically, this module provides implementations of the specular Euler schemes, the Crank-Nicolson scheme, and the specular trigonometric scheme.

==========================
__author__ = "Kiyuob Jung"
__version__ = "1.0.0"
__license__ = "MIT"
"""


import numpy as np
import pandas as pd
import specular_derivative as sd
from tqdm import tqdm 
from typing import Optional, Callable, Tuple, List

# save path setting
import os
if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists('tables'):
    os.makedirs('tables')


def Explicit_Euler_Scheme(F: Callable[[float, np.ndarray], np.ndarray], 
                          u_0: Callable[[float], np.ndarray], 
                          t_0: float, 
                          T: float, 
                          h: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem (IVP), using the explicit Euler method.

    Parameters
    ----------
    F : callable
        The given source function F in (IVP).
    u_0 : callable
        The given initial condition u_0 in (IVP).
        The initial condition is determined by calling u_0(t_0).
    t_0 : float
        The starting time of the simulation.
    T : float
        The end time of the simulation.
    h : float, optional
        The step size for the integration. Default is 1e-6.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - t_history: An array of time points from the simulation.
        - u_history: An array of the approximated solution u(t) at each corresponding time point.
    """
    t_curr = t_0
    u_curr = u_0(t_0)

    t_history = [t_curr]
    u_history = [u_curr]

    steps = int((T - t_0) / h)

    for _ in tqdm(range(steps), desc="Running Explicit Euler scheme"):
        t_curr, u_curr = t_curr + h, u_curr + h*F(t_curr, u_curr)

        t_history.append(t_curr)
        u_history.append(u_curr)

    return np.array(t_history), np.array(u_history)


def Implicit_Euler_Scheme(F: Callable[[float, np.ndarray], np.ndarray], 
                          u_0: Callable[[float], np.ndarray], 
                          t_0: float, 
                          T: float, 
                          h: float = 1e-6, 
                          tol: float = 1e-6, 
                          max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem (IVP), using the implicit Euler method.

    Parameters
    ----------
    F : callable
        The given source function F in (IVP).
    u_0 : callable
        The given initial condition u_0 in (IVP).
        The initial condition is determined by calling u_0(t_0).
    t_0 : float
        The starting time of the simulation.
    T : float
        The end time of the simulation.
    h : float, optional
        The step size for the integration. Default is 1e-6.
    tol : float, optional
        The tolerance for the fixed-point iteration to determine convergence.
        Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the fixed-point solver.
        Default is 100.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - t_history: An array of time points from the simulation.
        - u_history: An array of the approximated solution u(t) at each time point.
    """
    t_curr = t_0
    u_curr = u_0(t_0)

    t_history = [t_curr]
    u_history = [u_curr]

    steps = int((T - t_0) / h)

    for m in tqdm(range(steps), desc="Running Implicit Euler scheme"):
        t_next = t_curr + h

        # Initial guess: explicit Euler 
        u_temp = u_curr + h*F(t_curr, u_curr)
        u_guess = u_temp

        # Fixed-point iteration
        for _ in range(max_iter):
            u_guess = u_curr + h*F(t_next, u_temp)
            if np.linalg.norm(u_guess - u_temp) < tol:
                break
            u_temp = u_guess
        else:
            print(f"Warning: step {m+1} did not converge.")

        t_curr, u_curr = t_next, u_guess  
        t_history.append(t_curr)
        u_history.append(u_curr)

    return np.array(t_history), np.array(u_history)


def Crank_Nicolson_Scheme(F: Callable[[float, np.ndarray], np.ndarray], 
                          u_0: Callable[[float], np.ndarray], 
                          t_0: float, 
                          T: float, 
                          h: float = 1e-6, 
                          tol: float = 1e-6, 
                          max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem (IVP), using the Crank-Nicolson method.

    Parameters
    ----------
    F : callable
        The given source function F in (IVP).
    u_0 : callable
        The given initial condition u_0 in (IVP).
        The initial condition is determined by calling u_0(t_0).
    t_0 : float
        The starting time of the simulation.
    T : float
        The end time of the simulation.
    h : float, optional
        The step size for the integration. Default is 1e-6.
    tol : float, optional
        The tolerance for the fixed-point iteration to determine convergence.
        Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the fixed-point solver.
        Default is 100.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - t_history: An array of time points from the simulation.
        - u_history: An array of the approximated solution u(t) at each time point.
    """
    t_curr = t_0
    u_curr = u_0(t_0)

    t_history = [t_curr]
    u_history = [u_curr]

    steps = int((T - t_0) / h)

    for m in tqdm(range(steps), desc="Running Crank-Nicolson scheme"):
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
            print(f"Warning: step {m+1} did not converge.")

        t_curr, u_curr = t_next, u_guess  
        t_history.append(t_curr)
        u_history.append(u_curr)

    return np.array(t_history), np.array(u_history)


def Specular_Trigonometric_Scheme(F: Callable[[float, np.ndarray], np.ndarray], 
                                  u_0: Callable[[float], np.ndarray], 
                                  t_0: float, 
                                  T: float, 
                                  h: float = 1e-6, 
                                  u_1: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem (IVP), using the specular trigonometric scheme.

    Parameters
    ----------
    F : callable
        The given source function F in (IVP).
    u_0 : callable
        The given initial condition u_0 in (IVP).
        The initial condition is determined by calling u_0(t_0).
    t_0 : float
        The starting time of the simulation.
    T : float
        The end time of the simulation.
    h : float, optional
        The step size for the integration. Default is 1e-6.
    u_1 : np.ndarray, optional
        The state u at time t_0 + h. If not provided (None), it will be calculated using a single explicit Euler step. 
        Default is None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - t_history: An array of time points from the simulation.
        - u_history: An array of the approximated solution u(t) at each time point.
    """
    t_prev = t_0
    u_prev = u_0(t_0)
    
    t_history = [t_0]
    u_history = [u_0(t_0)]  
    
    t_curr = t_prev + h

    if u_1 == None:
        # explicit Euler to get u_1
        u_curr = u_prev + h*F(t_prev, u_prev)    
    else:
        u_curr = u_1
        
    t_history.append(t_curr)
    u_history.append(u_curr)

    steps = int((T - t_0) / h)

    for _ in tqdm(range(steps - 1), desc="Running Specular trigonometric scheme"):
        t_next = t_curr + h
        u_next = u_curr + h*np.tan(2*np.arctan(F(t_curr, u_curr)) - np.arctan((u_curr - u_prev)/h))

        # Update for next step
        t_prev, u_prev = t_curr, u_curr
        t_curr, u_curr = t_next, u_next

        t_history.append(t_curr)
        u_history.append(u_curr)
        
    return np.array(t_history), np.array(u_history)


def Specular_Euler_Scheme_Type_1(F: Callable[[float, np.ndarray], np.ndarray], 
                                 u_0: Callable[[float], np.ndarray], 
                                 t_0: float, 
                                 T: float, 
                                 h: float = 1e-6, 
                                 u_1: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem (IVP), using the Type 1 specular Euler scheme.

    Parameters
    ----------
    F : callable
        The given source function F in (IVP).
    u_0 : callable
        The given initial condition u_0 in (IVP).
        The initial condition is determined by calling u_0(t_0).
    t_0 : float
        The starting time of the simulation.
    T : float
        The end time of the simulation.
    h : float, optional
        The step size for the integration. Default is 1e-6.
    u_1 : np.ndarray, optional
        The state u at time t_0 + h. If not provided (None), it will be calculated using a single explicit Euler step. 
        Default is None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - t_history: An array of time points from the simulation.
        - u_history: An array of the approximated solution u(t) at each time point.
    """
    t_prev = t_0
    u_prev = u_0(t_0)
    
    t_history = [t_prev]
    u_history = [u_prev]  
    
    t_curr = t_prev + h
    if u_1 == None:
        # explicit Euler to get u_1
        u_curr = u_prev + h*F(t_prev, u_prev)    
    else:
        u_curr = u_1
    
    t_history.append(t_curr)
    u_history.append(u_curr)

    steps = int((T - t_0) / h)

    for _ in tqdm(range(steps - 1), desc="Running Specular Euler Scheme Type 1"):
        t_next = t_curr + h
        u_next = u_curr + h*sd.A(float(F(t_curr, u_curr)), float(F(t_prev, u_prev)))   

        # Update for next step
        t_prev, u_prev = t_curr, u_curr
        t_curr, u_curr = t_next, u_next

        t_history.append(t_curr)
        u_history.append(u_curr)
        
    return np.array(t_history), np.array(u_history)


def Specular_Euler_Scheme_Type_2(F: Callable[[float, np.ndarray], np.ndarray], 
                                 u_0: Callable[[float], np.ndarray], 
                                 t_0: float, 
                                 T: float, 
                                 h: float = 1e-6, 
                                 u_1: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem (IVP), using the Type 2 specular Euler scheme.

    Parameters
    ----------
    F : callable
        The given source function F in (IVP).
    u_0 : callable
        The given initial condition u_0 in (IVP).
        The initial condition is determined by calling u_0(t_0).
    t_0 : float
        The starting time of the simulation.
    T : float
        The end time of the simulation.
    h : float, optional
        The step size for the integration. Default is 1e-6.
    u_1 : np.ndarray, optional
        The state u at time t_0 + h. If not provided (None), it will be calculated using a single explicit Euler step. 
        Default is None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - t_history: An array of time points from the simulation.
        - u_history: An array of the approximated solution u(t) at each time point.
    """
    t_prev = t_0
    u_prev = u_0(t_0)
    
    t_history = [t_prev]
    u_history = [u_prev]  
    
    t_curr = t_prev + h
    if u_1 == None:
        # explicit Euler to get u_1
        u_curr = u_prev + h*F(t_prev, u_prev)    
    else:
        u_curr = u_1
    
    t_history.append(t_curr)
    u_history.append(u_curr)

    steps = int((T - t_0) / h)

    for _ in tqdm(range(steps - 1), desc="Running Specular Euler Scheme Type 2"):
        t_next = t_curr + h
        u_next = u_curr + h*sd.A(float(F(t_curr, u_curr)), float((u_curr - u_prev)/h)) 

        # Update for next step
        t_prev, u_prev = t_curr, u_curr
        t_curr, u_curr = t_next, u_next

        t_history.append(t_curr)
        u_history.append(u_curr)
        
    return np.array(t_history), np.array(u_history)


def Specular_Euler_Scheme_Type_3(F: Callable[[float, np.ndarray], np.ndarray], 
                                 u_0: Callable[[float], np.ndarray], 
                                 t_0: float, 
                                 T: float, 
                                 h: float = 1e-6, 
                                 tol: float = 1e-6, 
                                 max_iter: int = 100, 
                                 u_1: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem (IVP), using the Type 3 specular Euler scheme.

    Parameters
    ----------
    F : callable
        The given source function F in (IVP).
    u_0 : callable
        The given initial condition u_0 in (IVP).
        The initial condition is determined by calling u_0(t_0).
    t_0 : float
        The starting time of the simulation.
    T : float
        The end time of the simulation.
    h : float, optional
        The step size for the integration. Default is 1e-6.
    tol : float, optional
        The tolerance for the fixed-point iteration to determine convergence.
        Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the fixed-point solver.
        Default is 100.
    u_1 : np.ndarray, optional
        The state u at time t_0 + h. If not provided, it's bootstrapped using
        a single explicit Euler step. Default is None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - t_history: An array of time points from the simulation.
        - u_history: An array of the approximated solution u(t) at each time point.
    """    
    t_prev = t_0
    u_prev = u_0(t_0)
    
    t_history = [t_prev]
    u_history = [u_prev]  

    t_curr = t_prev + h
    if u_1 == None:
        # explicit Euler to get u_1
        u_curr = u_prev + h*F(t_prev, u_prev)    
    else:
        u_curr = u_1
    
    t_history.append(t_curr)
    u_history.append(u_curr)

    steps = int((T - t_0) / h)

    for m in tqdm(range(steps - 1), desc="Running Specular Euler Scheme Type 3"):
        t_next = t_curr + h

        # Initial guess: explicit Euler
        u_temp = u_curr + h*F(t_curr, u_curr)
        u_guess = u_temp

        beta = F(t_prev, u_prev)  # fixed second argument
        
        # Fixed-point iteration
        for _ in range(max_iter):
            alpha = (u_temp - u_curr) / h
            u_guess = u_curr + h*sd.A(float(alpha), float(beta)) 

            if np.linalg.norm(u_guess - u_temp) < tol:
                break

            u_temp = u_guess
        else:
            print(f"Warning: fixed-point iteration did not converge at step {m+1}")

        # Update for next step
        t_prev, u_prev = t_curr, u_curr
        t_curr, u_curr = t_next, u_guess 

        t_history.append(t_curr)
        u_history.append(u_curr)

    return np.array(t_history), np.array(u_history)

def Specular_Euler_Scheme_Type_4(F: Callable[[float, np.ndarray], np.ndarray], 
                                 u_0: Callable[[float], np.ndarray], 
                                 t_0: float, 
                                 T: float, 
                                 h: float = 1e-6, 
                                 tol: float = 1e-6, 
                                 max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem (IVP), using the Type 4 specular Euler scheme.

    Parameters
    ----------
    F : callable
        The given source function F in (IVP).
    u_0 : callable
        The given initial condition u_0 in (IVP).
        The initial condition is determined by calling u_0(t_0).
    t_0 : float
        The starting time of the simulation.
    T : float
        The end time of the simulation.
    h : float, optional
        The step size for the integration. Default is 1e-6.
    tol : float, optional
        The tolerance for the fixed-point iteration to determine convergence.
        Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the fixed-point solver.
        Default is 100.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - t_history: An array of time points from the simulation.
        - u_history: An array of the approximated solution u(t) at each time point.
    """
    t_curr = t_0
    u_curr = u_0(t_0)

    t_history = [t_curr]
    u_history = [u_curr]

    steps = int((T - t_0) / h)

    for m in tqdm(range(steps), desc="Running Specular Euler Scheme Type 4"):
        t_next = t_curr + h

        # Initial guess: explicit Euler
        u_temp = u_curr + h*F(t_curr, u_curr)
        u_guess = u_temp
        
        beta = F(t_curr, u_curr)  # fixed second argument

        # Fixed-point iteration
        for _ in range(max_iter):
            alpha = (u_temp - u_curr) / h
            u_guess = u_curr + h*sd.A(float(alpha), float(beta)) 

            if np.linalg.norm(u_guess - u_temp) < tol:
                break

            u_temp = u_guess
        else:
            print(f"Warning: fixed-point iteration did not converge at step {m+1}")

        # Update for next step
        t_curr, u_curr = t_next, u_guess  

        t_history.append(t_curr)
        u_history.append(u_curr)

    return np.array(t_history), np.array(u_history)


def Specular_Euler_Scheme_Type_5(F: Callable[[float, np.ndarray], np.ndarray], 
                                 u_0: Callable[[float], np.ndarray], 
                                 t_0: float, 
                                 T: float, 
                                 h: float = 1e-6, 
                                 tol: float = 1e-6, 
                                 max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem (IVP), using the Type 5 specular Euler scheme.

    Parameters
    ----------
    F : callable
        The given source function F in (IVP).
    u_0 : callable
        The given initial condition u_0 in (IVP).
        The initial condition is determined by calling u_0(t_0).
    t_0 : float
        The starting time of the simulation.
    T : float
        The end time of the simulation.
    h : float, optional
        The step size for the integration. Default is 1e-6.
    tol : float, optional
        The tolerance for the fixed-point iteration to determine convergence.
        Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the fixed-point solver.
        Default is 100.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - t_history: An array of time points from the simulation.
        - u_history: An array of the approximated solution u(t) at each time point.
    """
    t_curr = t_0
    u_curr = u_0(t_0)

    t_history = [t_curr]
    u_history = [u_curr]

    steps = int((T - t_0) / h)

    for m in tqdm(range(steps), desc="Running Specular Euler Scheme Type 5"):
        beta = F(t_curr, u_curr)  # fixed second argument
        t_curr = t_curr + h

        # Initial guess: explicit Euler
        u_temp = u_curr + h*beta 
        u_guess = u_temp

        # Fixed-point iteration
        for _ in range(max_iter):
            alpha = F(t_curr, u_temp)
            u_guess = u_curr + h*sd.A(alpha, beta) 

            if np.linalg.norm(u_guess - u_temp) < tol:
                break

            u_temp = u_guess
        else:
            print(f"Warning: fixed-point iteration did not converge at step {m+1}")

        # Update for next step
        u_curr = u_guess    

        t_history.append(t_curr)
        u_history.append(u_curr)

    return np.array(t_history), np.array(u_history)


def Specular_Euler_Scheme_Type_6(F: Callable[[float, np.ndarray], np.ndarray], 
                                 u_0: Callable[[float], np.ndarray], 
                                 t_0: float, 
                                 T: float, 
                                 h: float = 1e-6, 
                                 tol: float = 1e-6, 
                                 max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem (IVP), using the Type 6 specular Euler scheme.

    Parameters
    ----------
    F : callable
        The given source function F in (IVP).
    u_0 : callable
        The given initial condition u_0 in (IVP).
        The initial condition is determined by calling u_0(t_0).
    t_0 : float
        The starting time of the simulation.
    T : float
        The end time of the simulation.
    h : float, optional
        The step size for the integration. Default is 1e-6.
    tol : float, optional
        The tolerance for the fixed-point iteration to determine convergence.
        Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the fixed-point solver.
        Default is 100.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - t_history: An array of time points from the simulation.
        - u_history: An array of the approximated solution u(t) at each time point.
    """
    t_curr = t_0
    u_curr = u_0(t_0)

    t_history = [t_curr]
    u_history = [u_curr]

    steps = int((T - t_0) / h)

    for m in tqdm(range(steps), desc="Running Specular Euler Scheme Type 6"):
        t_next = t_curr + h

        # Initial guess: explicit Euler
        u_temp = u_curr + h*F(t_curr, u_curr)
        u_guess = u_temp

        # Fixed-point iteration
        for _ in range(max_iter):
            alpha = F(t_next, u_temp)
            beta = (u_temp - u_curr) / h

            u_guess = u_curr + h*sd.A(float(alpha), float(beta)) 

            if np.linalg.norm(u_guess - u_temp) < tol:
                break
            
            u_temp = u_guess
        else:
            print(f"Warning: fixed-point iteration did not converge at step {m+1}")

        # Update for next step
        t_curr, u_curr = t_next, u_guess    

        t_history.append(t_curr)
        u_history.append(u_curr)

    return np.array(t_history), np.array(u_history)


def total_error(n: int, 
                exact_solution: np.ndarray, 
                numerical_result: np.ndarray, 
                norm: str = 'max') -> float: 
    """
    Calculates the total error between an exact solution and a numerical approximation with respect to the maximum norm (L-infinity), the discrete L2 norm, or the L1 norm.

    Parameters
    ----------
    n : int
        The number of steps, used to determine the step size h.
    exact_solution : np.ndarray
        An array containing the values of the true solution at the grid points.
    numerical_result : np.ndarray
        An array containing the values of the approximated solution.
    norm : str, optional
        The type of norm to use for the error calculation ('max', 'l2', or 'l1').
        Default is 'max'.

    Returns
    -------
    float
        The computed error value. Returns 0 if the norm is not recognized.
    """
    h = 1/n 

    if norm == 'max':
        return float(np.max(np.abs(exact_solution - numerical_result)))
    elif norm == 'l2':
        return float(np.sqrt(np.sum((exact_solution - numerical_result)**2) * h))
    elif norm == 'l1':
        return float(np.sum(np.abs(exact_solution - numerical_result)))
    else:
        return 0.0


def compute_ratios(error_list: List[Tuple[int, float]]) -> List[Tuple[int, float, Optional[float]]]:
    """
    Computes the convergence ratios from a list of errors.

    Parameters
    ----------
    error_list : list[tuple[int, float]]
        A list of tuples, where each tuple is of the form (n, error).
        'n' typically represents the number of steps or a related parameter, and 'error' is the computed numerical error for that 'n'.
        The list should be sorted by increasing 'n'.

    Returns
    -------
    list[tuple[int, float, float | None]]
        A list of tuples, each of the form (n, error, ratio). 
        The ratio for the first entry is None as there is no previous error to compare against.
    """
    ratio_list = []
    for i in range(len(error_list)):
        n, e = error_list[i]

        if i == 0:
            ratio_list.append((n, e, None))  

        else:
            e_prev = error_list[i - 1][1]
            ratio = np.log2(e_prev / e)
            ratio_list.append((n, e, ratio))

    return ratio_list


def save_table_to_txt(
    df: pd.DataFrame,
    filename: str,
    error_precision: int = 2,
    ratio_precision: int = 2
) -> None:
    """
    Saves a DataFrame of convergence results to a text file in LaTeX table format.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with three columns: 'n', 'error', and 'ratio'.
    filename : str
        The name of the text file to save the results to.
    error_precision : int, optional
        The number of decimal places for the 'error' column. Default is 12.
    ratio_precision : int, optional
        The number of decimal places for the 'ratio' column. Default is 4.

    Returns
    -------
    None
        This function does not return any value; it writes directly to a file.
    """
    with open(filename, "w") as f:
        for n, error, ratio in df.itertuples(index=False, name=None):
            error_str = f"{error:.{error_precision}e}"
            ratio_str = f"{ratio:.{ratio_precision}f}" if pd.notna(ratio) else "--"
            f.write(f"{n:<8}& {error_str} & {ratio_str} \\\\\n")
