import numpy as np
from tqdm import tqdm
import time
from typing import Callable, List, Tuple, Optional, TypeAlias, Sequence
from .result import OptimizationResult
from .step_size import StepSize
from ..calculation import derivative, gradient

SUPPORTED_METHODS = ['specular gradient', 'implicit', 'stochastic', 'hybrid']

ComponentFunc: TypeAlias = Callable[[int | float | np.number | list | np.ndarray], int | float | np.number]

def gradient_method(
    f: Callable[[int | float | np.number | list | np.ndarray], int | float | np.number],
    x_0: int | float | list | np.ndarray,
    step_size: StepSize,
    h: float = 1e-6,
    form: str = 'specular gradient',
    tol: float = 1e-6,
    zero_tol: float = 1e-8,
    max_iter: int = 1000,
    f_j: Sequence[ComponentFunc] | None = None,
    switch_iter: int | None = 2,
    record_history: bool = True,
    print_bar: bool = True
) -> OptimizationResult:
    """
    The specular gradient method for minimizing a nonsmooth convex function.

    Parameters
    ----------
    f : callable
        The objective function to minimize.
    x_0 : int | float | list | np.ndarray
        The starting point for the optimization.
    step_size : StepSize
        The step size `h_k`.
    h : float, optional
        Mesh size used in the finite difference approximation. Must be positive.
        Default: ``1e-6``.
    form : str, optional
        The form of the specular gradient method.
        Supported forms: ``'specular gradient'``, ``'implicit'``, ``'stochastic'``, ``'hybrid'``.
        Default: ``'specular gradient'``.
    tol : float, optional
        Tolerance for iterations.
        Default: ``1e-6``.
    zero_tol : float, optional
        A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.
        Default: ``1e-8``.
    max_iter : int, optional
        Maximum number of iterations.
        Default: ``1000``.
    f_j : sequence of callable | None, optional
        The component function of ``f``.
        Used for the stochastic and hybrid forms to compute a random component of the objective function.
        Default: ``None``.
    switch_iter : int | None, optional
        The iteration to switch from a method to another for the hybrid form.
        Used for the hybrid form only.
        Default: ``2``.
    record_history : bool, optional
        Whether to record the history of variables and function values.
        Default: ``True``.
    print_bar : bool, optional
        Whether to print the progress bar.
        Default: ``True``.

    Returns
    -------
    OptimizationResult
        The result of the optimization containing the solution, function value, number of iterations, runtime, and history.
    
    Raises
    ------
    ValueError
        If ``h`` is not positive.
    TypeError
        If an unknown ``form`` is provided.
    """

    if h is None or h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")
    
    x = np.array(x_0, dtype=float).copy()
    n = x.size
    
    all_history = {}
    x_history = []
    f_history = []

    start_time = time.time()

    # the n-dimensional case
    if n > 1:
        if form == 'specular gradient':
            res_x, res_f, res_k = _vector(f, f_history, x, x_history, step_size, h, tol, zero_tol, max_iter, record_history, print_bar)

        elif form == 'stochastic':
            if f_j is None:
                raise ValueError("Component functions 'f_j' must be provided for the stochastic form.")

            form = 'stochastic specular gradient'
            res_x, res_f, res_k = _vector_stochastic(f_j, f, f_history, x, x_history, step_size, h, tol, zero_tol, max_iter, record_history, print_bar)

        elif form == 'hybrid':
            if f_j is None:
                raise ValueError("Component functions 'f_j' must be provided for the stochastic form.")
            
            form = 'hybrid specular gradient'
            switch_iter = switch_iter if switch_iter is not None else max_iter

            res_x, res_f, res_k = _vector(f, f_history, x, x_history, step_size, h, tol, zero_tol, switch_iter, record_history, print_bar)
            res_x, res_f, res_k = _vector_stochastic(f_j, f, f_history, res_x, x_history, step_size, h, tol, zero_tol, max_iter - switch_iter, record_history, print_bar)

        else:
            raise TypeError(f"Unknown form '{form}'. Supported forms: {SUPPORTED_METHODS}")

    # the one-dimensional case
    elif n == 1:
        x = x.item()

        if form == 'specular gradient':
            res_x, res_f, res_k = _scalar(f, f_history, x, x_history, step_size, h, tol, zero_tol, max_iter, record_history, print_bar)
            
        elif form == 'implicit':
            form = 'implicit specular gradient'
            res_x, res_f, res_k = _scalar_implicit(f, f_history, x, x_history, step_size, h, tol, max_iter, record_history, print_bar)
            
        else:
            raise TypeError(f"Unknown form '{form}'. Supported forms: {SUPPORTED_METHODS}")
    
    else:
        raise TypeError(f"Unknown form '{form}'. Supported forms: {SUPPORTED_METHODS}")
    
    runtime = time.time() - start_time

    if record_history:
        all_history["variables"] = x_history
        all_history["values"] = f_history

    return OptimizationResult(
        method=form,
        solution=res_x,
        func_val=res_f,
        iteration=res_k,
        runtime=runtime,
        all_history=all_history
    ) 

def _scalar(
    f: Callable[[int | float | np.number], int | float | np.number],
    f_history: list,
    x: int | float,
    x_history: list,
    step_size: StepSize,
    h: float = 1e-6,
    tol: float = 1e-6,
    zero_tol: float = 1e-8,
    max_iter: int = 1000,
    record_history: bool = True,
    print_bar: bool = True
) -> tuple:
    """
    Scalar implementation of :func:`gradient_method`.
    The specular gradient method in the one-dimensional case.
    """
    k = 1

    for _ in tqdm(range(1, max_iter + 1), desc="Running the specular gradient method", disable=not print_bar):
        if record_history is True:
            x_history.append(x)
            f_history.append(f(x))

        specular_derivative = derivative(f=f, x=x, h=h, zero_tol=zero_tol)

        if abs(specular_derivative) < tol:
            break
        
        x -= step_size(k)*(specular_derivative / abs(specular_derivative))
        k += 1
    
    return x, f(x), k

def _scalar_implicit(
    f: Callable[[int | float | np.number], int | float | np.number],
    f_history: list,
    x: int | float,
    x_history: list,
    step_size: StepSize,
    h: float = 1e-6,
    tol: float = 1e-6,
    max_iter: int = 1000,
    record_history: bool = True,
    print_bar: bool = True
) -> tuple:
    """
    Scalar implementation of :func:`gradient_method`.
    The implicit specular gradient method in the one-dimensional case.
    """
    k = 1

    for _ in tqdm(range(1, max_iter + 1), desc="Running the implicit specular gradient method", disable=not print_bar):
        if record_history is True:
            x_history.append(x)
            f_history.append(f(x))

        sum_of_one_sided_derivatives = (f(x + h) - f(x - h)) / h

        if abs(sum_of_one_sided_derivatives) < tol:
            break
        
        x -= step_size(k)*(sum_of_one_sided_derivatives / abs(sum_of_one_sided_derivatives))
        k += 1
    
    return x, f(x), k

def _vector(
    f: Callable[[list | np.ndarray], int | float | np.number],
    f_history: list,
    x: list | np.ndarray,
    x_history: list,
    step_size: StepSize,
    h: float = 1e-6,
    tol: float = 1e-6,
    zero_tol: float = 1e-8,
    max_iter: int = 1000, 
    record_history: bool = True,
    print_bar: bool = True
) -> tuple:
    """
    Vector implementation of :func:`gradient_method`.
    The specular gradient method in the n-dimensional case.
    """
    k = 1

    for _ in tqdm(range(1, max_iter + 1), desc="Running the specular gradient method", disable=not print_bar):
        if record_history is True:
            x_history.append(x)
            f_history.append(f(x))

        specular_gradient = gradient(f=f, x=x, h=h, zero_tol=zero_tol)
        norm = np.linalg.norm(specular_gradient)

        if norm < tol:
            break

        x -= step_size(k)*(specular_gradient / norm)
        k += 1
    
    return x, f(x), k

def _vector_stochastic(
    f_j: Sequence[ComponentFunc],
    f: Callable[[list | np.ndarray], int | float | np.number],
    f_history: list,
    x: list | np.ndarray,
    x_history: list,
    step_size: StepSize,
    h: float = 1e-6,
    tol: float = 1e-6,
    zero_tol: float = 1e-8,
    max_iter: int = 1000, 
    record_history: bool = True,
    print_bar: bool = True
) -> tuple:
    """
    Vector implementation of :func:`gradient_method`.
    The specular gradient method in the n-dimensional case.
    """
    k = 1

    for _ in tqdm(range(1, max_iter + 1), desc="Running the specular gradient method", disable=not print_bar):
        # A random index j is selected at each iteration
        j = np.random.randint(len(f_j))
        component_func = f_j[j]

        if record_history is True:
            x_history.append(x)
            f_history.append(f(x)) 

        component_specular_gradient = gradient(f=component_func, x=x, h=h, zero_tol=zero_tol)
        norm = np.linalg.norm(component_specular_gradient)

        if norm < tol:
            break

        x -= step_size(k)*(component_specular_gradient / norm)
        k += 1

    return x, f(x), k