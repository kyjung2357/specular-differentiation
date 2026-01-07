import numpy as np
from tqdm import tqdm
import time
from typing import Callable, List, Tuple, Optional
from .result import OptimizationResult
from .step_size import StepSize
from ..calculation import derivative, gradient

SUPPORTED_METHODS = ['specular gradient', 'implicit', 'stochastic', 'hybrid']

def gradient_method(
    f: Callable[[int | float | list | np.ndarray], int | float | np.floating],
    x_0: int | float | list | np.ndarray, 
    step_size: StepSize,  
    h: float = 1e-6, 
    form: str = 'specular gradient',
    tol: float = 1e-6, 
    zero_tol: float = 1e-8,
    max_iter: int = 1000, 
    f_j: Optional[Callable[[int | float | list | np.ndarray], int | float | np.floating]] = None,
    first_iter: Optional[int] = 2,   
    record_history: bool = True,
    print_bar: bool = True
) -> OptimizationResult:

    if h is None or h <= 0:
        raise ValueError(f"Numerical differentiation 'h' needs to be positive. Got {h}")
    
    x = np.array(x_0, dtype=float).copy()
    n = x.size
    
    all_history = {}
    x_history = []
    f_history = []

    start_time = time.time()

    # the n-dimensional case
    if n > 1:
        if form == 'specular gradient':
            res_x, res_f, res_k = _vector(f, x, step_size, h, tol, zero_tol, max_iter, record_history, x_history, f_history, print_bar)

        elif form == 'stochastic':
            form = 'stochastic specular gradient'
            pass # TODO

        elif form == 'hybrid':
            form = 'hybrid specular gradient'
            pass # TODO

        else:
            raise TypeError(f"Unknown form '{form}'. Supported forms: {SUPPORTED_METHODS}")

    # the one-dimensional case
    elif n == 1:
        x = x.item()

        if form == 'specular gradient':
            res_x, res_f, res_k = _scalar(f, x, step_size, h, tol, zero_tol, max_iter, record_history, x_history, f_history, print_bar)
            
        elif form == 'implicit':
            form = 'implicit specular gradient'
            res_x, res_f, res_k = _scalar_implicit(f, x, step_size, h, tol, max_iter, record_history, x_history, f_history, print_bar)
            
        else:
            raise TypeError(f"Unknown form '{form}'. Supported forms: {SUPPORTED_METHODS}")
    
    else:
        raise TypeError(f"Unknown form '{form}'. Supported forms: {SUPPORTED_METHODS}")
    
    runtime = time.time() - start_time  # type: ignore

    if record_history:
        all_history["variables"] = x_history
        all_history["values"] = f_history

    return OptimizationResult(
        method=form,
        solution=res_x, # type: ignore
        func_val=res_f, # type: ignore
        iteration=res_k, # type: ignore
        runtime=runtime,
        all_history=all_history
    ) 

def _scalar(
    f,
    x,
    step_size,
    h,
    tol,
    zero_tol,
    max_iter,
    record_history,
    x_history,
    f_history,
    print_bar
) -> tuple:
    """
    Scalar implementation of :func:`gradient_method`.
    The specular gradient method in the one-dimensional case.    
    """
    k = 1

    for _ in tqdm(range(1, max_iter + 1), desc="Running the specular gradient method", disable=not print_bar):
        if record_history is True:
            x_history.append(x) # type: ignore
            f_history.append(f(x))

        specular_derivative = derivative(f=f, x=x, h=h, zero_tol=zero_tol)

        if abs(specular_derivative) < tol:
            break
        
        x -= step_size(k)*(specular_derivative / abs(specular_derivative))
        k += 1
    
    return x, f(x), k

def _vector(
    f,
    x,
    step_size,
    h,
    tol,
    zero_tol,
    max_iter,
    record_history,
    x_history,
    f_history,
    print_bar
) -> tuple:
    """
    Vector implementation of :func:`gradient_method`.
    The specular gradient method in the n-dimensional case.
    """
    k = 1

    for _ in tqdm(range(1, max_iter + 1), desc="Running the specular gradient method", disable=not print_bar):
        if record_history is True:
            x_history.append(x) # type: ignore
            f_history.append(f(x))

        specular_derivative = gradient(f=f, x=x, h=h, zero_tol=zero_tol)

        if abs(specular_derivative) < tol:
            break
        
        x -= step_size(k)*(specular_derivative / abs(specular_derivative))
        k += 1
    
    return x, f(x), k

def _scalar_implicit(
    f,
    x,
    step_size,
    h,
    tol,
    max_iter,
    record_history,
    x_history,
    f_history,
    print_bar
) -> tuple:
    """
    Scalar implementation of :func:`gradient_method`.
    The implicit specular gradient method in the one-dimensional case.
    """
    k = 1

    for _ in tqdm(range(1, max_iter + 1), desc="Running the implicit specular gradient method", disable=not print_bar):
        if record_history is True:
            x_history.append(x) # type: ignore
            f_history.append(f(x))

        sum_of_one_sided_derivatives = (f(x + h) - f(x - h)) / h

        if abs(sum_of_one_sided_derivatives) < tol:
            break
        
        x -= step_size(k)*(sum_of_one_sided_derivatives / abs(sum_of_one_sided_derivatives))
        k += 1
    
    return x, f(x), k

