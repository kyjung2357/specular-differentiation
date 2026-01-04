import numpy as np
from tqdm import tqdm
import time
from typing import Callable, List, Tuple, Optional
from .result import OptimizationResult
from .step_size import StepSize
from ..calculation import derivative

SUPPORTED_METHODS = ['specular gradient method', '', 'implicit', 'stochastic', 'hybrid']

def gradient_method(
    method: str,
    objective_func: Callable[[int | float | list | np.ndarray], int | float | np.floating],
    initial_point: int | float | list | np.ndarray, 
    step_size: StepSize,  
    max_iteration: int = 1000, 
    tol: float = 1e-6, 
    h: Optional[float] = 1e-6, 
    first_iteration: Optional[int] = 2,   
    name: Optional[str] = "no name",
    record_history: bool = True,
    record_time: bool = True
) -> OptimizationResult:

    x = np.array(initial_point, dtype=float).copy()
    f = objective_func
    n = x.ndim
    k = 0
    
    history = {}
    x_history = []
    f_history = []

    if record_time is True:
        start_time = time.time()

    # the n-dimensional case
    if n > 0:
        if method == '' or method == 'specular gradient method':
            pass # TODO

        else:
            raise ValueError(f"Unknown method '{method}'. Supported methods: {SUPPORTED_METHODS}")

    # the one-dimensional case
    else:
        x = x.item()

        if method == '' or method == 'specular gradient method':
            if h is None or h <= 0:
                raise ValueError("Numerical differentiation requires a positive step size 'h'.")
            
            print(f"[Specular gradient method]    {name}")

            for _ in tqdm(range(1, max_iteration + 1)):
                if record_history is True:
                    x_history.append(x)
                    f_history.append(f(x))

                specular_derivative = derivative(f=f, x=x, h=h) # type: ignore

                if abs(specular_derivative) < tol:
                    break

                k += 1
                x -= step_size(k)*(specular_derivative / abs(specular_derivative))

        elif method == 'implicit':
            if h is None or h <= 0:
                raise ValueError("Numerical differentiation requires a positive step size 'h'.")
            
            print(f"[Implicit specular gradient method]    {name}")

            for _ in tqdm(range(1, max_iteration + 1)):
                if record_history is True:
                    x_history.append(x)
                    f_history.append(f(x))

                sum_of_one_sided_derivatives = (f(x + h) - f(x - h)) / h

                if abs(sum_of_one_sided_derivatives) < tol:
                    break
                
                k += 1
                x -= step_size(k)*(sum_of_one_sided_derivatives / abs(sum_of_one_sided_derivatives))
        else:
            raise ValueError(f"Unknown method '{method}'. Supported methods: {SUPPORTED_METHODS}")

    print("\n")
    
    if record_history is True:
        history["variables"] = x_history
        history["values"] = f_history

    if record_time is True:
        history["time"] = time.time() - start_time  # type: ignore

    history["method"] = method
    history["name"] = name

    return OptimizationResult(solution=x, objective_func_val=f(x), iteration=k, history=history) # type: ignore


# def _specular_gradient_method_scalar(
#     method: str,
#     objective_func: Callable[[int | float | list | np.ndarray], int | float | np.floating],
#     initial_point: int | float | list | np.ndarray, 
#     step_size: StepSize,  
#     max_iteration: int = 1000, 
#     tol: float = 1e-6, 
#     h: Optional[float] = 1e-6, 
#     first_iteration: Optional[int] = 2,   
#     name: Optional[str] = "no name",
#     record_history: bool = True,
#     record_time: bool = True):

# def _specular_gradient_method_vector(
#     method: str,
#     objective_func: Callable[[int | float | list | np.ndarray], int | float | np.floating],
#     initial_point: int | float | list | np.ndarray, 
#     step_size: StepSize,  
#     max_iteration: int = 1000, 
#     tol: float = 1e-6, 
#     h: Optional[float] = 1e-6, 
#     first_iteration: Optional[int] = 2,   
#     name: Optional[str] = "no name",
#     record_history: bool = True,
#     record_time: bool = True):