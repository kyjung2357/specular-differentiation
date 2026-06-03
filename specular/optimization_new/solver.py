from .result import OptimizationResult
from .._typing import Scalar, Vector, ScalarToScalarFunc, VectorToScalarFunc, ComponentFuncs

from typing import TypedDict, Callable, Any
import time
import numpy as np
from tqdm.auto import tqdm

_SUPPORTED_METHODS = {
    "gradient_descent","specular_gradient", "stochastic_specular_gradient", "hybrid_specular_gradient",
    "Adam",
    "BFGS", "specular_BFGS", "specular_modified_BFGS"
}
_SUPPORTED_LINE_SEARCH = {
    "exact",
    "Armijo", "specular_Armijo",
    "Wolfe", "specular_Wolfe",
    "strong_Wolfe", "specular_strong_Wolfe"
}
_SUPPORTED_STEP_SCHEDULE = {
    "constant",
    "not_summable", "square_summable_not_summable",
    "geometric_series",
    "user_defined"
}
_SUPPORTED_STEP_SIZE = _SUPPORTED_LINE_SEARCH | _SUPPORTED_STEP_SCHEDULE

class SolverOptions(TypedDict, total=False):
    h: float
    zero_tol: float
    
    # Hybrid specular gradient method
    f_j: ComponentFuncs
    m: int
    switch_iter: int
    
    # Line search methods
    t_0: float
    c_1: float
    c_2: float
    c_3: float
    rho: float
    line_search_max_iter: int
    skip_on_fail: bool

def minimize(
        objective_function: Callable[..., Scalar],
        initial_point: Scalar | Vector,
        step_size: str,
        method: str,
        max_iter: int = 1000,
        tol: float = 1e-6,
        print_bar: bool = True,
        options: SolverOptions | None = None
) -> OptimizationResult:
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"Unknown method '{method}'. Supported methods: {_SUPPORTED_METHODS}")
    
    if step_size not in _SUPPORTED_STEP_SIZE:
        raise ValueError(f"Unknown step size '{step_size}. Supported step sizes: {_SUPPORTED_STEP_SIZE}")

    if options:
        if 'h' in options:
            h = options['h']
            if h is None or h <= 0:
                raise ValueError(f"Mesh size 'h' must be positive. Got {h}")
            
        if 'zero_tol' in options:
            zero_tol = options['zero_tol']
            if zero_tol is not None and zero_tol < 0:
                raise ValueError(f"'zero_tol' cannot be negative. Got {zero_tol}")
                
    x = initial_point
    f = objective_function

    stop_reason = "max_iter reached"

    x_history = [x]
    f_history = [f(x)]

    start_time = time.time()

    desc_text = f"Running {method} method"

    for k in tqdm(
        range(1, max_iter), 
        desc=desc_text, 
        disable=not print_bar, 
        leave=False
    ):
        t_k = step_size #TODO
        d_k = direction #TODO

        if np.linalg.norm(d_k) < tol:
            stop_reason = "gradient norm below tolerance"
            break

        x = x + t_k * d_k

        x_history.append(x)
        f_history.append(f(x))

    runtime = time.time() - start_time

    return OptimizationResult(
        method=method,
        solution=x_history[-1],
        func_val=f_history[-1],
        iteration=max_iter,
        runtime=runtime,
        history={'variables': x_history, 'values': f_history},
        stop_reason=stop_reason
    )