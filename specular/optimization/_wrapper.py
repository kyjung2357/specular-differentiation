from .step_size.step_schedule import StepSchedule
from .step_size.line_search import LineSearch
from .result import OptimizationResult
from ..calculation import derivative, gradient
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

_SUPPORTED_STEP_SCHEDULE = StepSchedule._SUPPORTED_OPTIONS.keys()
_SUPPORTED_LINE_SEARCH = LineSearch._SUPPORTED_OPTIONS.keys()
_SUPPORTED_STEP_SIZE = _SUPPORTED_STEP_SCHEDULE | _SUPPORTED_LINE_SEARCH


def _classical_gradient(f: Callable, x, h: float) -> np.ndarray | float:
    if np.isscalar(x) or np.asarray(x).ndim == 0:
        return float((f(x + h) - f(x - h)) / (2.0 * h))
    x_arr = np.asarray(x, dtype=float)
    grad = np.empty_like(x_arr)
    for i in range(x_arr.size):
        e = np.zeros_like(x_arr)
        e[i] = h
        grad[i] = (f(x_arr + e) - f(x_arr - e)) / (2.0 * h)
    return grad

def _specular_gradient(f: Callable, x, h: float, zero_tol: float) -> np.ndarray | float:
    if np.isscalar(x) or np.asarray(x).ndim == 0:
        x_val = float(x) if not isinstance(x, np.ndarray) else float(x.item())
        return derivative(f, x_val, h=h, zero_tol=zero_tol)
    return gradient(f, x, h=h, zero_tol=zero_tol)

class WrapperOptions(TypedDict, total=False):
    h: float
    zero_tol: float
    
    # Hybrid specular gradient method
    f_j: ComponentFuncs
    m: int
    switch_iter: int
    
    # Step size schedules
    a: float
    b: float
    r: float
    user_defined_rule: Callable

    # Line search methods
    t_0: float
    c_1: float
    c_2: float
    c_3: float
    rho: float
    max_alpha: float
    line_search_max_iter: int
    skip_on_fail: bool

def _wrapper(
        objective_function: Callable[..., Scalar],
        initial_point: Scalar | Vector,
        method: str,
        step_size: str,
        max_iter: int = 1000,
        tol: float = 1e-6,
        print_bar: bool = True,
        options: WrapperOptions | None = None
) -> OptimizationResult:
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"Unknown method '{method}'. Supported methods: {_SUPPORTED_METHODS}")
    
    options = options or {}

    h = float(options.get('h', 1e-6))
    zero_tol = float(options.get('zero_tol', 1e-8))

    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")
    if zero_tol < 0:
        raise ValueError(f"'zero_tol' cannot be negative. Got {zero_tol}")
    
    if "specular" in method:
        grad_fn = lambda x_: _specular_gradient(objective_function, x_, h, zero_tol)
    else:
        grad_fn = lambda x_: _classical_gradient(objective_function, x_, h)

    from .search_direction import get_direction_finder
    direction_finder = get_direction_finder(method, options)
    
    if step_size in StepSchedule._SUPPORTED_OPTIONS:
        t_ = StepSchedule(
            name=step_size,
            a=options.get('a'),
            b=options.get('b'),
            r=options.get('r'),
            user_defined_rule=options.get('user_defined_rule')
        )
    elif step_size in LineSearch._SUPPORTED_OPTIONS:
        t_ = LineSearch(
            name=step_size,
            f=objective_function,
            h=h,
            zero_tol=zero_tol,
            t_0=float(options.get('t_0', 1.0)),
            c_1=float(options.get('c_1', 1e-4)),
            c_2=float(options.get('c_2', 0.9)),
            c_3=float(options.get('c_3', 0.9)),
            rho=float(options.get('rho', 0.5)),
            max_iter=int(options.get('line_search_max_iter', 20)),
            max_alpha=float(options.get('max_alpha', 1e8)),
            skip_on_fail=bool(options.get('skip_on_fail', False)),
        )
    else:
        raise ValueError(f"Unknown step size '{step_size}'. Supported: {list(_SUPPORTED_STEP_SIZE)}")

    is_stoch_or_hybrid = "stochastic" in method or "hybrid" in method
    if is_stoch_or_hybrid:
        f_j = options.get('f_j')
        m = int(options.get('m', 1))
        switch_iter = int(options.get('switch_iter', 10))
        
        if f_j is None:
            raise ValueError(f"f_j must be provided for {method}")
        if isinstance(f_j, (list, tuple)):
            num_components = len(f_j)
            if num_components == 0:
                raise ValueError("f_j must contain at least one component function.")
            def get_component(j):
                return f_j[j]
        elif callable(f_j):
            num_components = m
            if num_components <= 0:
                raise ValueError(f"m must be positive when f_j is callable. Got {m}")
            def get_component(j):
                return lambda x_: f_j(x_, j)
        else:
            raise TypeError(f"f_j must be a sequence of component functions or a callable. Got {type(f_j)}")

    x = initial_point
    f = objective_function

    stop_reason = "max_iter reached"

    x_history = [x]
    f_history = [f(x)]

    start_time = time.time()

    desc_text = f"Running {method} method"

    for k in tqdm(
        range(1, max_iter + 1), 
        desc=desc_text, 
        disable=not print_bar, 
        leave=False
    ):
        if is_stoch_or_hybrid and ("stochastic" in method or k >= switch_iter):
            j = np.random.randint(num_components)
            f_current = get_component(j)
            if "specular" in method:
                grad = _specular_gradient(f_current, x, h, zero_tol)
            else:
                grad = _classical_gradient(f_current, x, h)
        else:
            f_current = f
            grad = grad_fn(x)

        d_k = direction_finder(k, x, grad)

        if np.linalg.norm(d_k) < tol:
            stop_reason = "gradient norm below tolerance"
            break

        t_k = t_(k, x=x, d_k=d_k, grad=grad, f=f_current)
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