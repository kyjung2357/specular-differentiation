import numpy as np
from tqdm import tqdm
import time
import inspect
from collections.abc import Sequence as SequenceABC
from typing import Callable, cast

from .result import OptimizationResult
from .line_search import LineSearch, LineSearchError
from .step_schedule import StepSchedule
from ..calculation import derivative, gradient
from .._typing import Scalar, Vector, ScalarToScalarFunc, VectorToScalarFunc, ComponentFuncs

SUPPORTED_METHODS = ['specular gradient', 'implicit', 'stochastic', 'hybrid']

def gradient_method(
    f: ScalarToScalarFunc | VectorToScalarFunc,
    x_0: Scalar | Vector,
    step_size: StepSchedule | LineSearch,
    h: float = 1e-6,
    form: str = 'specular gradient',
    tol: float = 1e-6,
    zero_tol: float = 1e-8,
    max_iter: int = 1000,
    f_j: ComponentFuncs | None = None,
    m: int = 1,
    switch_iter: int | None = 2,
    record_history: bool = True,
    fill_iteration: bool = False,
    print_bar: bool = True
) -> OptimizationResult:
    """
    The specular gradient method for minimizing a nonsmooth convex function.

    Parameters:
        f (callable):
            The objective function to minimize.
        x_0 (int | float | list | np.ndarray):
            The starting point for the optimization.
        step_size (StepSchedule):
            The step size `h_k`.
        h (float, optional):
            Mesh size used in the finite difference approximation. Must be positive.
        form (str, optional):
            The form of the specular gradient method.
            Supported forms: ``'specular gradient'``, ``'implicit'``, ``'stochastic'``, ``'hybrid'``.
        tol (float, optional):
            Tolerance for iterations.
        zero_tol (float, optional):
            A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.
        max_iter (int, optional):
            Maximum number of iterations.
        f_j (sequence of callable | callable | None, optional):
            The component function of ``f``.
            Used for the stochastic and hybrid forms to compute a random component of the objective function.

            * If a sequence of callables is provided, each callable should accept a single argument (the variable `x`).

            * If a single callable is provided, it should accept two arguments: the variable `x` and an index `j`, and return the `j`-th component function value at `x`.
        switch_iter (int | None, optional):
            The iteration to switch from a method to another for the hybrid form.
            Used for the hybrid form only.
        record_history (bool, optional):
            Whether to record the history of variables and function values.
        print_bar (bool, optional):
            Whether to print the progress bar.

    Returns:
        The result of the optimization containing the solution, function value, number of iterations, runtime, and history.
    
    Raises:
        ValueError:
            If ``h`` is not positive.
        TypeError:
            If an unknown ``form`` is provided.
    """

    if h is None or h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")
    
    x = np.array(x_0, dtype=float).copy()
    n = x.size
    stop_reason = "max_iter reached"

    all_history = {}
    x_history = []
    f_history = []

    start_time = time.time()

    # the n-dimensional case
    if n > 1:
        f = cast(VectorToScalarFunc, f)

        if form == 'specular gradient':
            res_x, res_f, res_k, stop_reason = _vector(
                f, f_history, x, x_history, step_size, h, tol, zero_tol, max_iter, record_history, print_bar
            )

        elif form == 'stochastic':
            if f_j is None:
                raise ValueError("Component functions 'f_j' must be provided for the stochastic form.")

            form = 'stochastic specular gradient'
            res_x, res_f, res_k, stop_reason = _vector_stochastic(
                f, f_history, x, x_history, step_size, h, tol, zero_tol, f_j, m, max_iter, record_history, print_bar
            ) # type: ignore

        elif form == 'hybrid':
            if f_j is None:
                raise ValueError("Component functions 'f_j' must be provided for the stochastic form.")
            
            # Phase 1: deterministic
            form = 'hybrid specular gradient'
            switch_iter = switch_iter if switch_iter is not None else max_iter
            remaining_iter = max_iter - switch_iter

            # Phase 2: stochastic
            res_x, res_f, res_k, stop_reason = _vector(
                f, f_history, x, x_history, step_size, h, tol, zero_tol, switch_iter, record_history, print_bar
            )
            res_x, res_f, res_k, stop_reason = _vector_stochastic(
                f=f,
                f_history=f_history,
                x=res_x,
                x_history=x_history,
                step_size=step_size,
                h=h,
                tol=tol,
                zero_tol=zero_tol,
                f_j=f_j,
                m=m,
                max_iter=remaining_iter,
                record_history=record_history,
                print_bar=print_bar,
                k_start=res_k + 1
            )

        else:
            raise TypeError(f"Unknown form '{form}'. Supported forms: {SUPPORTED_METHODS}")

    # the one-dimensional case
    elif n == 1:
        f = cast(ScalarToScalarFunc, f)
        x = x.item()

        if form == 'specular gradient':
            res_x, res_f, res_k, stop_reason = _scalar(
                f, f_history, x, x_history, step_size, h, tol, zero_tol, max_iter, record_history, print_bar
            )
            
        elif form == 'implicit':
            form = 'implicit specular gradient'
            res_x, res_f, res_k, stop_reason = _scalar_implicit(
                f, f_history, x, x_history, step_size, h, tol, max_iter, record_history, print_bar
            )
            
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
        all_history=all_history,
        fill_iteration=fill_iteration,
        max_iter=max_iter,
        stop_reason=stop_reason
    ) 

def _scalar(
    f: ScalarToScalarFunc,
    f_history: list,
    x: Scalar,
    x_history: list,
    step_size: StepSchedule | LineSearch,
    h: float,
    tol: float,
    zero_tol: float,
    max_iter: int,
    record_history: bool,
    print_bar: bool
) -> tuple:
    """
    Scalar implementation of ``specular.gradient_method``.
    The specular gradient method in the one-dimensional case.
    """
    x = float(x)
    k = 0
    stop_reason = "max_iter reached"

    for k in tqdm(range(1, max_iter + 1), desc="Running the specular gradient method", disable=not print_bar, leave=False):
        if record_history is True:
            x_history.append(x)
            f_history.append(f(x))

        specular_derivative = float(np.asarray(derivative(f=f, x=x, h=h, zero_tol=zero_tol), dtype=float).reshape(-1)[0])
        norm = abs(specular_derivative)

        if not np.isfinite(norm):
            raise FloatingPointError("Specular derivative norm is not finite.")

        if norm < tol:
            stop_reason = "gradient norm below tolerance"
            break

        d_k = float(specular_derivative / norm)

        if isinstance(step_size, LineSearch):
            t_k = float(step_size(
                f=f,
                x=x,
                direction=-d_k,
                gradient_current=specular_derivative
            ))
        else:
            t_k = float(step_size(k))

        x -= t_k * d_k
    
    return x, f(x), k, stop_reason

def _scalar_implicit(
    f: ScalarToScalarFunc,
    f_history: list,
    x: Scalar,
    x_history: list,
    step_size: StepSchedule | LineSearch,
    h: float,
    tol: float,
    max_iter: int,
    record_history: bool,
    print_bar: bool
) -> tuple:
    """
    Scalar implementation of ``specular.gradient_method``.
    The implicit specular gradient method in the one-dimensional case.
    """
    k = 0
    x = float(x)
    stop_reason = "max_iter reached"

    for k in tqdm(range(1, max_iter + 1), desc="Running the implicit specular gradient method", disable=not print_bar, leave=False):
        if record_history is True:
            x_history.append(x)
            f_history.append(f(x))

        # This is the sum of the right and left one-sided slopes, not a central difference.
        sum_of_one_sided_derivatives = float((f(x + h) - f(x - h)) / h)

        if abs(sum_of_one_sided_derivatives) < tol:
            stop_reason = "gradient norm below tolerance"
            break

        d_k = float(sum_of_one_sided_derivatives / abs(sum_of_one_sided_derivatives))

        if isinstance(step_size, LineSearch):
            t_k = float(step_size(
                f=f,
                x=x,
                direction=-d_k,
                gradient_current=sum_of_one_sided_derivatives
            ))
        else:
            t_k = float(step_size(k))

        x -= t_k * d_k

    return x, f(x), k, stop_reason

def _vector(
    f: VectorToScalarFunc,
    f_history: list,
    x: Vector,
    x_history: list,
    step_size: StepSchedule | LineSearch,
    h: float,
    tol: float,
    zero_tol: float,
    max_iter: int, 
    record_history: bool,
    print_bar: bool
) -> tuple:
    """
    Vector implementation of ``specular.gradient_method``.
    The specular gradient method in the n-dimensional case.
    """
    k = 0
    stop_reason = "max_iter reached"

    for k in tqdm(range(1, max_iter + 1), desc="Running the specular gradient method", disable=not print_bar, leave=False):
        if record_history is True:
            x_history.append(x.copy())
            f_history.append(f(x))

        computation = gradient(f=f, x=x, h=h, zero_tol=zero_tol, quasi_Fermat=True, monotonicity=False)
        specular_gradient = computation[0]
        norm = np.linalg.norm(specular_gradient)

        if not np.isfinite(norm):
            raise FloatingPointError("Specular gradient norm is not finite.")

        if norm < tol:
            stop_reason = "gradient norm below tolerance"
            break
        
        d_k = specular_gradient / norm

        if isinstance(step_size, LineSearch):
            t_k = step_size(
                f=f,
                x=x,
                direction=-d_k,
                gradient_current=specular_gradient,
                gradient_f=lambda z: np.asarray(gradient(f=f, x=z, h=h, zero_tol=zero_tol, quasi_Fermat=True, monotonicity=False,)[0], dtype=float,),
            )
        else:
            t_k = step_size(k)

        x -= t_k*d_k
    
    return x, f(x), k, stop_reason

def _vector_stochastic(
    f: VectorToScalarFunc,
    f_history: list,
    x: Vector,
    x_history: list,
    step_size: StepSchedule | LineSearch,
    h: float,
    tol: float,
    zero_tol: float,
    f_j: ComponentFuncs,
    m: int = 1,
    max_iter: int = 1000, 
    record_history: bool = True,
    print_bar: bool = True,
    k_start: int = 1,
) -> tuple:
    """
    Vector implementation of ``specular.gradient_method``.
    The stochastic specular gradient method in the $n$-dimensional case.
    """
    k = k_start - 1
    stop_reason = "max_iter reached" 

    if isinstance(f_j, SequenceABC) and not isinstance(f_j, (str, bytes)):
        if len(f_j) == 0:
            raise ValueError("f_j must contain at least one component function.")

        for component in f_j:
            if not callable(component):
                raise TypeError("Each element of f_j must be callable.")

        num_components = len(f_j)
    else:
        if not callable(f_j):
            raise TypeError(
                f"f_j must be a sequence of component functions or a callable. Got {type(f_j)} instead."
            )

        sig = inspect.signature(f_j)
        params = list(sig.parameters.values())
        has_varargs = any(
            p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            for p in params
        )

        if len(params) < 2 and not has_varargs:
            raise ValueError(
                f"The function f_j must accept at least 2 arguments (x and index). "
                f"Current signature is: {sig}"
            )
        
        if m <= 0:
            raise ValueError(f"m must be positive when f_j is callable. Got {m}")

        num_components = m

    for k in tqdm(range(k_start, k_start + max_iter), desc="Running the stochastic specular gradient method", disable=not print_bar, leave=False):
        if record_history is True:
            x_history.append(x.copy())
            f_history.append(f(x)) 

        # A random index j is selected at each iteration.
        j = np.random.randint(num_components)

        component_func: VectorToScalarFunc

        if isinstance(f_j, SequenceABC) and not isinstance(f_j, (str, bytes)):
            component_func = cast(VectorToScalarFunc, f_j[j])
        else:
            component_provider = cast(Callable[[Vector, int], Scalar], f_j)

            def indexed_component(x_val: Vector) -> Scalar:
                return component_provider(x_val, j)

            component_func = indexed_component

        computation = gradient(f=component_func, x=x, h=h, zero_tol=zero_tol, quasi_Fermat=True, monotonicity=False)

        component_specular_gradient = np.asarray(computation[0], dtype=float)
        norm = np.linalg.norm(component_specular_gradient)

        if not np.isfinite(norm):
            raise FloatingPointError("Component specular gradient norm is not finite.")

        if norm < tol:
            stop_reason = "gradient norm below tolerance"
            break

        d_k = component_specular_gradient / norm

        if isinstance(step_size, LineSearch):
            t_k = step_size(
                f=component_func,
                x=x,
                direction=-d_k,
                gradient_current=component_specular_gradient,
                gradient_f=lambda z: np.asarray(
                    gradient(
                        f=component_func,
                        x=z,
                        h=h,
                        zero_tol=zero_tol,
                        quasi_Fermat=True,
                        monotonicity=False,
                    )[0],
                    dtype=float,
                )
            )
        else:
            t_k = step_size(k)

        x -= t_k * d_k

    return x, f(x), k, stop_reason


def BFGS_method(
    f: ScalarToScalarFunc | VectorToScalarFunc,
    x_0: Scalar | Vector,
    h: float = 1e-6,
    tol: float = 1e-6,
    zero_tol: float = 1e-8,
    max_iter: int = 1000,
    line_search: str | LineSearch = "armijo",
    alpha_0: float = 1.0,
    c_1: float = 1e-4,
    c_2: float = 0.9,
    rho: float = 0.5,
    max_line_iter: int = 20,
    max_alpha: float = 1e8,
    raise_on_fail: bool = False,
    H_0: np.ndarray | list | None = None,
    record_history: bool = True,
    fill_iteration: bool = False,
    print_bar: bool = True,
) -> OptimizationResult:
    """
    The specular BFGS method for minimizing a nonsmooth convex function.
    """
    if h is None or h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x = np.asarray(x_0, dtype=float).reshape(-1).copy()
    n = x.size

    if n <= 1:
        raise ValueError(
            "BFGS requires n > 1. For 1D, use the standard specular gradient method."
        )

    f = cast(VectorToScalarFunc, f)

    def gradient_f(z: Vector) -> Vector:
        return np.asarray(
            gradient(
                f=f,
                x=np.asarray(z, dtype=float).reshape(-1),
                h=h,
                zero_tol=zero_tol,
                quasi_Fermat=True,
                monotonicity=False,
            )[0],
            dtype=float,
        ).reshape(-1)
    
    if isinstance(line_search, LineSearch):
        line_search_rule = line_search
        if line_search_rule.f is None:
            line_search_rule.f = f
        if line_search_rule.gradient_f is None:
            line_search_rule.gradient_f = gradient_f
    else:
        line_search_rule = LineSearch(
            name=line_search,
            alpha_0=alpha_0,
            c_1=c_1,
            c_2=c_2,
            rho=rho,
            max_iter=max_line_iter,
            max_alpha=max_alpha,
            raise_on_fail=raise_on_fail,
            f=f,
            gradient_f=gradient_f,
        )

    all_history = {}
    x_history = []
    f_history = []

    start_time = time.time()

    I = np.eye(n)

    if H_0 is None:
        H = I.copy()
    else:
        H = np.asarray(H_0, dtype=float)

        if H.shape != (n, n):
            raise ValueError(f"H_0 must have shape {(n, n)}. Got {H.shape}")

        H = H.copy()

    computation = gradient(
        f=f,
        x=x,
        h=h,
        zero_tol=zero_tol,
        quasi_Fermat=True,
        monotonicity=False,
    )
    spec_grad = np.asarray(computation[0], dtype=float).reshape(-1)
    k = 0

    stop_reason = "max_iter reached"
    completed_iteration = 0

    for k in tqdm(
        range(1, max_iter + 1),
        desc="Running the specular BFGS method",
        disable=not print_bar,
        leave=False,
    ):
        f_current = float(f(x))

        if record_history:
            x_history.append(x.copy())
            f_history.append(f_current)

        norm_g = np.linalg.norm(spec_grad)

        if not np.isfinite(norm_g):
            raise FloatingPointError("Specular gradient norm is not finite.")

        if norm_g < tol:
            stop_reason = "gradient norm below tolerance"
            break

        d_k = -H.dot(spec_grad)
        initial_slope = float(np.dot(spec_grad, d_k))

        if initial_slope >= 0.0:
            H = I.copy()
            d_k = -spec_grad

        try:
            t_k = line_search_rule(
                x=x,
                direction=d_k,
                gradient_current=spec_grad,
            )
        except (LineSearchError, ZeroDivisionError, FloatingPointError) as exc:
            stop_reason = f"line search failed: {exc}"
            break

        s = t_k * d_k
        x_new = x + s

        computation_new = gradient(
            f=f,
            x=x_new,
            h=h,
            zero_tol=zero_tol,
            quasi_Fermat=True,
            monotonicity=False,
        )
        spec_grad_new = np.asarray(computation_new[0], dtype=float).reshape(-1)

        y = spec_grad_new - spec_grad
        ys = float(np.dot(y, s))

        if not np.isfinite(ys) or ys == 0.0:
            stop_reason = "curvature denominator is zero or non-finite"
            break

        bfgs_rho = 1.0 / ys
        V = I - bfgs_rho * np.outer(s, y)
        H = V.dot(H).dot(V.T) + bfgs_rho * np.outer(s, s)

        x = x_new
        spec_grad = spec_grad_new
        computation = computation_new
        completed_iteration = k

    runtime = time.time() - start_time

    if record_history:
        all_history["variables"] = x_history
        all_history["values"] = f_history

    return OptimizationResult(
        method=f"specular BFGS ({line_search_rule.name})",
        solution=x,
        func_val=float(f(x)),
        iteration=completed_iteration,
        runtime=runtime,
        all_history=all_history,
        fill_iteration=fill_iteration,
        max_iter=max_iter,
        stop_reason=stop_reason
    )
