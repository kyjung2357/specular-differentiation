import numpy as np
from tqdm import tqdm
import time
import inspect
from typing import Callable, TypeAlias, Sequence
from collections.abc import Sequence as SequenceABC
from .result import OptimizationResult
from .line_search import LineSearch
from .step_size import StepSize
from ..calculation import derivative, gradient

SUPPORTED_METHODS = ['specular gradient', 'implicit', 'stochastic', 'hybrid']

ComponentFunc: TypeAlias = Callable[[int | float | np.number | list | np.ndarray], int | float | np.number]
ComponentFuncs: TypeAlias = Sequence[ComponentFunc] | Callable

def gradient_method(
    f: Callable[[int | float | np.number | list | np.ndarray], int | float | np.number],
    x_0: int | float | list | np.ndarray,
    step_size: StepSize,
    h: float = 1e-6,
    form: str = 'specular gradient',
    tol: float = 1e-6,
    zero_tol: float = 1e-8,
    max_iter: int = 1000,
    f_j: ComponentFuncs | None = None,
    m: int = 1,
    switch_iter: int | None = 2,
    record_history: bool = True,
    print_bar: bool = True
) -> OptimizationResult:
    """
    The specular gradient method for minimizing a nonsmooth convex function.

    Parameters:
        f (callable):
            The objective function to minimize.
        x_0 (int | float | list | np.ndarray):
            The starting point for the optimization.
        step_size (StepSize):
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
        m (int, optional):
            The number of component functions.
            Used for the stochastic and hybrid forms.
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
            res_x, res_f, res_k = _vector_stochastic(f, f_history, x, x_history, step_size, h, tol, zero_tol, f_j, m, max_iter, record_history, print_bar) # type: ignore

        elif form == 'hybrid':
            if f_j is None:
                raise ValueError("Component functions 'f_j' must be provided for the stochastic form.")
            
            # Phase 1: deterministic
            form = 'hybrid specular gradient'
            switch_iter = switch_iter if switch_iter is not None else max_iter
            remaining_iter = max_iter - switch_iter

            # Phase 2: stochastic
            res_x, res_f, res_k = _vector(
                f, f_history, x, x_history, step_size, h, tol, zero_tol, switch_iter, record_history, print_bar
            )
            res_x, res_f, res_k = _vector_stochastic(
                f, f_history, res_x, x_history, step_size, h, tol, zero_tol, f_j, m, remaining_iter, record_history, print_bar, k_start=res_k
            ) # type: ignore

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
    f: Callable[[int | float | np.number], int | float | np.number | list | np.ndarray],
    f_history: list,
    x: int | float,
    x_history: list,
    step_size: StepSize,
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
    k = 1

    for _ in tqdm(range(1, max_iter + 1), desc="Running the specular gradient method", disable=not print_bar, leave=False):
        if record_history is True:
            x_history.append(x)
            f_history.append(f(x))

        specular_derivative = derivative(f=f, x=x, h=h, zero_tol=zero_tol)
        norm = np.linalg.norm(specular_derivative)
        if norm < tol:
            break
        
        x -= step_size(k)*(specular_derivative / norm) # type: ignore
        k += 1
    
    return x, f(x), k

def _scalar_implicit(
    f: Callable[[int | float | np.number], int | float | np.number],
    f_history: list,
    x: int | float,
    x_history: list,
    step_size: StepSize,
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
    k = 1

    for _ in tqdm(range(1, max_iter + 1), desc="Running the implicit specular gradient method", disable=not print_bar, leave=False):
        if record_history is True:
            x_history.append(x)
            f_history.append(f(x))

        # This is the sum of the right and left one-sided slopes, not a central difference.
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
    k = 1

    for _ in tqdm(range(1, max_iter + 1), desc="Running the specular gradient method", disable=not print_bar, leave=False):
        if record_history is True:
            x_history.append(x.copy())
            f_history.append(f(x))

        computation = gradient(f=f, x=x, h=h, zero_tol=zero_tol, quasi_Fermat=True, monotonicity=False)
        specular_gradient = computation[0]
        norm = np.linalg.norm(specular_gradient)

        if not np.isfinite(norm):
            raise FloatingPointError("Specular gradient norm is not finite.")

        if norm < tol:
            break

        x -= step_size(k)*(specular_gradient / norm)
        k += 1
    
    return x, f(x), k

def _vector_stochastic(
    f: Callable[[list | np.ndarray], int | float | np.number],
    f_history: list,
    x: list | np.ndarray,
    x_history: list,
    step_size: StepSize,
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
    k = k_start

    for _ in tqdm(range(1, max_iter + 1), desc="Running the stochastic specular gradient method", disable=not print_bar, leave=False):
        if record_history is True:
            x_history.append(x.copy())
            f_history.append(f(x)) 

        if isinstance(f_j, SequenceABC):
            num_components = len(f_j)
        else:
            num_components = m
        
        # A random index j is selected at each iteration
        j = np.random.randint(num_components)

        if isinstance(f_j, SequenceABC):
            component_func = f_j[j]
        else:
            if not callable(f_j):
                raise TypeError(f"f_j must be a list of functions or a callable. Got {type(f_j)} instead.")

            sig = inspect.signature(f_j)
            params = list(sig.parameters.values())
            has_varargs = any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)

            if len(params) < 2 and not has_varargs:
                raise ValueError(
                    f"The function f_j must accept at least 2 arguments (x and index). "
                    f"Current signature is: {sig}"
                )

            component_func = lambda x_val: f_j(x_val, j)

        computation = gradient(f=component_func, x=x, h=h, zero_tol=zero_tol, quasi_Fermat=True, monotonicity=False)

        component_specular_gradient = computation[0]
        norm = np.linalg.norm(component_specular_gradient)

        if not np.isfinite(norm):
            raise FloatingPointError("Component specular gradient norm is not finite.")

        if norm < tol:
            break

        x -= step_size(k)*(component_specular_gradient / norm)
        k += 1

    return x, f(x), k


def BFGS_method(
    f: Callable[[int | float | np.number | list | np.ndarray], int | float | np.number],
    x_0: int | float | list | np.ndarray,
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
    safeguard: float = 1e-10,
    record_history: bool = True,
    print_bar: bool = True,
) -> OptimizationResult:
    """
    The specular BFGS method for minimizing a nonsmooth convex function.
    """
    if h is None or h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    if safeguard < 0:
        raise ValueError(f"safeguard must be nonnegative. Got {safeguard}")

    x = np.asarray(x_0, dtype=float).reshape(-1).copy()
    n = x.size

    if n <= 1:
        raise ValueError(
            "BFGS requires n > 1. For 1D, use the standard specular gradient method."
        )

    if isinstance(line_search, LineSearch):
        line_search_rule = line_search
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
    iteration = 0

    for iteration in tqdm(
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
            break

        direction = -H.dot(spec_grad)
        initial_slope = float(np.dot(spec_grad, direction))

        if initial_slope >= 0.0:
            H = I.copy()
            direction = -spec_grad

        alpha = line_search_rule(
            f=f,
            x=x,
            direction=direction,
            gradient_current=spec_grad,
            f_current=f_current,
            gradient_func=lambda z: np.asarray(
                gradient(
                    f=f,
                    x=z,
                    h=h,
                    zero_tol=zero_tol,
                    quasi_Fermat=True,
                    monotonicity=False,
                )[0],
                dtype=float,
            ).reshape(-1),
        )

        s = alpha * direction
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

        if ys > safeguard:
            bfgs_rho = 1.0 / ys
            V = I - bfgs_rho * np.outer(s, y)
            H = V.dot(H).dot(V.T) + bfgs_rho * np.outer(s, s)

        x = x_new
        spec_grad = spec_grad_new
        computation = computation_new

    runtime = time.time() - start_time

    if record_history:
        all_history["variables"] = x_history
        all_history["values"] = f_history

    return OptimizationResult(
        method=f"specular BFGS ({line_search_rule.name})",
        solution=x,
        func_val=float(f(x)),
        iteration=iteration,
        runtime=runtime,
        all_history=all_history,
    )
