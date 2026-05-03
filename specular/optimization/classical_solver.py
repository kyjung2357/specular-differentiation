from __future__ import annotations

from .line_search import LineSearch
from .result import OptimizationResult
from .step_size import StepSize
import time
import numpy as np
from typing import Callable, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

def gradient_descent_method(
    f_torch: Callable[[torch.Tensor], torch.Tensor], 
    x_0: Union[np.ndarray, list], 
    step_size: StepSize, 
    max_iter: int = 100
) -> OptimizationResult:
    """
    Performs optimization using standard gradient descent.

    Returns:
        The result of the optimization containing the solution, function value, number of iterations, runtime, and history.
    """
    import torch
    start_time = time.time()
    
    x = torch.tensor(x_0, dtype=torch.float32, requires_grad=True)

    x_history = [x.detach().cpu().numpy().copy()]
    f_history = [f_torch(x.detach()).item()]

    for k in range(1, max_iter + 1):
        if x.grad is not None:
            x.grad.zero_()

        loss = f_torch(x)
        loss.backward()

        with torch.no_grad():
            if x.grad is not None:
                x -= step_size(k) * x.grad
        
        x_history.append(x.detach().cpu().numpy().copy())
        f_history.append(loss.item())

    end_time = time.time()

    return OptimizationResult(
        method="gradient descent",
        solution=x_history[-1],
        func_val=f_history[-1],
        iteration=max_iter,
        runtime=end_time - start_time,
        all_history={
            "variables": np.array(x_history),
            "values": np.array(f_history)
        }
    )

def Adam(
    f_torch: Callable[[torch.Tensor], torch.Tensor],
    x_0: Union[np.ndarray, list],
    step_size: StepSize | float,
    max_iter: int = 100
) -> OptimizationResult:
    """
    Performs optimization using the Adam algorithm from PyTorch.
    
    Returns:
        The result of the optimization containing the solution, function value, number of iterations, runtime, and history.
    """
    import torch
    start_time = time.time()
    
    x = torch.tensor(x_0, dtype=torch.float32, requires_grad=True)
    
    initial_lr = step_size(1) if callable(step_size) else step_size
    optimizer = torch.optim.Adam([x], lr=initial_lr)

    x_history = [x.detach().cpu().numpy().copy()]
    f_history = [f_torch(x).item()]

    for k in range(1, max_iter + 1):
        if callable(step_size):
            current_lr = step_size(k)

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        optimizer.zero_grad()
        loss = f_torch(x)
        loss.backward()
        optimizer.step()
        
        x_history.append(x.detach().cpu().numpy().copy())
        f_history.append(loss.item())

    end_time = time.time()
    
    return OptimizationResult(
        method="Adam",
        solution=x_history[-1],
        func_val=f_history[-1],
        iteration=max_iter,
        runtime=end_time - start_time,
        all_history={
            "variables": np.array(x_history),
            "values": np.array(f_history)
        }
    )

def _normalize_line_search_name(line_search: str) -> str:
    aliases = {
        "strong wolfe": "strong_wolfe",
        "strong-wolfe": "strong_wolfe",
    }
    name = line_search.lower()
    return aliases.get(name, name)


def _finite_difference_gradient(
    f_np: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float,
) -> np.ndarray:
    if eps <= 0:
        raise ValueError(f"eps must be positive. Got {eps}")

    grad = np.zeros_like(x, dtype=float)

    for idx in range(x.size):
        step = np.zeros_like(x, dtype=float)
        step[idx] = eps
        grad[idx] = (float(f_np(x + step)) - float(f_np(x - step))) / (2.0 * eps)

    return grad


def _native_BFGS(
    f_np: Callable[[np.ndarray], float],
    x_0: Union[np.ndarray, list],
    max_iter: int,
    tol: float,
    line_search: str | LineSearch,
    grad_np: Callable[[np.ndarray], np.ndarray | list | int | float | np.number] | None,
    eps: float | None,
    alpha_0: float,
    c_1: float,
    c_2: float,
    rho: float,
    max_line_iter: int,
    max_alpha: float,
    raise_on_fail: bool,
    H_0: np.ndarray | list | None,
    safeguard: float,
) -> OptimizationResult:
    if safeguard < 0:
        raise ValueError(f"safeguard must be nonnegative. Got {safeguard}")

    x = np.asarray(x_0, dtype=float).reshape(-1).copy()

    if x.size == 0:
        raise ValueError("x_0 must contain at least one variable.")

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

    def gradient_func(z: np.ndarray) -> np.ndarray:
        if grad_np is None:
            grad = _finite_difference_gradient(
                f_np,
                z,
                1e-8 if eps is None else eps,
            )
        else:
            grad = np.asarray(grad_np(z), dtype=float).reshape(-1)

        if grad.shape != z.shape:
            raise ValueError(
                f"Gradient shape mismatch: expected {z.shape}, got {grad.shape}"
            )

        return grad

    start_time = time.time()

    n = x.size
    I = np.eye(n)

    if H_0 is None:
        H = I.copy()
    else:
        H = np.asarray(H_0, dtype=float)

        if H.shape != (n, n):
            raise ValueError(f"H_0 must have shape {(n, n)}. Got {H.shape}")

        H = H.copy()

    g = gradient_func(x)
    iteration = 0

    x_history = [x.copy()]
    f_history = [float(f_np(x))]

    for k in range(1, max_iter + 1):
        if np.linalg.norm(g) < tol:
            break

        direction = -H.dot(g)

        if float(np.dot(g, direction)) >= 0.0:
            H = I.copy()
            direction = -g

        alpha = line_search_rule(
            f=f_np,
            x=x,
            direction=direction,
            gradient_current=g,
            f_current=f_history[-1],
            gradient_func=gradient_func,
        )

        s = alpha * direction
        x_new = x + s
        g_new = gradient_func(x_new)

        y = g_new - g
        ys = float(np.dot(y, s))

        if ys > safeguard:
            bfgs_rho = 1.0 / ys
            V = I - bfgs_rho * np.outer(s, y)
            H = V.dot(H).dot(V.T) + bfgs_rho * np.outer(s, s)

        x = x_new
        g = g_new
        iteration = k
        x_history.append(x.copy())
        f_history.append(float(f_np(x)))

    end_time = time.time()

    return OptimizationResult(
        method=f"BFGS ({line_search_rule.name})",
        solution=x,
        func_val=f_history[-1],
        iteration=iteration,
        runtime=end_time - start_time,
        all_history={
            "variables": np.array(x_history),
            "values": np.array(f_history),
        }
    )


def _BFGS_scipy(
    f_np: Callable[[np.ndarray], float],
    x_0: Union[np.ndarray, list],
    max_iter: int,
    tol: float,
    grad_np: Callable[[np.ndarray], np.ndarray | list | int | float | np.number] | None,
    eps: float | None,
    c_1: float,
    c_2: float,
) -> OptimizationResult:
    from scipy.optimize import minimize

    start_time = time.time()
    x = np.asarray(x_0, dtype=float).reshape(-1).copy()

    x_history = [x.copy()]
    f_history = [float(f_np(x))]

    def bfgs_callback(x_k):
        x_val = np.array(x_k).copy()
        f_val = f_np(x_k)

        x_history.append(x_val)
        f_history.append(f_val)

    scipy_options = {'maxiter': max_iter, 'gtol': tol}

    if eps is not None:
        scipy_options['eps'] = eps

    scipy_options['c1'] = c_1
    scipy_options['c2'] = c_2

    result = minimize(
        f_np,
        x,
        method='BFGS',
        jac=grad_np,
        callback=bfgs_callback,
        options=scipy_options
    )

    end_time = time.time()

    return OptimizationResult(
        method="BFGS",
        solution=result.x,
        func_val=result.fun,
        iteration=result.nit,
        runtime=end_time - start_time,
        all_history={
            "variables": np.array(x_history),
            "values": np.array(f_history)
        }
    )


def BFGS(
    f_np: Callable[[np.ndarray], float],
    x_0: Union[np.ndarray, list],
    max_iter: int = 100,
    tol: float = 1e-6,
    line_search: str | LineSearch = "strong_wolfe",
    grad_np: Callable[[np.ndarray], np.ndarray | list | int | float | np.number] | None = None,
    eps: float | None = None,
    alpha_0: float = 1.0,
    c_1: float = 1e-4,
    c_2: float = 0.9,
    rho: float = 0.5,
    max_line_iter: int = 20,
    max_alpha: float = 1e8,
    raise_on_fail: bool = False,
    H_0: np.ndarray | list | None = None,
    safeguard: float = 1e-10,
) -> OptimizationResult:
    """
    Performs optimization using the BFGS algorithm.

    The native BFGS implementation is used except when ``line_search`` is
    ``'strong_wolfe'``. In that case, SciPy's BFGS implementation is used.

    Returns:
        The result of the optimization containing the solution, function value, number of iterations, runtime, and history.
    """
    if (
        H_0 is None
        and isinstance(line_search, str)
        and _normalize_line_search_name(line_search) == "strong_wolfe"
    ):
        return _BFGS_scipy(
            f_np=f_np,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            grad_np=grad_np,
            eps=eps,
            c_1=c_1,
            c_2=c_2,
        )

    return _native_BFGS(
        f_np=f_np,
        x_0=x_0,
        max_iter=max_iter,
        tol=tol,
        line_search=line_search,
        grad_np=grad_np,
        eps=eps,
        alpha_0=alpha_0,
        c_1=c_1,
        c_2=c_2,
        rho=rho,
        max_line_iter=max_line_iter,
        max_alpha=max_alpha,
        raise_on_fail=raise_on_fail,
        H_0=H_0,
        safeguard=safeguard,
    )
