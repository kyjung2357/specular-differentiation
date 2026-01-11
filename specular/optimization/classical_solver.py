from .result import OptimizationResult
from .step_size import StepSize
import time
import torch
import numpy as np
from scipy.optimize import minimize
from typing import Callable, Union

def gradient_descent_method(
    f_torch: Callable[[torch.Tensor], torch.Tensor], 
    x_0: Union[np.ndarray, list], 
    step_size: StepSize, 
    max_iter: int = 100
) -> OptimizationResult:
    """
    Performs optimization using standard gradient descent.

    Returns
    -------
    OptimizationResult
        The result of the optimization containing the solution, function value, number of iterations, runtime, and history.
    """
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
    
    Returns
    -------
    OptimizationResult
        The result of the optimization containing the solution, function value, number of iterations, runtime, and history.
    """
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

def BFGS(
    f_np: Callable[[np.ndarray], float], 
    x_0: np.ndarray, 
    max_iter: int = 100, 
    tol: float = 1e-6
) -> OptimizationResult:
    """
    Performs optimization using the BFGS algorithm from SciPy.

    Returns
    -------
    OptimizationResult
        The result of the optimization containing the solution, function value, number of iterations, runtime, and history.
    """
    start_time = time.time()
    
    x_history = [np.array(x_0).copy()]
    f_history = [f_np(x_0)]

    def bfgs_callback(x_k):
        x_val = np.array(x_k).copy()
        f_val = f_np(x_k)
        
        x_history.append(x_val)
        f_history.append(f_val)

    result = minimize(
        f_np,
        x_0,
        method='BFGS',
        callback=bfgs_callback,
        options={'maxiter': max_iter, 'gtol': tol}
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
