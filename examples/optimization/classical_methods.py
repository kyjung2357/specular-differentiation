import numpy as np
import torch
import random
from functools import partial
from scipy.optimize import minimize
from typing import Callable, Union, List, Tuple

def Adam(
    f_torch: Callable[[torch.Tensor], torch.Tensor], 
    x_0: Union[np.ndarray, list], 
    step_size: float = 0.001, 
    max_iter: int = 100
) -> Tuple[np.ndarray, List[float]]:
    """
    Performs optimization using the Adam algorithm from PyTorch.

    This function minimizes a given objective function `f_torch` starting from an
    initial point `x_0` using PyTorch's built-in Adam optimizer.

    Parameters
    ----------
    f_torch : callable
        The objective function to be minimized. It should accept a PyTorch tensor
        and return a scalar PyTorch tensor.
    x_0 : np.ndarray | list
        The starting point for the optimization.
    step_size : float, optional
        The step size for the Adam optimizer. 
        Default is 0.001.
    max_iter : int, optional
        The maximum number of iterations to perform. 
        Default is 100.

    Returns
    -------
    Tuple[np.ndarray, List[float]]
        A tuple containing:
        - x_final: The final optimized point as a NumPy array.
        - values: A list of the objective function values at each iteration.
    """
    x = torch.tensor(x_0, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=step_size)

    values = [f_torch(x.detach()).item()] 

    for _ in range(max_iter):
        optimizer.zero_grad()
        loss = f_torch(x)
        loss.backward()
        optimizer.step()
        values.append(loss.item())

    return x.detach().numpy(), values

def BFGS(
    f_np: Callable[[np.ndarray], float], 
    x_0: np.ndarray, 
    max_iter: int = 100, 
    gtol: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    Performs optimization using the BFGS algorithm from SciPy.

    This function wraps the `scipy.optimize.minimize` call to provide a
    consistent interface with other custom optimizers.

    Parameters
    ----------
    f_np : callable
        The objective function to be minimized. 
    x_0 : np.ndarray
        The starting point for the optimization.
    max_iter : int, optional
        The maximum number of iterations to perform. 
        Default is 100.
    gtol : float, optional
        Tolerance for the gradient norm. The iteration will stop when the gradient's norm is less than this value. 
        Default is 1e-6.

    Returns
    -------
    tuple[np.ndarray, list[float]]
        A tuple containing:
        - x_final: The final optimized point.
        - values: A list of the objective function values at each iteration.
    """
    values = [f_np(x_0)]

    def bfgs_callback(x_k):
        values.append(f_np(x_k))

    result = minimize(f_np,
                      x_0,
                      method='BFGS',
                      callback=bfgs_callback,
                      options={'maxiter': max_iter, 'gtol': gtol})

    return result.x, values

def Gradient_descent_method(
    f_torch: Callable[[torch.Tensor], torch.Tensor], 
    x_0: Union[np.ndarray, list], 
    step_size: float = 0.001, 
    max_iter: int = 100
) -> Tuple[np.ndarray, List[float]]: # type: ignore
    """
    Performs optimization using the standard gradient descent algorithm.

    This function minimizes a given objective function `f_torch` starting from an
    initial point `x_0`. The update is performed manually using PyTorch's
    autograd engine to compute gradients.

    Parameters
    ----------
    f_torch : callable
        The objective function to be minimized. It should accept a PyTorch tensor
        and return a scalar PyTorch tensor.
    x_0 : np.ndarray | list
        The starting point for the optimization.
    step_size : float, optional
        The step size for the gradient descent steps. 
        Default is 0.001.
    max_iter : int, optional
        The maximum number of iterations to perform. 
        Default is 100.

    Returns
    -------
    Tuple[np.ndarray, List[float]]
        A tuple containing:
        - x_final: The final optimized point as a NumPy array.
        - values: A list of the objective function values at each iteration.
    """
    x = torch.tensor(x_0, dtype=torch.float32, requires_grad=True)

    values = [f_torch(x.detach()).item()]

    for _ in range(max_iter):
        if x.grad is not None:
            x.grad.zero_()

        loss = f_torch(x)

        loss.backward()

        with torch.no_grad():
            x -= step_size * x.grad # type: ignore

        values.append(loss.item())

    return x.detach().numpy(), values