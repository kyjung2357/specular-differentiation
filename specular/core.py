"""
==================================================
Calculations of specular directional derivatives
==================================================

This module provides implementations of the calculations of specular directional derivatives.
"""

from typing import Callable
import math
import numpy as np
# from ._validators import check_integer_index_i, check_positive_h, check_types_array_like_x_v, check_types_array_only_x

def A(
    alpha: float | np.floating | int,
    beta: float | np.floating | int,
    epsilon: float | np.floating = 1e-6
) -> float:
    """
    Compute the specular derivative from one-sided directional derivatives.

    Given real numbers `alpha` and `beta`, the function `A:R^2 -> R` is defined by 
    
        A(alpha, beta) = (alpha * beta - 1 + sqrt((1 + alpha^2)(1 + beta^2))) / (alpha + beta)

    if alpha + beta != 0; otherwise, it returns 0.

    Parameters
    ----------
    alpha : float, np.floating
        One-sided directional derivative.
    beta : float, np.floating
        One-sided directional derivative.
    epsilon : float, np.floating
        A small threshold used to determine if the denominator (alpha + beta) is close to zero for numerical stability. (default: 1e-6)

    Returns
    -------
    float
        The function A.

    Raises
    ------
    TypeError
        If `alpha` or `beta` are invalid types after conversion (e.g., list, dict).
    ValueError
        If a complex input has a non-zero imaginary part.

    Examples
    --------
    >>> import specular
    >>> specular.A(1.0, 2.0)
    1.3874258867227933
    """
    alpha = float(alpha)
    beta = float(beta)
    

    if abs(alpha + beta) <= epsilon:
        return 0.0
    else:
        numerator = alpha * beta - 1.0 + math.sqrt((1.0 + alpha**2) * (1.0 + beta**2))
        return numerator / (alpha + beta)

def derivative(
    f: Callable[[float | np.floating], float | np.floating],
    x: float | np.floating | int,
    h: float | np.floating = 1e-6
) -> float:
    """
    Approximates the specular derivative of a real-valued function `f: R -> R` at point `x`.

    This is computed using the `specular_directional_derivative` function in the direction `v=1.0`.

    Parameters
    ----------
    f : callable
        A real-valued function of a single real variable.
    x : float, np.floating, int
        The point at which the derivative is evaluated.
    h : float, np.floating
        Step size for the finite difference approximation (default: 1e-6).

    Returns
    -------
    float
        The approximated specular derivative of f at x in direction +1.

    Raises
    ------
    TypeError
        If the type of `x` is not a scalar (float) or array-like (list, np.ndarray).
    ValueError
        If the step size `h` is not positive (i.e., h <= 0).
    
    Examples
    --------
    >>> import specular

    >>> f = lambda x: max(x, 0.0)
    >>> specular.derivative(f, x=0.0)
    0.41421356237309515

    >>> f = lambda x: abs(x)
    >>> specular.derivative(f, x=0.0)
    0.0
    """
    try:
        x = float(x)
        h = float(h)
    except TypeError:
        raise TypeError(f"Input 'x' must be a scalar (float/int). Got {type(x).__name__}. Use `directional_derivative` for vectors.")
    
    if h <= 0:
        raise ValueError(f"Step size 'h' must be positive. Got {h}")
    
    alpha = (f(x + h) - f(x))/h
    beta = (f(x) - f(x - h))/h

    return A(alpha, beta)

def directional_derivative(
    f: Callable[[list | np.ndarray], float | np.floating],
    x: list | np.ndarray,
    v: list | np.ndarray,
    h: float | np.floating = 1e-6
) -> float:
    """
    Approximates the specular directional derivative of a function `f: R^n -> R` at a point `x` 
    in the direction `v`, using finite differences and the averaging operator `A`.

    This method computes one-sided finite differences from both directions (forward and backward)
    and applies the function `A(alpha, beta)` to return a specular directional derivative.

    Parameters
    ----------
    f : callable
        A real-valued function defined on an open subset of R^n.
    x : list, np.ndarray
        The point at which the derivative is evaluated.
    v : list, np.ndarray
        The direction in which the derivative is taken.
    h : float, np.floating
        The step size used in the finite difference approximation (default: 1e-6). Must be positive.

    Returns
    -------
    float
        The approximated specular directional derivative of `f` at `x` in the direction `v`.

    Raises
    ------
    TypeError
        If `x` or `v` are not of valid array-like types.
    ValueError
        If `h <= 0`.

    Examples
    --------
    >>> import specular
    >>> import math

    >>> f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    >>> specular.directional_derivative(f, x=[0.0, 0.1, -0.1], v=[1.0, -1.0, 2.0])
    -2.1213203434708223
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    if x.ndim == 0 or v.ndim == 0:
        raise ValueError(
            "Inputs 'x' and 'v' must be vectors (arrays). For scalar inputs, use `specular.derivative`."
        )
    
    if x.shape != v.shape:
        raise ValueError(f"Shape mismatch: x {x.shape} vs v {v.shape}")
    
    if h <= 0:
        raise ValueError(f"Step size 'h' must be positive. Got {h}")
    
    alpha = (f(x + h * v) - f(x))/h
    beta = (f(x) - f(x - h * v))/h

    return A(alpha, beta)

def partial_derivative(
    f: Callable[[list | np.ndarray], float | np.floating],
    x: list | np.ndarray,
    i: int | np.integer,
    h: float| np.floating = 1e-6
) -> float:
    """
    Approximates the i-th specular partial derivative of a real-valued function `f: R^n -> R` at point `x` for n > 1.

    This is computed using the `specular_directional_derivative` function with the direction of the `i`-th standard basis vector of `R^n`.

    Parameters
    ----------
    f : callable
        A real-valued function defined on R^n.
    x : list, np.ndarray
        The point at which the derivative is evaluated.
    i : int, np.integer
        the direction 1 <= i <= n of x_i-th partial derivative.
    h : float, np.floating
        Step size for the finite difference approximation (default: 1e-6).

    Returns
    -------
    float
        The approximated i-th partial specular derivative of f at x.

    Raises
    ------
    TypeError
        If `i` is not an integer.
    ValueError
        If `i` is out of valid range (1 <= i <= dimension of x).

    Examples
    --------
    >>> import specular
    >>> import math 
    
    >>> f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    >>> specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=2)
    0.8859268982863702
    """
    x = np.asarray(x, dtype=float)

    if not isinstance(i, (int, np.integer)):
        raise TypeError(f"Index 'i' must be an integer. Got {type(i).__name__}")
    
    n = x.size
    if i < 1 or i > n:
        raise ValueError(f"Index 'i' must be between 1 and {n} (dimension of x). Got {i}")

    e_i = np.zeros_like(x)
    e_i[i - 1] = 1.0

    return directional_derivative(f, x, e_i, h)

def gradient(
    f: Callable[[list | np.ndarray], float | np.floating],
    x: list | np.ndarray,
    h: float| np.floating = 1e-6
) -> np.ndarray:
    """
    Approximates the specular gradient of a real-valued function `f: R^n -> R` at point `x` for n > 1.

    The specular gradient is defined as the vector of all partial specular derivatives 
    along the standard basis directions. Each component is computed using the 
    `specular_partial_derivative` function.

    Parameters
    ----------
    f : callable
        A real-valued function defined on R^n.
    x : list, np.ndarray
        The point at which the specular gradient is evaluated.
    h : float, np.floating
        Step size for the finite difference approximation (default: 1e-6).

    Returns
    -------
    np.ndarray
        A vector (NumPy array) representing the specular gradient of f at x.

    Examples
    --------
    >>> import specular
    >>> import numpy as np

    >>> f = lambda x: np.linalg.norm(x)
    >>> specular.gradient(f, x=[1.4, -3.47, 4.57, 9.9])
    array([ 0.12144298, -0.3010051 ,  0.39642458,  0.85877534])
    """
    x = np.asarray(x, dtype=float)
    
    if x.ndim == 0:
        raise ValueError(
            "Input 'x' must be a vector. Use `specular.derivative` for scalar inputs."
        )
    
    result = np.zeros_like(x)

    for i in range(len(x)):
        result[i] = partial_derivative(f, x, i+1, h=h) 

    return result