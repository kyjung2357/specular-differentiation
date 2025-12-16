"""
==================================================
Calculations of specularly directional derivatives
==================================================

This module provides implementations of the calculations of specularly directional derivatives.

==========================
__author__ = "Kiyuob Jung"
__version__ = "1.1.0"
__license__ = "MIT"
"""


from typing import Callable, Union
import numpy as np

def safe_to_float(x):
    if isinstance(x, complex):
        if np.isclose(x.imag, 0):
            return float(x.real)
        else:
            raise ValueError(f"Complex value with non-zero imaginary part: {x}")
    return float(x)

def A(alpha: float, beta: float, epsilon: float = 0) -> float:
    """
    Compute the specular derivative from one-sided directional derivatives.

    Given real numbers `alpha` and `beta`, the function `A:R^2 -> R` is defined by 
    
        A(alpha, beta) = (alpha * beta - 1 + sqrt((1 + alpha^2)(1 + beta^2))) / (alpha + beta)

    if alpha + beta != 0; otherwise, it returns 0.

    Parameters
    ----------
    alpha : float
        One-sided directional derivative.
    beta : float
        One-sided directional derivative.

    Returns
    -------
    float
        The specular derivative.

    Raises
    ------
    AssertionError
        If `alpha` or `beta` is not a float.

    Examples
    --------
    >>> A(1.0, 2.0)
    1.2295687883848642
    """
    alpha = safe_to_float(alpha)
    beta = safe_to_float(beta)

    assert isinstance(alpha, (int, float, np.floating)), "alpha must be a float. {} is of type {}".format(alpha, type(alpha))
    assert isinstance(beta, (int, float, np.floating)), "beta must be a float. {} is of type {}".format(beta, type(beta))

    if np.abs(alpha + beta) <= epsilon:
        return 0.0
    else:
        return float((alpha*beta - 1 + np.sqrt((1 + alpha**2)*(1 + beta**2)))/(alpha + beta))


def specularly_directional_derivative(
        f: Callable[[np.ndarray], float],
        x: Union[float, list, np.ndarray],
        v: Union[float, list, np.ndarray],
        h: float = 1e-6
    ) -> float:
    """
    Approximates the specularly directional derivative of a function `f: R^n -> R` at a point `x` 
    in the direction `v`, using finite differences and the averaging operator `A`.

    This method computes one-sided finite differences from both directions (forward and backward)
    and applies the function `A(alpha, beta)` to return a specularly directional derivative.

    Parameters
    ----------
    f : callable
        A real-valued function defined on an open subset of R^n.
    x : float or array_like
        The point at which the derivative is evaluated.
    v : float or array_like
        The direction in which the derivative is taken.
    h : float, optional
        The step size used in the finite difference approximation (default: 1e-6). Must be positive.

    Returns
    -------
    float
        The approximated specularly directional derivative of `f` at `x` in the direction `v`.

    Raises
    ------
    AssertionError
        If `x`, `v`, or `h` are not of valid types or if `h <= 0`.

    Examples
    --------
    >>> import specular_derivative as sd
    >>> import math

    One-dimensional input:
    >>> f = lambda x: max(x, 0)
    >>> sd.specularly_directional_derivative(f, x=0.0, v=1)
    0.41421356237309515

    Three-dimensional input:
    >>> f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    >>> sd.specularly_directional_derivative(f, x=[0.0, 0.1, -0.1], v=[1.0, -1.0, 2.0])
    -2.1213203434708223
    """
    assert isinstance(x, (int, float, list, np.ndarray)), "x must be a int, float, list or a numpy array"
    assert isinstance(v, (int, float, list, np.ndarray)), "v must be a int, float, list or a numpy array"
    assert isinstance(h, (int, float, np.floating)), "h must be a int or float (positive real number)"
    assert h > 0, "h must be positive"

    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    alpha = (f(x + h * v) - f(x))/h
    beta = (f(x) - f(x - h * v))/h

    return A(alpha, beta)


def specular_derivative(
        f: Callable[[np.ndarray], float],
        x: Union[float, list, np.floating],
        h: float = 1e-6
    ) -> float:
    """
    Approximates the specular derivative of a real-valued function `f: R -> R` at point `x`.

    This is computed using the `specularly_directional_derivative` function in the direction `v=1.0`.

    Parameters
    ----------
    f : callable
        A real-valued function of a single real variable.
    x : float
        The point at which the derivative is evaluated.
    h : float, optional
        Step size for the finite difference approximation (default: 1e-6).

    Returns
    -------
    float
        The approximated specular derivative of f at x in direction +1.

    Raises
    ------
    AssertionError
        If x is not a scalar (int or float).
    
    Examples
    --------
    >>> import specular_derivative as sd

    >>> f = lambda x: max(x, 0.0)
    >>> sd.specular_derivative(f, x=0.0)
    0.41421356237309515

    >>> f = lambda x: abs(x)
    >>> sd.specular_derivative(f, x=0.0)
    0.0
    """
    assert isinstance(x, (int, float, np.floating)), "x must be a int or float"
    
    return specularly_directional_derivative(f, float(x), 1.0, h=h) 


def specular_partial_derivative(
        f: Callable[[np.ndarray], float],
        x: Union[list, np.ndarray],
        i: int,
        h: float = 1e-6
    ) -> float:
    """
    Approximates the i-th specular partial derivative of a real-valued function `f: R^n -> R` at point `x` for n > 1.

    This is computed using the `specularly_directional_derivative` function with the direction of the `i`-th standard basis vector of `R^n`.

    Parameters
    ----------
    f : callable
        A real-valued function defined on R^n.
    x : list or np.ndarray
        The point at which the derivative is evaluated.
    i : int
        The coordinate index (starting from 0) indicating the direction of differentiation.
    h : float, optional
        Step size for the finite difference approximation (default: 1e-6).

    Returns
    -------
    float
        The approximated i-th partial specular derivative of f at x.

    Raises
    ------
    AssertionError
        If x is not a list or np.ndarray.
        If i is not an integer or out of bounds.
        If x has length less than 2.

    Examples
    --------
    >>> import specular_derivative as sd
    >>> import math 
    
    >>> f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    >>> sd.specular_partial_derivative(f, x=[0.1, 2.3, -1.2], i=2)
    -0.4622227292028128
    """
    assert isinstance(x, (list, np.ndarray)), "x must be a list or a numpy array"
    assert isinstance(i, int), "i must be an integer"
    assert len(x) >= 2, "x must have length at least 2; use the function `specular_derivative` for the one dimension"

    x = np.asarray(x, dtype=float)
    e_i = np.zeros_like(x)
    e_i[i] = 1.0

    return specularly_directional_derivative(f, x, e_i, h=h)


def specular_gradient(
        f: Callable[[np.ndarray], float],
        x: Union[list, np.ndarray],
        h: float = 1e-6
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
    x : list or np.ndarray
        The point at which the specular gradient is evaluated.
    h : float, optional
        Step size for the finite difference approximation (default: 1e-6).

    Returns
    -------
    np.ndarray
        A vector (NumPy array) representing the specular gradient of f at x.

    Examples
    --------
    >>> import specular_derivative as sd
    >>> import numpy as np

    >>> f = lambda x: np.linalg.norm(x)
    >>> sd.specular_gradient(f, x=[1.4, -3.47, 4.57, 9.9])
    array([ 0.12144298, -0.3010051 ,  0.39642458,  0.85877534])
    """
    result = np.zeros_like(x)

    for i in range(len(x)):
        result[i] = specular_partial_derivative(f, x, i, h=h) 

    return result