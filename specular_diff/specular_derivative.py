"""
==================================================
Calculations of specular directional derivatives
==================================================

This module provides implementations of the calculations of specular directional derivatives.
"""

from typing import Callable
import numpy as np
from ._validators import check_integer_index_i, check_positive_h, check_types_array_like_x_v, check_types_array_only_x

def _to_float(x):
    if isinstance(x, complex):
        if np.isclose(x.imag, 0):
            return float(x.real)
        else:
            raise ValueError(f"Complex value with non-zero imaginary part: {x}")
    return float(x)

def A(
        alpha: float,
        beta: float,
        epsilon: float = 0
        ) -> float:
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
    TypeError
        If `alpha` or `beta` are invalid types after conversion (e.g., list, dict).
    ValueError
        If a complex input has a non-zero imaginary part.

    Examples
    --------
    >>> A(1.0, 2.0)
    1.2295687883848642
    """
    alpha = _to_float(alpha)
    beta = _to_float(beta)
    
    if not isinstance(alpha, (float, np.floating)):
        raise TypeError(f"Internal variable 'alpha' must be a float. Got type {type(alpha).__name__}")
    if not isinstance(beta, (float, np.floating)):
        raise TypeError(f"Internal variable 'beta' must be a float. Got type {type(beta).__name__}")
    
    if np.abs(alpha + beta) <= epsilon:
        return 0.0
    else:
        return float((alpha*beta - 1 + np.sqrt((1 + alpha**2)*(1 + beta**2)))/(alpha + beta))


@check_positive_h 
@check_types_array_like_x_v
def specular_directional_derivative(
        f: Callable[[np.ndarray], float],
        x: float | list | np.ndarray,
        v: float | list | np.ndarray,
        h: float = 1e-6
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
    x : float or array_like
        The point at which the derivative is evaluated.
    v : float or array_like
        The direction in which the derivative is taken.
    h : float, optional
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
    >>> import math

    One-dimensional input:
    >>> f = lambda x: max(x, 0)
    >>> specular_directional_derivative(f, x=0.0, v=1)
    0.41421356237309515

    Three-dimensional input:
    >>> f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    >>> specular_directional_derivative(f, x=[0.0, 0.1, -0.1], v=[1.0, -1.0, 2.0])
    -2.1213203434708223
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    alpha = (f(x + h * v) - f(x))/h
    beta = (f(x) - f(x - h * v))/h

    return A(alpha, beta)


@check_positive_h
@check_types_array_like_x_v
def specular_derivative(
        f: Callable[[np.ndarray], float],
        x: float | list | np.floating,
        h: float = 1e-6
        ) -> float:
    """
    Approximates the specular derivative of a real-valued function `f: R -> R` at point `x`.

    This is computed using the `specular_directional_derivative` function in the direction `v=1.0`.

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
    TypeError
        If the type of `x` is not a scalar (float) or array-like (list, np.ndarray).
    ValueError
        If the step size `h` is not positive (i.e., h <= 0).
    
    Examples
    --------
    >>> f = lambda x: max(x, 0.0)
    >>> specular_derivative(f, x=0.0)
    0.41421356237309515

    >>> f = lambda x: abs(x)
    >>> specular_derivative(f, x=0.0)
    0.0
    """
    return specular_directional_derivative(f, x, 1.0, h=h)  # type: ignore


@check_integer_index_i
@check_types_array_only_x
def specular_partial_derivative(
        f: Callable[[np.ndarray], float],
        x: list | np.ndarray,
        i: int,
        h: float = 1e-6
        ) -> float:
    """
    Approximates the i-th specular partial derivative of a real-valued function `f: R^n -> R` at point `x` for n > 1.

    This is computed using the `specular_directional_derivative` function with the direction of the `i`-th standard basis vector of `R^n`.

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
    TypeError
        If `x` is not a list or np.ndarray, or if `i` is not an integer.
    ValueError
        If `x` has length less than 2.

    Examples
    --------
    >>> import math 
    
    >>> f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    >>> specular_partial_derivative(f, x=[0.1, 2.3, -1.2], i=2)
    0.8859268982863702
    """
    x = np.asarray(x, dtype=float)
    e_i = np.zeros_like(x)
    e_i[i-1] = 1.0

    return specular_directional_derivative(f, x, e_i, h=h)


@check_types_array_only_x
def specular_gradient(
        f: Callable[[np.ndarray], float],
        x: list | np.ndarray,
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
    >>> import numpy as np

    >>> f = lambda x: np.linalg.norm(x)
    >>> specular_gradient(f, x=[1.4, -3.47, 4.57, 9.9])
    array([ 0.85877534,  0.12144298, -0.3010051 ,  0.39642458])
    """
    result = np.zeros_like(x)

    for i in range(len(x)):
        result[i] = specular_partial_derivative(f, x, i, h=h) 

    return result