"""
========================================
Calculations of specular differentiation
========================================

This module provides implementations of the function :math:`\\mathcal{A}`, specular directional derivatives, specular partial derivatives, specular derivatives, specular gradients, and specular Jacobians.

Computations are based on the finite difference approximation of one-sided (directional) derivatives.
"""

from typing import Callable
import math
import numpy as np


def A(
    alpha: float | np.number | int | list | np.ndarray,
    beta: float | np.number | int | list | np.ndarray,
    zero_tol: float = 1e-8
) -> float | np.ndarray:
    """
    Compute the function :math:`\\mathcal{A}` from one-sided directional derivatives.

    Given real numbers ``alpha`` and ``beta``, the function :math:`\\mathcal{A}:\\mathbb{R}^2 \\to \\mathbb{R}` is defined by 
    
        :math:`\\mathcal{A}(\\alpha, \\beta) = \\frac{\\alpha \\beta - 1 + \\sqrt((1 + \\alpha^2)(1 + \\beta^2))}{\\alpha + \\beta}` if ``alpha + beta != 0``; otherwise, it returns ``0``.

    Parameters
    ----------
    alpha : float | np.number | int | list | np.ndarray
        One-sided directional derivative.
    beta : float | np.number | int | list | np.ndarray
        One-sided directional derivative.
    zero_tol : float, optional
        A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.
        Default: ``1e-8``.

    Returns
    -------
    float
        The function :math:`\\mathcal{A}`.

    Raises
    ------
    ValueError
        If ``alpha`` and ``beta`` have different shape.

    Examples
    --------
    >>> import specular
    >>> specular.calculation.A(1.0, 2.0)
    1.3874258867227933
    >>> specular.calculation.A([1.0, 2.4], [2.0, 4.1])
    array([1.38742589, 3.04807583])
    """
    if np.isscalar(alpha) and np.isscalar(beta):
        return _A_scalar(alpha, beta, zero_tol=zero_tol) # type: ignore
    
    return _A_vector(alpha, beta, zero_tol=zero_tol) # type: ignore


def _A_scalar(
    alpha: float | np.number, 
    beta: float | np.number, 
    zero_tol: float = 1e-8
) -> float:
    """Scalar implementation of :func:`A`."""
    denominator = alpha + beta

    if abs(denominator) <= zero_tol:
        return 0.0
    
    numerator = alpha * beta - 1.0 + math.sqrt((1.0 + alpha**2) * (1.0 + beta**2))

    return numerator / denominator # type: ignore


def _A_vector(
    alpha: list | np.ndarray, 
    beta: list | np.ndarray, 
    zero_tol: float = 1e-8
) -> np.ndarray:
    """Vector implementation of :func:`A`."""
    alpha = np.asanyarray(alpha, dtype=float)
    beta = np.asanyarray(beta, dtype=float)

    if alpha.shape != beta.shape:
        raise ValueError(f"Shape mismatch: alpha {alpha.shape} vs beta {beta.shape}")

    denominator = alpha + beta

    mask = np.abs(denominator) > zero_tol
    result = np.zeros_like(denominator)

    alpha_valid = alpha[mask]
    beta_valid = beta[mask]
    denominator_valid = denominator[mask]

    if alpha_valid.size > 0:
        numerator = alpha_valid * beta_valid - 1.0 + np.sqrt((1.0 + alpha_valid**2) * (1.0 + beta_valid**2))
        
        result[mask] = numerator / denominator_valid

    return result


def derivative(
    f: Callable[[int | float | np.number], int | float | np.number | list | np.ndarray],
    x: float | np.number | int,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float | np.ndarray:
    """
    Approximates the specular derivative of a function :math:`f:\\mathbb{R} \\to \\mathbb{R}^m` at a scalar point ``x``.
    
    If ``f`` returns a scalar, the result is a float.
    If ``f`` returns a vector, the result is a vector (component-wise derivative).


    Parameters
    ----------
    f : callable
        A function of a single real variable, returning a scalar or a vector.
    x : float | np.number | int 
        The point at which the derivative is evaluated.
    h : float, optional
        Mesh size used in the finite difference approximation. Must be positive.
        Default: ``1e-6``.
    zero_tol : float, optional 
        A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.
        Default: ``1e-8``.

    Returns
    -------
    float | np.ndarray
        The approximated specular derivative of ``f`` at ``x``.

    Raises
    ------
    TypeError
        If the type of ``x`` is not a scalar.
    ValueError
        If ``h`` is not positive.
    
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
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")
    
    try:
        x = float(x)

    except TypeError:
        raise TypeError(
            f"Input 'x' must be a scalar. "
            f"Got {type(x).__name__}. "
            "Use `specular.directional_derivative`, `specular.gradient`, or `specular.jacobian` for vectors inputs."
        )

    f_val = f(x)

    # f is real-valued
    if np.ndim(f_val) == 0:
        alpha = (f(x + h) - f_val) / h # type: ignore
        beta = (f_val - f(x - h)) / h # type: ignore
        
        return _A_scalar(alpha, beta, zero_tol=zero_tol) # type: ignore

    # f is vector-valued
    else:
        f_val = np.asarray(f_val, dtype=float)
        f_right = np.asarray(f(x + h), dtype=float)
        f_left = np.asarray(f(x - h), dtype=float)

        alpha = (f_right - f_val) / h
        beta = (f_val - f_left) / h
        
        return _A_vector(alpha, beta, zero_tol=zero_tol)


def directional_derivative(
    f: Callable[[list | np.ndarray], int | float | np.number],
    x: list | np.ndarray,
    v: list | np.ndarray,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float:
    """
    Approximates the specular directional derivative of a function :math:`f:\\mathbb{R}^n \\to \\mathbb{R}` at a point ``x`` in the direction ``v``.
    The one-sided derivatives ``alpha`` and ``beta`` are approximated.
    The function :math:`\\mathcal{A}` returns the specular directional derivative for each component of the vector ``x``.

    Parameters
    ----------
    f : callable
        A real-valued function defined on an open subset of :math:`\\mathbb{R}^n`.
    x : list | np.ndarray
        The point at which the derivative is evaluated.
    v : list | np.ndarray
        The direction in which the derivative is taken.
    h : float, optional
        Mesh size used in the finite difference approximation. Must be positive.
        Default: ``1e-6``.
    zero_tol : float, optional
        A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.
        Default: ``1e-8``.

    Returns
    -------
    float
        The approximated specular directional derivative of ``f`` at ``x`` in the direction ``v`` as a scalar.

    Raises
    ------
    TypeError
        If ``x`` or ``v`` are not of valid array-like types.
    ValueError
        If ``x`` and ``v`` have different shape.
        If ``h`` is not positive.

    Examples
    --------
    >>> import specular
    >>> import math
    >>> f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    >>> specular.directional_derivative(f, x=[0.0, 0.1, -0.1], v=[1.0, -1.0, 2.0])
    -2.1213203434708223
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")
    
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    if x.ndim == 0:
        raise TypeError(
            f"Input 'x' must be a vector. "
            f"Got {type(x).__name__}. "
            "Use `specular.derivative` for scalar inputs."
        )
    
    if v.ndim == 0:
        raise TypeError(
            "Input 'v' must be a vector. "
            f"Got {type(v).__name__}."
        )
    
    if x.shape != v.shape:
        raise ValueError(f"Shape mismatch: x {x.shape} vs v {v.shape}")
    
    f_val = f(x) 

    if np.ndim(f_val) != 0:
        raise ValueError(
            "Function f must return a scalar value. "
            f"Got shape {np.shape(f_val)}."
        )
    
    alpha = (f(x + h * v) - f_val)/h
    beta = (f_val - f(x - h * v))/h

    return _A_scalar(alpha, beta, zero_tol=zero_tol)


def partial_derivative(
    f: Callable[[list | np.ndarray], int |float | np.number],
    x: list | np.ndarray,
    i: int | np.integer,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float:
    """
    Approximates the i-th specular partial derivative of a real-valued function :math:`f:\\mathbb{R}^n \\to \\mathbb{R}` at point ``x`` for ``n > 1``.

    This is computed using :func:`specular_directional_derivative` with the direction of the ``i``-th standard basis vector of :math:`\\mathbb{R}^n`.

    Parameters
    ----------
    f : callable
        A real-valued function defined on :math:`\\mathbb{R}^n`.
        ``n`` is the dimension of ``x``.
    x : list | np.ndarray
        The point at which the derivative is evaluated.
    i : int | np.integer
        The index of the specular partial derivative with respect to :math:`x_i` (``1 <= i <= n``).
    h : float, optional
        Mesh size used in the finite difference approximation. Must be positive.
        Default: ``1e-6``.
    zero_tol : float, optional
        A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.
        Default: ``1e-8``.

    Returns
    -------
    float
        The approximated ``i``-th partial specular derivative of ``f`` at ``x`` as a scalar.

    Raises
    ------
    TypeError
        If ``i`` is not an integer.
    ValueError
        If ``i`` is out of the valid range (``1 <= i <= n``).

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

    return directional_derivative(f, x, e_i, h, zero_tol)


def gradient(
    f: Callable[[list | np.ndarray], int |float | np.number],
    x: list | np.ndarray,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> np.ndarray:
    """
    Approximates the specular gradient of a real-valued function :math:`f:\\mathbb{R}^n \\to \\mathbb{R}` at point ``x`` for ``n > 1``.

    The specular gradient is defined as the vector of all partial specular derivatives along the standard basis directions.

    Parameters
    ----------
    f : callable
        A real-valued function defined on :math:`\\mathbb{R}^n`.
    x : list | np.ndarray
        The point at which the specular gradient is evaluated.
    h : float, optional
        Mesh size used in the finite difference approximation. Must be positive.
        Default: ``1e-6``.
    zero_tol : float, optional
        A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.
        Default: ``1e-8``.

    Returns
    -------
    np.ndarray
        The approximated specular gradient of ``f`` at ``x`` as a vector.

    Raises
    ------
    TypeError
        If ``x`` is not of a valid array-like type.
    ValueError
        If ``f`` does not return a scalar value.
        If ``h`` is not positive.

    Examples
    --------
    >>> import specular
    >>> import numpy as np
    >>> f = lambda x: np.linalg.norm(x)
    >>> specular.gradient(f, x=[1.4, -3.47, 4.57, 9.9])
    array([ 0.12144298, -0.3010051 ,  0.39642458,  0.85877534])
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")
    
    x = np.asarray(x, dtype=float)
    
    if x.ndim != 1:
        raise TypeError(
            f"Input 'x' must be a vector. "
            f"Got {type(x).__name__} with shape {x.shape}. "
            "Use `specular.derivative` for scalar inputs."
        )
    
    n = x.size
    I = np.eye(n)

    f_val = f(x)

    if np.ndim(f_val) != 0:
        raise ValueError(
            "Function f must return a scalar value. "
            f"Got shape {np.shape(f_val)}."
        )

    x_right = x + h*I
    x_left = x - h*I

    f_right = np.array([f(row) for row in x_right], dtype=float)
    f_left = np.array([f(row) for row in x_left], dtype=float)
    
    # try:
    #     f_right = f(x_right)
    #     if np.ndim(f_right) != 1 or np.size(f_right) != n:
    #         raise ValueError (
    #             "Function f must return a scalar for each input vector. "
    #             f"Got shape {np.ndim(f_right)}."
    #         )
        
    # except Exception:
    #     f_right = np.array([f(row) for row in x_right])

    # try:
    #     f_left = f(x_left)
    #     if np.ndim(f_left) != 1 or np.size(f_left) != n:
    #         raise ValueError(
    #             "Function f must return a scalar for each input vector. "
    #             f"Got shape {np.ndim(f_left)}.")
        
    # except Exception:
    #     f_left = np.array([f(row) for row in x_left])

    alpha = (f_right - f_val) / h 
    beta = (f_val - f_left) / h 

    return _A_vector(alpha, beta, zero_tol=zero_tol) # type: ignore


def jacobian(
    f: Callable[[list | np.ndarray], int | float | np.number | list | np.ndarray],
    x: list | np.ndarray,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> np.ndarray:
    """
    Approximates the specular Jacobian matrix of a vector-valued function :math:`f:\\mathbb{R}^n \\to \\mathbb{R}^m`.

    Returns a matrix of shape (m, n) where J[j, i] is the partial derivative of f_j with respect to x_i (1 <= i <= n, 1 <= j <= m).

    Raises
    ------
    TypeError
        If ``x`` is not of a valid array-like type.
    ValueError
        If ``h`` is not positive.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise TypeError(
            f"Input 'x' must be a vector. "
            f"Got {type(x).__name__} with shape {x.shape}. "
            "Use `specular.derivative` for scalar inputs."
        )

    n = x.size

    f_val = np.asarray(f(x), dtype=float)

    if f_val.ndim == 0:
        f_val = f_val.reshape(1)

    m = f_val.size

    I = np.eye(n)
    
    x_right = x + h * I
    x_left = x - h * I

    # try:
    #     f_right = np.asarray(f(x_right), dtype=float)
    #     f_left = np.asarray(f(x_left), dtype=float)

    #     if f_right.size != n * m or f_left.size != n * m:
    #         raise ValueError("Vectorization mismatch")

    # except Exception:
    #     f_right = np.array([f(row) for row in x_right], dtype=float)
    #     f_left = np.array([f(row) for row in x_left], dtype=float)

    # f_right = np.asarray(f_right, dtype=float).reshape(n, m)
    # f_left = np.asarray(f_left, dtype=float).reshape(n, m)

    f_right = np.array([f(row) for row in x_right], dtype=float)
    f_left = np.array([f(row) for row in x_left], dtype=float)

    f_right = f_right.reshape(n, m)
    f_left = f_left.reshape(n, m)

    alpha = (f_right - f_val) / h
    beta = (f_val - f_left) / h

    J_transposed = _A_vector(alpha, beta, zero_tol=zero_tol)

    return J_transposed.T