"""
This module provides implementations of specular directional derivatives, specular partial derivatives, specular derivatives, specular gradients, and specular Jacobians.

The calculations are based on the function $\\mathcal{A}:\\mathbb{R}^2 \\to \\mathbb{R}$ defined by 

$$
\\mathcal{A}(\\alpha, \\beta) =
\\begin{cases}
    \\frac{\\alpha \\beta - 1 + \\sqrt{(1 + \\alpha^2)(1 + \\beta^2)}}{\\alpha + \\beta} & \\text{if } \\alpha + \\beta \\neq 0, \\\\
    0 & \\text{otherwise.}
\\end{cases}
$$

The parameters $\\alpha$ and $\\beta$ are intended to represent right and left derivatives.
In the code, computations are based on the finite difference approximation of one-sided (directional) derivatives:

$$
\\alpha \\approx \\frac{f(x + hv) - f(x)}{h}
\\qquad \\text{and} \\qquad
\\beta \\approx \\frac{f(x) - f(x - hv)}{h},
$$

where a function $f : \\mathbb{R}^n \\to \\mathbb{R}$, a real number $h > 0$, and vectors $x, v \\in \\mathbb{R}^n$.
"""

from typing import Callable
import math
import numpy as np


def A(
    alpha: float | np.number | int,
    beta: float | np.number | int,
    zero_tol: float = 1e-8
) -> float:
    """
    Compute the function $\\mathcal{A}$ from one-sided directional derivatives.

    Parameters:
        alpha (float | np.number | int):
            One-sided directional derivative.
        beta (float | np.number | int):
            One-sided directional derivative.
        zero_tol (float, optional):
            A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.

    Returns:
        The function $\\mathcal{A}$.

    Examples:
        >>> import specular
        >>> specular.calculation.A(1.0, 2.0)
        1.3874258867227933
    """
    denominator = alpha + beta

    if abs(denominator) <= zero_tol:
        return 0.0
    
    numerator = alpha * beta - 1.0 + math.sqrt((1.0 + alpha**2) * (1.0 + beta**2))

    return numerator / denominator


def _A_vector(
    f_right: np.ndarray, 
    f_val: np.ndarray, 
    f_left: np.ndarray, 
    h: float = 1e-6, 
    zero_tol: float = 1e-8
) -> np.ndarray:
    """Vector implementation of ``A``."""
    h_sq = h * h
    
    alpha = f_right - f_val
    beta = f_val - f_left

    numerator = (alpha * beta) - h_sq + (np.hypot(alpha, h) * np.hypot(beta, h))

    denominator = f_right - f_left
    denominator *= h

    mask = np.abs(denominator) > zero_tol * h
    result = np.zeros_like(denominator)

    np.divide(numerator, denominator, out=result, where=mask)

    return result


def derivative(
    f: Callable[[int | float | np.number], int | float | np.number | list | np.ndarray],
    x: float | np.number | int,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float | np.ndarray:
    """
    Approximates the specular derivative of a function $f:\\mathbb{R} \\to \\mathbb{R}^m$ at a scalar point $x$.
    
    If ``f`` returns a scalar, the result is a float.

    If ``f`` returns a vector, the result is a vector (component-wise derivative).

    Parameters:
        f (callable):
            A function of a single real variable, returning a scalar or a vector.
        x (float | np.number | int):
            The point at which the derivative is evaluated.
        h (float, optional):
            Mesh size used in the finite difference approximation. Must be positive.
        zero_tol (float, optional):
            A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.

    Returns:
        The approximated specular derivative of ``f`` at ``x``.

    Raises:
        TypeError:
            If the type of ``x`` is not a scalar.
        ValueError:
            If ``h`` is not positive.
    
    Examples:
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
        
        return A(alpha, beta, zero_tol=zero_tol) # type: ignore

    # f is vector-valued
    else:
        f_right = np.asarray(f(x + h), dtype=float)
        f_val = np.asarray(f_val, dtype=float)
        f_left = np.asarray(f(x - h), dtype=float)
        
        return _A_vector(f_right, f_val, f_left, h, zero_tol)


def directional_derivative(
    f: Callable[[list | np.ndarray], int | float | np.number],
    x: list | np.ndarray,
    v: list | np.ndarray,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float:
    """
    Approximates the specular directional derivative of a function $f:\\mathbb{R}^n \\to \\mathbb{R}$ at a point $x$ in the direction $v$.

    Parameters:
        f (callable):
            A function of a real vector variable, returning a scalar.
        x (list | np.ndarray):
            The point at which the derivative is evaluated.
        v (list | np.ndarray):
            The direction in which the derivative is taken.
        h (float, optional):
            Mesh size used in the finite difference approximation. Must be positive.
        zero_tol (float, optional):
            A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.

    Returns:
        The approximated specular directional derivative of ``f`` at ``x`` in the direction ``v`` as a scalar.

    Raises:
        TypeError:
            If ``x`` or ``v`` are not of valid array-like types.
        ValueError:
            If ``x`` and ``v`` have different shape.
            If ``h`` is not positive.

    Examples:
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
    norm = float(np.linalg.norm(v))

    return norm * A(alpha / norm, beta / norm, zero_tol=zero_tol)


def partial_derivative(
    f: Callable[[list | np.ndarray], int |float | np.number],
    x: list | np.ndarray,
    i: int | np.integer,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float:
    """
    Approximates the i-th specular partial derivative of a real-valued function $f:\\mathbb{R}^n \\to \\mathbb{R}$ at point $x$ for $n > 1$.

    This is computed using ``specular.directional_derivative`` with the direction of the $i$-th standard basis vector of $\\mathbb{R}^n$.

    Parameters:
        f (callable):
            A function of a real vector variable, returning a scalar.
        x (list | np.ndarray):
            The point at which the derivative is evaluated.
        i (int | np.integer):
            The index of the specular partial derivative with respect to $x_i$ (``1 <= i <= n``).
        h (float, optional):
            Mesh size used in the finite difference approximation. Must be positive.
        zero_tol (float, optional):
            A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.

    Returns:
        The approximated ``i``-th partial specular derivative of ``f`` at ``x`` as a scalar.

    Raises:
        TypeError:
            If ``i`` is not an integer.
        ValueError:
            If ``i`` is out of the valid range (``1 <= i <= n``).

    Examples:
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
    Approximates the specular gradient of a real-valued function $f:\\mathbb{R}^n \\to \\mathbb{R}$ at point $x$ for $n > 1$.

    The specular gradient is defined as the vector of all partial specular derivatives along the standard basis directions.

    Parameters:
        f (callable):
            A function of a real vector variable, returning a scalar.
        x (list | np.ndarray):
            The point at which the specular gradient is evaluated.
        h (float, optional):
            Mesh size used in the finite difference approximation. Must be positive.
        zero_tol (float, optional):
            A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.

    Returns:
        The approximated specular gradient of ``f`` at ``x`` as a vector.

    Raises:
        TypeError:
            If ``x`` is not of a valid array-like type.
        ValueError:
            If ``f`` does not return a scalar value.
            If ``h`` is not positive.

    Examples:
        >>> import specular
        >>> import numpy as np
        >>> f = lambda x: np.linalg.norm(x)
        >>> specular.gradient(f, x=[1.4, -3.47, 4.57, 9.9])
        array([ 0.12144298, -0.3010051 ,  0.39642458,  0.85877534])
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")
    
    x = np.asarray(x, dtype=float).copy()
    
    if x.ndim != 1:
        raise TypeError(
            f"Input 'x' must be a vector. "
            f"Got {type(x).__name__} with shape {x.shape}. "
            "Use `specular.derivative` for scalar inputs."
        )
    
    n = x.size

    f_val_scalar = f(x)

    if np.ndim(f_val_scalar) != 0:
        raise ValueError(
            "Function 'f' must return a scalar value. "
            f"Got shape {np.shape(f_val_scalar)}."
        )

    f_right = np.empty(n, dtype=float)
    f_left = np.empty(n, dtype=float)

    for i in range(n):
        origin_val = x[i]
        
        x[i] = origin_val + h
        f_right[i] = f(x)
        
        x[i] = origin_val - h
        f_left[i] = f(x)
        
        x[i] = origin_val
        
    f_val_arr = np.full_like(f_right, f_val_scalar)

    return _A_vector(f_right, f_val_arr, f_left, h, zero_tol)


def jacobian(
    f: Callable[[list | np.ndarray], int | float | np.number | list | np.ndarray],
    x: list | np.ndarray,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> np.ndarray:
    """
    Approximates the specular Jacobian matrix of a vector-valued function $f:\\mathbb{R}^n \\to \\mathbb{R}^m$.

    Returns a matrix of shape ($m$, $n$) where $J[j, i]$ is the partial derivative of a component function $f_j$ with respect to $x_i$ ($1 \\leq i \\leq n$, $1 \\leq j \\leq m$.

    Parameters:
        f (callable):
            A function of a real vector variable, returning a vector.
        x (list | np.ndarray):
            The point at which the specular gradient is evaluated.
        h (float, optional):
            Mesh size used in the finite difference approximation. Must be positive.
        zero_tol (float, optional):
            A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.

    Returns:
        The approximated specular Jacobian of ``f`` at ``x`` as a matrix.

    Raises:
        TypeError:
            If ``x`` is not of a valid array-like type.
        ValueError:
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

    identity = np.eye(n)
    
    x_right = x + h * identity
    x_left = x - h * identity

    f_right = np.array([f(row) for row in x_right], dtype=float)
    f_left = np.array([f(row) for row in x_left], dtype=float)

    f_right = f_right.reshape(n, m)
    f_left = f_left.reshape(n, m)

    f_val = np.tile(f_val, (n, 1))

    J_transposed = _A_vector(f_right, f_val, f_left, h, zero_tol)

    return J_transposed.T