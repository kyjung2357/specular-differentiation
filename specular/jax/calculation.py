"""
==============================================
Calculations of specular differentiation (JAX)
==============================================

This module provides JAX-based implementations of the function :math:`\\mathcal{A}`, specular directional derivatives, specular partial derivatives, specular derivatives, specular gradients, and specular Jacobians.

It utilizes `jax.numpy` for GPU/TPU acceleration and `jax.vmap` for auto-vectorization.
"""

from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def A(
    alpha: ArrayLike,
    beta: ArrayLike,
    zero_tol: float = 1e-8
) -> Array:
    """
    JAX version of :func:`specular.calculation.A`.
    """
    alpha = jnp.asarray(alpha, dtype=float)
    beta = jnp.asarray(beta, dtype=float)

    denominator = alpha + beta

    mask = jnp.abs(denominator) > zero_tol
    
    safe_denominator = jnp.where(mask, denominator, 1.0)

    numerator = alpha * beta - 1.0 + jnp.sqrt((1.0 + alpha**2) * (1.0 + beta**2))
    
    result = numerator / safe_denominator

    return jnp.where(mask, result, 0.0)


def derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: float | int | ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> ArrayLike:
    """
    JAX version of :func:`specular.derivative`.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x = jnp.asarray(x, dtype=float)
    
    if x.ndim != 0:
         raise TypeError(f"Input 'x' must be a scalar. Got shape {x.shape}.")
    
    f_val = jnp.asarray(f(x))
    f_right = jnp.asarray(f(x + h))
    f_left = jnp.asarray(f(x - h))
    
    alpha = (f_right - f_val) / h
    beta = (f_val - f_left) / h

    return A(alpha, beta, zero_tol=zero_tol)


def directional_derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    v: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float | Array:
    """
    JAX version of :func:`specular.directional_derivative`.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x = jnp.asarray(x, dtype=float)
    v = jnp.asarray(v, dtype=float)

    if x.ndim == 0 or v.ndim == 0:
        raise TypeError("Input 'x' and 'v' must be vectors.")
    
    if x.shape != v.shape:
        raise ValueError(f"Shape mismatch: x {x.shape} vs v {v.shape}")

    f_val = jnp.asarray(f(x))
    
    if f_val.ndim != 0:
        raise ValueError(f"Function f must return a scalar. Got shape {f_val.shape}")

    alpha = (f(x + h * v) - f_val) / h
    beta = (f_val - f(x - h * v)) / h

    return A(alpha, beta, zero_tol=zero_tol)


def partial_derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    i: int,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float | Array:
    """
    JAX version of :func:`specular.partial_derivative`.
    """
    x = jnp.asarray(x, dtype=float)
    n = x.size
    
    if i < 1 or i > n:
        raise ValueError(f"Index 'i' must be between 1 and {n}.")

    e_i = jnp.zeros_like(x)
    e_i = e_i.at[i - 1].set(1.0) 

    return directional_derivative(f, x, e_i, h, zero_tol)


def gradient(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> Array:
    """
    JAX version of :func:`specular.gradient`.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x = jnp.asarray(x, dtype=float)
    
    if x.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {x.shape}.")

    n = x.size
    I = jnp.eye(n)

    f_val = jnp.asarray(f(x))

    if f_val.ndim != 0:
        raise ValueError(f"Function f must return a scalar. Got shape {f_val.shape}.")

    x_right_batch = x + h * I
    x_left_batch = x - h * I
    
    f_right = jax.vmap(f)(x_right_batch)
    f_left = jax.vmap(f)(x_left_batch)

    alpha = (f_right - f_val) / h
    beta = (f_val - f_left) / h

    return A(alpha, beta, zero_tol=zero_tol)


def jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> Array:
    """
    JAX version of :func:`specular.jacobian`.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x = jnp.asarray(x, dtype=float)
    
    if x.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {x.shape}.")

    n = x.size
    
    f_center = jnp.asarray(f(x), dtype=float)
    
    if f_center.ndim == 0:
        f_center = f_center.reshape(1)
        
    m = f_center.size
    I = jnp.eye(n)
    
    x_right_batch = x + h * I
    x_left_batch = x - h * I

    f_right = jax.vmap(f)(x_right_batch)
    f_left = jax.vmap(f)(x_left_batch)
    
    f_right = jnp.asarray(f_right, dtype=float).reshape(n, m)
    f_left = jnp.asarray(f_left, dtype=float).reshape(n, m)
    
    alpha = (f_right - f_center) / h
    beta = (f_center - f_left) / h
    
    J_transposed = A(alpha, beta, zero_tol=zero_tol)
    
    return J_transposed.T