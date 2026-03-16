"""
This module provides JAX-based implementations of specular directional derivatives, 
specular partial derivatives, specular derivatives, specular gradients, and specular Jacobians.

It evaluates exact Auto-diff gradients (jax.grad, jax.jacobian) at infinitesimally 
shifted points (x + h, x - h) to capture the true analytic right and left derivative limits, 
and aggregates them using the exact formulation of the specular A-function.
"""

from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

jax.config.update("jax_enable_x64", True)


@jit
def A(
    alpha: ArrayLike,
    beta: ArrayLike,
    zero_tol: float = 1e-8
) -> Array:
    """
    JAX implementation of the scalar function A from analytical left/right derivatives.
    Automatically broadcasts over 0D (scalars), 1D (vectors), and 2D (matrices) arrays.
    """
    alpha = jnp.asarray(alpha, dtype=float)
    beta = jnp.asarray(beta, dtype=float)
    
    denominator = alpha + beta
    mask = jnp.abs(denominator) > zero_tol
    
    safe_den = jnp.where(mask, denominator, 1.0)
    numerator = alpha * beta - 1.0 + jnp.sqrt((1.0 + alpha**2) * (1.0 + beta**2))
    
    result = numerator / safe_den
    return jnp.where(mask, result, 0.0)


def derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> Array:
    """
    JAX version of ``specular.derivative``.
    Supports f: R -> R and f: R -> R^m seamlessly.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x_arr = jnp.asarray(x, dtype=float)
    if x_arr.ndim != 0:
         raise TypeError(f"Input 'x' must be a scalar. Got shape {x_arr.shape}.")
    
    grad_f = jax.jacobian(f)
    
    alpha = jnp.asarray(grad_f(x_arr + h), dtype=float)
    beta = jnp.asarray(grad_f(x_arr - h), dtype=float)
    
    return A(alpha, beta, zero_tol)


def directional_derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    v: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> Array:
    """
    JAX version of ``specular.directional_derivative``.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x_arr = jnp.asarray(x, dtype=float)
    v_arr = jnp.asarray(v, dtype=float)

    if x_arr.ndim == 0 or v_arr.ndim == 0:
        raise TypeError("Input 'x' and 'v' must be vectors.")
    if x_arr.shape != v_arr.shape:
        raise ValueError(f"Shape mismatch: x {x_arr.shape} vs v {v_arr.shape}")

    if jnp.ndim(jnp.asarray(f(x_arr))) != 0:
        raise ValueError("Function f must return a scalar for directional derivative.")

    grad_f = jax.grad(f)
    norm_v = jnp.linalg.norm(v_arr)
    
    alpha_raw = jnp.dot(grad_f(x_arr + h * v_arr), v_arr)
    beta_raw = jnp.dot(grad_f(x_arr - h * v_arr), v_arr)
    
    return norm_v * A(alpha_raw / norm_v, beta_raw / norm_v, zero_tol)


def partial_derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    i: int,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> Array:
    """
    JAX version of ``specular.partial_derivative``.
    """
    x_arr = jnp.asarray(x, dtype=float)
    n = x_arr.size
    if i < 1 or i > n:
        raise ValueError(f"Index 'i' must be between 1 and {n}.")

    e_i = jnp.zeros_like(x_arr).at[i - 1].set(1.0)
    return directional_derivative(f, x_arr, e_i, h, zero_tol)


def gradient(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> Array:
    """
    JAX version of ``specular.gradient``.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x_arr = jnp.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {x_arr.shape}.")
    
    n = x_arr.size
    h_ident = h * jnp.eye(n)
    
    grad_vmap = vmap(jax.grad(f))
    
    right_grads = jnp.asarray(grad_vmap(x_arr + h_ident), dtype=float)
    left_grads = jnp.asarray(grad_vmap(x_arr - h_ident), dtype=float)
    
    alpha = jnp.diag(right_grads)
    beta = jnp.diag(left_grads)
    
    return A(alpha, beta, zero_tol)


def jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> Array:
    """
    JAX version of ``specular.jacobian``.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x_arr = jnp.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {x_arr.shape}.")

    def f_1d(val):
        return jnp.atleast_1d(jnp.asarray(f(val), dtype=float))

    n = x_arr.size
    h_ident = h * jnp.eye(n)
    
    jac_vmap = vmap(jax.jacobian(f_1d))
    
    right_jacs = jnp.asarray(jac_vmap(x_arr + h_ident), dtype=float)
    left_jacs = jnp.asarray(jac_vmap(x_arr - h_ident), dtype=float)
    
    alpha = jnp.diagonal(right_jacs, axis1=0, axis2=2)
    beta = jnp.diagonal(left_jacs, axis1=0, axis2=2)
    
    return A(alpha, beta, zero_tol)