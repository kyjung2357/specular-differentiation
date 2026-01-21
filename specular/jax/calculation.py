"""
This module provides JAX-based implementations of the function $\\mathcal{A}$, specular directional derivatives, specular partial derivatives, specular derivatives, specular gradients, and specular Jacobians.

It utilizes `jax.numpy` for GPU/TPU acceleration and `jax.vmap` for auto-vectorization.
"""
from functools import partial
from typing import Callable, Union, Tuple
import jax
import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

jax.config.update("jax_enable_x64", True)

@partial(jit, static_argnames=['quasi_Fermat', 'monotonicity'])
def _A_vector(
    f_right: Array,
    f_val: Array,
    f_left: Array,
    h: float,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False, 
    monotonicity: bool = False
) -> Union[Array, Tuple[Array, ...]]:
    """
    JAX version of ``specular.calculation._A_vector``.
    """
    alpha = f_right - f_val
    beta = f_val - f_left

    numerator = alpha * beta - h * h
    denominator = (f_right - f_left) * h
    
    mask = jnp.abs(denominator) > (zero_tol * h)
    
    safe_denominator = jnp.where(mask, denominator, 1.0)
    
    omega = numerator / safe_denominator
    
    result = omega + jnp.sign(denominator) * jnp.hypot(1.0, omega)

    returns = [jnp.where(mask, result, 0.0)]
    
    if quasi_Fermat:
        returns.append(jnp.where(mask, jnp.sign(numerator), 0.0))
        
    if monotonicity:
        returns.append(jnp.where(mask, jnp.sign(denominator), 0.0))
        
    if len(returns) == 1:
        return returns[0]
    
    return tuple(returns)


def derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: float | int | ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> ArrayLike:
    """
    JAX version of ``specular.derivative``.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x = jnp.asarray(x, dtype=float)
    
    if x.ndim != 0:
         raise TypeError(f"Input 'x' must be a scalar. Got shape {x.shape}.")
    
    f_right = jnp.asarray(f(x + h))
    f_val = jnp.asarray(f(x))
    f_left = jnp.asarray(f(x - h))
    
    return _A_vector(f_right, f_val, f_left, h, zero_tol)


def directional_derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    v: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float | Array:
    """
    JAX version of ``specular.directional_derivative``.
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

    f_right = jnp.asarray(f(x + h * v))
    f_left = jnp.asarray(f(x - h * v))
    
    return _A_vector(f_right, f_val, f_left, h, zero_tol)



def partial_derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    i: int,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float | Array:
    """
    JAX version of ``specular.partial_derivative``.
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
    JAX version of ``specular.gradient``.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x = jnp.asarray(x, dtype=float)
    
    if x.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {x.shape}.")
    
    f_val = jnp.asarray(f(x))
    
    n = x.size
    h_identity = h * jnp.eye(n)

    f_vmap = vmap(f)
    
    f_right = f_vmap(x + h_identity)
    f_left = f_vmap(x - h_identity)

    return _A_vector(f_right, f_val, f_left, h, zero_tol)


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

    x = jnp.asarray(x, dtype=float)
    
    if x.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {x.shape}.")

    n = x.size
    
    f_val = jnp.atleast_1d(f(x))
    m = f_val.size
    
    h_idenitiy = h * jnp.eye(n)
    
    f_vmap = vmap(f)

    f_right = jnp.asarray(f_vmap(x + h_idenitiy)).reshape(n, m)
    f_left = jnp.asarray(f_vmap(x - h_idenitiy)).reshape(n, m)

    J_transposed = _A_vector(f_right, f_val, f_left, h, zero_tol)

    return J_transposed.T