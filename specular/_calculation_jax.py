from numbers import Integral
from typing import Callable
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def A(
    alpha: ArrayLike,
    beta: ArrayLike,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> Array | list[Array]:
    alpha_arr = jnp.asarray(alpha, dtype=float)
    beta_arr = jnp.asarray(beta, dtype=float)

    denominator = alpha_arr + beta_arr
    mask = jnp.abs(denominator) > zero_tol
    safe_denominator = jnp.where(mask, denominator, 1.0)
    numerator = alpha_arr * beta_arr - 1.0 + jnp.sqrt(
        (1.0 + alpha_arr**2) * (1.0 + beta_arr**2)
    )

    A_vals = jnp.where(mask, numerator / safe_denominator, 0.0)

    returns = [A_vals]

    if quasi_Fermat:
        returns.append(jnp.sign(alpha_arr * beta_arr - 1.0))

    if monotonicity:
        returns.append(jnp.sign(denominator))

    if len(returns) == 1:
        return A_vals

    return returns


def _A_vector(
    alpha: ArrayLike,
    beta: ArrayLike,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> Array | list[Array]:
    alpha_arr = jnp.asarray(alpha, dtype=float)
    beta_arr = jnp.asarray(beta, dtype=float)

    denominator = alpha_arr + beta_arr
    mask = jnp.abs(denominator) > zero_tol
    safe_denominator = jnp.where(mask, denominator, 1.0)
    numerator = alpha_arr * beta_arr - 1.0 + jnp.sqrt(
        (1.0 + alpha_arr**2) * (1.0 + beta_arr**2)
    )
    A_vals = jnp.where(
        mask,
        numerator / safe_denominator,
        0.0,
    )

    returns = [A_vals]

    if quasi_Fermat:
        returns.append(jnp.sign(alpha_arr * beta_arr - 1.0))

    if monotonicity:
        returns.append(jnp.sign(denominator))

    if len(returns) == 1:
        return returns[0]

    return returns


def derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> Array | list[Array]:
    x_arr = jnp.asarray(x, dtype=float)
    if x_arr.ndim != 0:
        raise TypeError(f"Input 'x' must be a scalar. Got shape {x_arr.shape}.")

    f_val = jnp.asarray(f(x_arr), dtype=float)

    if f_val.ndim == 0:
        def f_scalar(t):
            return jnp.asarray(f(t), dtype=float)

        grad_f = jax.grad(f_scalar)
        alpha = grad_f(x_arr + h)
        beta = grad_f(x_arr - h)
        return A(alpha, beta, zero_tol, quasi_Fermat, monotonicity)

    def f_vector(t):
        return jnp.asarray(f(t), dtype=float)

    jac_f = jax.jacrev(f_vector)
    alpha = jac_f(x_arr + h)
    beta = jac_f(x_arr - h)

    return _A_vector(alpha, beta, zero_tol, quasi_Fermat, monotonicity)


def directional_derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    v: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
) -> Array:
    x_arr = jnp.asarray(x, dtype=float)
    v_arr = jnp.asarray(v, dtype=float)

    if x_arr.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {x_arr.shape}.")

    if v_arr.ndim != 1:
        raise TypeError(f"Input 'v' must be a vector. Got shape {v_arr.shape}.")

    if x_arr.shape != v_arr.shape:
        raise ValueError(f"Shape mismatch: x {x_arr.shape} vs v {v_arr.shape}")

    norm = jnp.linalg.norm(v_arr)
    if float(norm) == 0.0:
        return jnp.asarray(0.0, dtype=float)

    f_val = jnp.asarray(f(x_arr), dtype=float)
    if f_val.ndim != 0:
        raise ValueError(
            "Function 'f' must return a scalar value. "
            f"Got shape {f_val.shape}."
        )

    def f_scalar(y):
        return jnp.asarray(f(y), dtype=float)

    grad_f = jax.grad(f_scalar)
    alpha = jnp.dot(grad_f(x_arr + h * v_arr), v_arr)
    beta = jnp.dot(grad_f(x_arr - h * v_arr), v_arr)
    return norm * A(alpha / norm, beta / norm, zero_tol)


def partial_derivative(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    i: int,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
) -> Array:
    x_arr = jnp.asarray(x, dtype=float)

    if x_arr.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {x_arr.shape}.")

    if not isinstance(i, Integral):
        raise TypeError(f"Index 'i' must be an integer. Got {type(i).__name__}")

    n = x_arr.size
    if i < 1 or i > n:
        raise ValueError(f"Index 'i' must be between 1 and {n} (dimension of x). Got {i}")

    e_i = jnp.zeros_like(x_arr).at[int(i) - 1].set(1.0)
    return directional_derivative(f, x_arr, e_i, h, zero_tol)


def gradient(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> Array | list[Array]:
    x_arr = jnp.asarray(x, dtype=float)

    if x_arr.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {x_arr.shape}.")

    f_val = jnp.asarray(f(x_arr), dtype=float)
    if f_val.ndim != 0:
        raise ValueError(
            "Function 'f' must return a scalar value. "
            f"Got shape {f_val.shape}."
        )

    def f_scalar(y):
        return jnp.asarray(f(y), dtype=float)

    n = x_arr.size
    h_identity = h * jnp.eye(n, dtype=x_arr.dtype)
    x_right = x_arr + h_identity
    x_left = x_arr - h_identity

    grad_f = jax.grad(f_scalar)
    grad_right = jax.vmap(grad_f)(x_right)
    grad_left = jax.vmap(grad_f)(x_left)

    alpha = jnp.diagonal(grad_right)
    beta = jnp.diagonal(grad_left)
    return _A_vector(alpha, beta, zero_tol, quasi_Fermat, monotonicity)


def jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> Array | list[Array]:
    x_arr = jnp.asarray(x, dtype=float)

    if x_arr.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {x_arr.shape}.")

    jnp.atleast_1d(jnp.asarray(f(x_arr), dtype=float))
    n = x_arr.size
    h_identity = h * jnp.eye(n, dtype=x_arr.dtype)
    x_right = x_arr + h_identity
    x_left = x_arr - h_identity

    def f_vector(y):
        return jnp.atleast_1d(jnp.asarray(f(y), dtype=float))

    jac_f = jax.jacrev(f_vector)
    jac_right = jax.vmap(jac_f)(x_right)
    jac_left = jax.vmap(jac_f)(x_left)

    alpha = jnp.diagonal(jac_right, axis1=0, axis2=2)
    beta = jnp.diagonal(jac_left, axis1=0, axis2=2)

    results = _A_vector(alpha, beta, zero_tol, quasi_Fermat, monotonicity)

    if isinstance(results, list):
        return results

    return results
