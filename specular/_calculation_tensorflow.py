from numbers import Integral
from typing import Callable
import numpy as np
import tensorflow as tf


def _to_f64(x) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        return tf.cast(x, tf.float64)
    return tf.constant(x, dtype=tf.float64)


def _call_f(f: Callable, x_t: tf.Tensor) -> tf.Tensor:
    """Call f with the numpy version of x; return result as float64 tensor."""
    return _to_f64(f(x_t.numpy()))


def _A_core(
    alpha_t: tf.Tensor,
    beta_t: tf.Tensor,
    zero_tol: float,
    quasi_Fermat: bool,
    monotonicity: bool,
):
    denom = alpha_t + beta_t
    mask = tf.abs(denom) > zero_tol
    safe_denom = tf.where(mask, denom, tf.ones_like(denom))
    numer = alpha_t * beta_t - 1.0 + tf.sqrt(
        (1.0 + alpha_t ** 2) * (1.0 + beta_t ** 2)
    )
    A_vals = tf.where(mask, numer / safe_denom, tf.zeros_like(numer))

    returns = [A_vals]
    if quasi_Fermat:
        returns.append(tf.sign(alpha_t * beta_t - 1.0))
    if monotonicity:
        returns.append(tf.sign(denom))

    if len(returns) == 1:
        return A_vals
    return returns


def A(
    alpha,
    beta,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> tf.Tensor | list[tf.Tensor]:
    return _A_core(_to_f64(alpha), _to_f64(beta), zero_tol, quasi_Fermat, monotonicity)


def _A_vector(
    alpha,
    beta,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> tf.Tensor | list[tf.Tensor]:
    return _A_core(_to_f64(alpha), _to_f64(beta), zero_tol, quasi_Fermat, monotonicity)


def derivative(
    f: Callable,
    x,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> tf.Tensor | list[tf.Tensor]:
    x_t = _to_f64(x)
    if x_t.ndim != 0:
        raise TypeError(
            f"Input 'x' must be a scalar. Got shape {x_t.shape}. "
            "Use `specular.directional_derivative`, `specular.gradient`, or `specular.jacobian` for vector inputs."
        )

    x_s = float(x_t.numpy())
    f_val = _to_f64(f(x_s))
    f_right = _to_f64(f(x_s + h))
    f_left = _to_f64(f(x_s - h))

    alpha = (f_right - f_val) / h
    beta = (f_val - f_left) / h

    return A(alpha, beta, zero_tol, quasi_Fermat, monotonicity)


def directional_derivative(
    f: Callable,
    x,
    v,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
) -> tf.Tensor:
    x_t = _to_f64(x)
    v_t = _to_f64(v)

    if x_t.ndim != 1:
        raise TypeError(
            f"Input 'x' must be a vector. Got shape {x_t.shape}. "
            "Use `specular.derivative` for scalar inputs."
        )
    if v_t.ndim != 1:
        raise TypeError(f"Input 'v' must be a vector. Got shape {v_t.shape}.")
    if x_t.shape != v_t.shape:
        raise ValueError(f"Shape mismatch: x {x_t.shape} vs v {v_t.shape}")

    norm = tf.linalg.norm(v_t)
    if float(norm.numpy()) == 0.0:
        return tf.constant(0.0, dtype=tf.float64)

    f_val = _call_f(f, x_t)
    if f_val.ndim != 0:
        raise ValueError(
            "Function 'f' must return a scalar value. "
            f"Got shape {f_val.shape}."
        )

    f_right = _call_f(f, x_t + h * v_t)
    f_left = _call_f(f, x_t - h * v_t)

    alpha = (f_right - f_val) / h
    beta = (f_val - f_left) / h

    return norm * A(alpha / norm, beta / norm, zero_tol)


def partial_derivative(
    f: Callable,
    x,
    i: int,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
) -> tf.Tensor:
    x_t = _to_f64(x)

    if x_t.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {x_t.shape}.")
    if not isinstance(i, Integral):
        raise TypeError(f"Index 'i' must be an integer. Got {type(i).__name__}")

    n = int(x_t.shape[0])
    if i < 1 or i > n:
        raise ValueError(f"Index 'i' must be between 1 and {n} (dimension of x). Got {i}")

    e_i = tf.one_hot(i - 1, n, dtype=tf.float64)
    return directional_derivative(f, x_t, e_i, h, zero_tol)


def gradient(
    f: Callable,
    x,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> tf.Tensor | list[tf.Tensor]:
    x_t = _to_f64(x)

    if x_t.ndim != 1:
        raise TypeError(
            f"Input 'x' must be a vector. Got shape {x_t.shape}. "
            "Use `specular.derivative` for scalar inputs."
        )

    f_val = _call_f(f, x_t)
    if f_val.ndim != 0:
        raise ValueError(
            "Function 'f' must return a scalar value. "
            f"Got shape {f_val.shape}."
        )

    n = int(x_t.shape[0])
    h_identity = h * tf.eye(n, dtype=tf.float64)
    x_right = x_t + h_identity  # (n, n): row i = x + h*e_i
    x_left = x_t - h_identity   # (n, n): row i = x - h*e_i

    f_right = tf.stack([_call_f(f, row) for row in x_right])
    f_left = tf.stack([_call_f(f, row) for row in x_left])
    f_val_arr = tf.fill([n], f_val)

    alpha = (f_right - f_val_arr) / h
    beta = (f_val_arr - f_left) / h

    return _A_vector(alpha, beta, zero_tol, quasi_Fermat, monotonicity)


def jacobian(
    f: Callable,
    x,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> tf.Tensor | list[tf.Tensor]:
    x_t = _to_f64(x)

    if x_t.ndim != 1:
        raise TypeError(
            f"Input 'x' must be a vector. Got shape {x_t.shape}. "
            "Use `specular.derivative` for scalar inputs."
        )

    f_val = tf.reshape(_call_f(f, x_t), [-1])  # shape (m,)
    n = int(x_t.shape[0])
    m = int(f_val.shape[0])

    h_identity = h * tf.eye(n, dtype=tf.float64)
    x_right = x_t + h_identity  # (n, n)
    x_left = x_t - h_identity   # (n, n)

    f_right = tf.reshape(
        tf.stack([tf.reshape(_call_f(f, row), [-1]) for row in x_right]),
        [n, m],
    )
    f_left = tf.reshape(
        tf.stack([tf.reshape(_call_f(f, row), [-1]) for row in x_left]),
        [n, m],
    )
    f_val_arr = tf.tile(tf.expand_dims(f_val, 0), [n, 1])  # (n, m)

    alpha = (f_right - f_val_arr) / h
    beta = (f_val_arr - f_left) / h

    results = _A_vector(alpha, beta, zero_tol, quasi_Fermat, monotonicity)

    if isinstance(results, list):
        return [tf.transpose(r) for r in results]
    return tf.transpose(results)
