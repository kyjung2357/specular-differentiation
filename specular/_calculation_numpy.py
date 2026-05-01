from typing import Callable, cast
import math
import numpy as np


def _A_scalar(
    alpha: float | np.number | int,
    beta: float | np.number | int,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> float | list[float]:
    denominator = alpha + beta

    if abs(denominator) <= zero_tol:
        A_val = 0.0
    else:
        numerator = alpha * beta - 1.0 + math.sqrt((1.0 + alpha**2) * (1.0 + beta**2))
        A_val = float(numerator / denominator)

    returns = [A_val]

    if quasi_Fermat:
        returns.append(float(np.sign(alpha * beta - 1.0)))

    if monotonicity:
        returns.append(float(np.sign(denominator)))

    if len(returns) == 1:
        return A_val

    return returns


def _A_vector(
    alpha: np.ndarray,
    beta: np.ndarray,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> np.ndarray | list[np.ndarray]:
    alpha_arr, beta_arr = np.broadcast_arrays(
        np.asarray(alpha, dtype=float),
        np.asarray(beta, dtype=float),
    )
    denominator = alpha_arr + beta_arr
    mask = np.abs(denominator) > zero_tol
    numerator = alpha_arr * beta_arr - 1.0 + np.sqrt(
        (1.0 + alpha_arr**2) * (1.0 + beta_arr**2)
    )

    A_vals = np.zeros_like(denominator, dtype=float)
    np.divide(numerator, denominator, out=A_vals, where=mask)

    returns = [A_vals]

    if quasi_Fermat:
        returns.append(np.sign(alpha_arr * beta_arr - 1.0))

    if monotonicity:
        returns.append(np.sign(denominator))

    if len(returns) == 1:
        return returns[0]

    return returns


def A(
    alpha: float | np.number | int | np.ndarray,
    beta: float | np.number | int | np.ndarray,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> float | np.ndarray | list[float] | list[np.ndarray]:
    if np.ndim(alpha) == 0 and np.ndim(beta) == 0:
        return _A_scalar(
            float(alpha), float(beta), zero_tol, quasi_Fermat, monotonicity
        )
    
    alpha_arr = np.asarray(alpha, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)

    return _A_vector(
        alpha_arr,
        beta_arr,
        zero_tol=zero_tol,
        quasi_Fermat=quasi_Fermat,
        monotonicity=monotonicity,
    )


def derivative(
    f: Callable[[int | float | np.number], int | float | np.number | list | np.ndarray],
    x: float | np.number | int,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> float | np.ndarray | list[float] | list[np.ndarray]:
    try:
        x = float(x)
    except TypeError:
        raise TypeError(
            f"Input 'x' must be a scalar. "
            f"Got {type(x).__name__}. "
            "Use `specular.directional_derivative`, `specular.gradient`, or `specular.jacobian` for vector inputs."
        )

    f_val = f(x)

    if np.ndim(f_val) == 0:
        f_scalar = cast(Callable[[float], float | np.number], f)

        f_right = float(f_scalar(x + h))
        f_val_scalar = float(cast(float | np.number, f_val))
        f_left = float(f_scalar(x - h))

        alpha = (f_right - f_val_scalar) / h
        beta = (f_val_scalar - f_left) / h

        return _A_scalar(alpha, beta, zero_tol, quasi_Fermat, monotonicity)

    f_right = np.asarray(f(x + h), dtype=float)
    f_val = np.asarray(f_val, dtype=float)
    f_left = np.asarray(f(x - h), dtype=float)

    alpha = (f_right - f_val) / h
    beta = (f_val - f_left) / h

    return _A_vector(alpha, beta, zero_tol, quasi_Fermat, monotonicity)


def directional_derivative(
    f: Callable[[list | np.ndarray], int | float | np.number],
    x: list | np.ndarray,
    v: list | np.ndarray,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
) -> float:
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    if x.ndim != 1:
        raise TypeError(
            f"Input 'x' must be a vector. "
            f"Got {type(x).__name__} with shape {x.shape}. "
            "Use `specular.derivative` for scalar inputs."
        )

    if v.ndim != 1:
        raise TypeError(
            f"Input 'v' must be a vector. "
            f"Got {type(v).__name__} with shape {v.shape}."
        )

    if x.shape != v.shape:
        raise ValueError(f"Shape mismatch: x {x.shape} vs v {v.shape}")

    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return 0.0

    f_val = f(x)

    if np.ndim(f_val) != 0:
        raise ValueError(
            "Function 'f' must return a scalar value. "
            f"Got shape {np.shape(f_val)}."
        )

    f_val_scalar = float(cast(float | np.number, f_val))
    f_right = float(f(x + h * v))
    f_left = float(f(x - h * v))

    alpha = (f_right - f_val_scalar) / h
    beta = (f_val_scalar - f_left) / h

    return norm * cast(float, _A_scalar(alpha / norm, beta / norm, zero_tol))


def partial_derivative(
    f: Callable[[list | np.ndarray], int | float | np.number],
    x: list | np.ndarray,
    i: int | np.integer,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
) -> float:
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise TypeError(
            f"Input 'x' must be a vector. "
            f"Got {type(x).__name__} with shape {x.shape}."
        )

    if not isinstance(i, (int, np.integer)):
        raise TypeError(f"Index 'i' must be an integer. Got {type(i).__name__}")

    n = x.size
    if i < 1 or i > n:
        raise ValueError(f"Index 'i' must be between 1 and {n} (dimension of x). Got {i}")

    e_i = np.zeros_like(x)
    e_i[i - 1] = 1.0

    return directional_derivative(f, x, e_i, h, zero_tol)


def gradient(
    f: Callable[[list | np.ndarray], int | float | np.number],
    x: list | np.ndarray,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> np.ndarray | list[np.ndarray]:
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
        x_i = x[i]
        x[i] = x_i + h
        f_right[i] = f(x)
        x[i] = x_i - h
        f_left[i] = f(x)
        x[i] = x_i

    f_val_arr = np.full_like(f_right, float(cast(float | np.number, f_val_scalar)))

    alpha = (f_right - f_val_arr) / h
    beta = (f_val_arr - f_left) / h

    return _A_vector(alpha, beta, zero_tol, quasi_Fermat, monotonicity)


def jacobian(
    f: Callable[[list | np.ndarray], int | float | np.number | list | np.ndarray],
    x: list | np.ndarray,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> np.ndarray | list[np.ndarray]:
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
    h_identity = h * np.eye(n)
    x_right = x + h_identity
    x_left = x - h_identity

    f_right = np.array([f(row) for row in x_right], dtype=float).reshape(n, m)
    f_left = np.array([f(row) for row in x_left], dtype=float).reshape(n, m)
    f_val = np.tile(f_val, (n, 1))

    alpha = (f_right - f_val) / h
    beta = (f_val - f_left) / h

    results = _A_vector(alpha, beta, zero_tol, quasi_Fermat, monotonicity)

    if isinstance(results, list):
        return [result.T for result in results]

    return results.T
