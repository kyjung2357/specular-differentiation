from numbers import Integral
from typing import Callable
import torch


def _to_f64(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(dtype=torch.float64)
    return torch.as_tensor(x, dtype=torch.float64)


def _call_f(f: Callable, x_t: torch.Tensor) -> torch.Tensor:
    """Call f with numpy array; return result as float64 CPU tensor."""
    return torch.as_tensor(f(x_t.numpy()), dtype=torch.float64)


def _A_core(
    alpha_t: torch.Tensor,
    beta_t: torch.Tensor,
    zero_tol: float,
    quasi_Fermat: bool,
    monotonicity: bool,
):
    denom = alpha_t + beta_t
    mask = torch.abs(denom) > zero_tol
    safe_denom = torch.where(mask, denom, torch.ones_like(denom))
    numer = alpha_t * beta_t - 1.0 + torch.sqrt(
        (1.0 + alpha_t ** 2) * (1.0 + beta_t ** 2)
    )
    A_vals = torch.where(mask, numer / safe_denom, torch.zeros_like(numer))

    returns = [A_vals]
    if quasi_Fermat:
        returns.append(torch.sign(alpha_t * beta_t - 1.0))
    if monotonicity:
        returns.append(torch.sign(denom))

    if len(returns) == 1:
        return A_vals
    return returns


def A(
    alpha,
    beta,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
    return _A_core(_to_f64(alpha), _to_f64(beta), zero_tol, quasi_Fermat, monotonicity)


def _A_vector(
    alpha,
    beta,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
    return _A_core(_to_f64(alpha), _to_f64(beta), zero_tol, quasi_Fermat, monotonicity)


def derivative(
    f: Callable,
    x,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
    x_t = _to_f64(x)
    if x_t.ndim != 0:
        raise TypeError(
            f"Input 'x' must be a scalar. Got shape {tuple(x_t.shape)}. "
            "Use `specular.directional_derivative`, `specular.gradient`, or `specular.jacobian` for vector inputs."
        )

    x_s = float(x_t)
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
) -> torch.Tensor:
    x_t = _to_f64(x)
    v_t = _to_f64(v)

    if x_t.ndim != 1:
        raise TypeError(
            f"Input 'x' must be a vector. Got shape {tuple(x_t.shape)}. "
            "Use `specular.derivative` for scalar inputs."
        )
    if v_t.ndim != 1:
        raise TypeError(f"Input 'v' must be a vector. Got shape {tuple(v_t.shape)}.")
    if x_t.shape != v_t.shape:
        raise ValueError(f"Shape mismatch: x {tuple(x_t.shape)} vs v {tuple(v_t.shape)}")

    norm = torch.linalg.norm(v_t)
    if float(norm) == 0.0:
        return torch.tensor(0.0, dtype=torch.float64)

    f_val = _call_f(f, x_t)
    if f_val.ndim != 0:
        raise ValueError(
            "Function 'f' must return a scalar value. "
            f"Got shape {tuple(f_val.shape)}."
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
) -> torch.Tensor:
    x_t = _to_f64(x)

    if x_t.ndim != 1:
        raise TypeError(f"Input 'x' must be a vector. Got shape {tuple(x_t.shape)}.")
    if not isinstance(i, Integral):
        raise TypeError(f"Index 'i' must be an integer. Got {type(i).__name__}")

    n = x_t.shape[0]
    if i < 1 or i > n:
        raise ValueError(f"Index 'i' must be between 1 and {n} (dimension of x). Got {i}")

    e_i = torch.zeros(n, dtype=torch.float64)
    e_i[i - 1] = 1.0
    return directional_derivative(f, x_t, e_i, h, zero_tol)


def gradient(
    f: Callable,
    x,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
    x_t = _to_f64(x)

    if x_t.ndim != 1:
        raise TypeError(
            f"Input 'x' must be a vector. Got shape {tuple(x_t.shape)}. "
            "Use `specular.derivative` for scalar inputs."
        )

    f_val = _call_f(f, x_t)
    if f_val.ndim != 0:
        raise ValueError(
            "Function 'f' must return a scalar value. "
            f"Got shape {tuple(f_val.shape)}."
        )

    n = x_t.shape[0]
    h_identity = h * torch.eye(n, dtype=torch.float64)
    x_right = x_t + h_identity  # (n, n): row i = x + h*e_i
    x_left = x_t - h_identity   # (n, n): row i = x - h*e_i

    f_right = torch.stack([_call_f(f, row) for row in x_right])
    f_left = torch.stack([_call_f(f, row) for row in x_left])
    f_val_arr = torch.full((n,), f_val.item(), dtype=torch.float64)

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
) -> torch.Tensor | list[torch.Tensor]:
    x_t = _to_f64(x)

    if x_t.ndim != 1:
        raise TypeError(
            f"Input 'x' must be a vector. Got shape {tuple(x_t.shape)}. "
            "Use `specular.derivative` for scalar inputs."
        )

    f_val = _call_f(f, x_t).reshape(-1)  # shape (m,)
    n = x_t.shape[0]
    m = f_val.shape[0]

    h_identity = h * torch.eye(n, dtype=torch.float64)
    x_right = x_t + h_identity  # (n, n)
    x_left = x_t - h_identity   # (n, n)

    f_right = torch.stack([_call_f(f, row).reshape(-1) for row in x_right])  # (n, m)
    f_left = torch.stack([_call_f(f, row).reshape(-1) for row in x_left])    # (n, m)
    f_val_arr = f_val.unsqueeze(0).expand(n, -1)  # (n, m)

    alpha = (f_right - f_val_arr) / h
    beta = (f_val_arr - f_left) / h

    results = _A_vector(alpha, beta, zero_tol, quasi_Fermat, monotonicity)

    if isinstance(results, list):
        return [r.T for r in results]
    return results.T
