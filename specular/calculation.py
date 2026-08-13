import numpy as np
import numpy.typing as npt
from types import ModuleType
from typing import Callable, List

from . import backend

ArrayLike = npt.ArrayLike

_loaded_backends: dict[str, ModuleType] = {}

def _get_backend_module() -> ModuleType:
    """Return the implementation module for the current backend."""
    current = backend._CURRENT_BACKEND

    if current not in _loaded_backends:
        if current == "cpu_numpy":
            from . import _calculation_numpy as mod
        elif current == "cpu_numba":
            from . import _calculation_numba as mod
        elif current in {"cpu_jax", "gpu_jax"}:
            from . import _calculation_jax as mod
        elif current in {"cpu_pytorch", "gpu_pytorch"}:
            from . import _calculation_pytorch as mod
        else:
            raise ValueError(f"Unknown backend: {current}")

        _loaded_backends[current] = mod

    return _loaded_backends[current]

def A(
    alpha: "float | np.number | int | np.ndarray",
    beta: "float | np.number | int | np.ndarray",
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False,
) -> "float | np.ndarray | list[float] | list[np.ndarray]":
    """Compute the specular function A(alpha, beta).

    Examples:
        >>> import specular
        >>> specular.A(1.0, 2.0)
        1.3874258867227933
    """
    return _get_backend_module().A(
        alpha, beta, zero_tol, quasi_Fermat, monotonicity
    )

def derivative(
    f: "Callable[[int | float | np.number], int | float | np.number | list | np.ndarray]",
    x: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False
) -> "float | np.ndarray | list[float] | list[np.ndarray]":
    """Approximate the specular derivative of f at scalar x.

    Examples:
        >>> import specular
        >>> f = lambda x: abs(x)
        >>> specular.derivative(f, x=0.0)
        0.0
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    return _get_backend_module().derivative(
        f, x, h, zero_tol, quasi_Fermat, monotonicity
    )


def directional_derivative(
    f: "Callable[[list | np.ndarray], int | float | np.number]",
    x: ArrayLike,
    v: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float:
    """Approximate the specular directional derivative of f at x in direction v."""
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    return _get_backend_module().directional_derivative(f, x, v, h, zero_tol)


def partial_derivative(
    f: "Callable[[list | np.ndarray], int | float | np.number]",
    x: ArrayLike,
    i: "int | np.integer",
    h: float = 1e-6,
    zero_tol: float = 1e-8
) -> float:
    """Approximate the i-th specular partial derivative of f at x (1-indexed)."""
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    return _get_backend_module().partial_derivative(f, x, i, h, zero_tol)


def gradient(
    f: "Callable[[list | np.ndarray], int | float | np.number]",
    x: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False
) -> "np.ndarray | List[np.ndarray]":
    """Approximate the specular gradient of f at x.

    Examples:
        >>> import specular
        >>> import numpy as np
        >>> f = lambda x: np.linalg.norm(x)
        >>> specular.gradient(f, x=[1.4, -3.47, 4.57, 9.9])
        array([ 0.12144298, -0.3010051 ,  0.39642458,  0.85877534])
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    return _get_backend_module().gradient(
        f, x, h, zero_tol, quasi_Fermat, monotonicity
    )

def jacobian(
    f: "Callable[[list | np.ndarray], int | float | np.number | list | np.ndarray]",
    x: ArrayLike,
    h: float = 1e-6,
    zero_tol: float = 1e-8,
    quasi_Fermat: bool = False,
    monotonicity: bool = False
) -> "np.ndarray | List[np.ndarray]":
    """Approximate the specular Jacobian of f at x, shape (m, n)."""
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    return _get_backend_module().jacobian(
        f, x, h, zero_tol, quasi_Fermat, monotonicity
    )