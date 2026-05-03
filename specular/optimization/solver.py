import numpy as np
from typing import Any, Callable, Sequence, TypeAlias, cast

from .. import backend
from .line_search import LineSearch
from .result import OptimizationResult
from .step_size import StepSize

ComponentFunc: TypeAlias = Callable[
    [int | float | np.number | list | np.ndarray],
    int | float | np.number,
]


def _get_solver_module():
    if backend._CURRENT_BACKEND in {"cpu_jax", "gpu_jax"}:
        from . import _solver_jax as mod
    else:
        from . import _solver_numpy as mod

    return mod


def gradient_method(
    f: Callable[
        [int | float | np.number | list | np.ndarray],
        int | float | np.number,
    ],
    x_0: Any,
    step_size: StepSize,
    h: float = 1e-6,
    form: str = "specular gradient",
    tol: float = 1e-6,
    zero_tol: float = 1e-8,
    max_iter: int = 1000,
    f_j: Sequence[ComponentFunc] | Callable | None = None,
    m: int = 1,
    switch_iter: int | None = 2,
    record_history: bool = True,
    print_bar: bool = True,
) -> OptimizationResult:
    """
    Minimize a nonsmooth convex function using the current backend.
    """
    impl = cast(Any, _get_solver_module().gradient_method)

    return impl(
        f=f,
        x_0=x_0,
        step_size=step_size,
        h=h,
        form=form,
        tol=tol,
        zero_tol=zero_tol,
        max_iter=max_iter,
        f_j=f_j,
        m=m,
        switch_iter=switch_iter,
        record_history=record_history,
        print_bar=print_bar,
    )


def BFGS_method(
    f: Callable[
        [int | float | np.number | list | np.ndarray],
        int | float | np.number,
    ],
    x_0: Any,
    h: float = 1e-6,
    tol: float = 1e-6,
    zero_tol: float = 1e-8,
    max_iter: int = 1000,
    line_search: str | LineSearch = "armijo",
    alpha_0: float = 1.0,
    c_1: float = 1e-4,
    c_2: float = 0.9,
    rho: float = 0.5,
    max_line_iter: int = 20,
    max_alpha: float = 1e8,
    raise_on_fail: bool = False,
    H_0: np.ndarray | list | None = None,
    safeguard: float = 1e-10,
    record_history: bool = True,
    print_bar: bool = True,
) -> OptimizationResult:
    """
    Minimize a nonsmooth convex function using the specular BFGS method.
    """
    if backend._CURRENT_BACKEND in {"cpu_jax", "gpu_jax"}:
        raise NotImplementedError(
            "BFGS_method is currently implemented for the NumPy backend only."
        )

    module = _get_solver_module()

    if not hasattr(module, "BFGS_method"):
        raise NotImplementedError(
            "BFGS_method is currently implemented for the NumPy optimization backend only."
        )

    impl = cast(Any, getattr(module, "BFGS_method"))

    return impl(
        f=f,
        x_0=x_0,
        h=h,
        tol=tol,
        zero_tol=zero_tol,
        max_iter=max_iter,
        line_search=line_search,
        alpha_0=alpha_0,
        c_1=c_1,
        c_2=c_2,
        rho=rho,
        max_line_iter=max_line_iter,
        max_alpha=max_alpha,
        raise_on_fail=raise_on_fail,
        H_0=H_0,
        safeguard=safeguard,
        record_history=record_history,
        print_bar=print_bar,
    )
