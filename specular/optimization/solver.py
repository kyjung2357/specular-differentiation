import numpy as np
from typing import Any, Callable, Sequence, TypeAlias, cast

from .. import backend
from .step_schedule import StepSchedule
from .line_search import LineSearch
from .result import OptimizationResult

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
    step_size: StepSchedule | LineSearch,
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
    The specular gradient method for minimizing a nonsmooth convex function.

    Parameters:
        f (callable):
            The objective function to minimize.
        x_0 (int | float | list | np.ndarray):
            The starting point for the optimization.
        step_size (StepSchedule | LineSearch):
            Step-length rule. If a StepSchedule is provided, the method uses h_k = step_size(k).
            If a LineSearch is provided, the method chooses h_k by applying the line-search rule along the current descent direction.
        h (float, optional):
            Mesh size used in the finite difference approximation. Must be positive.
        form (str, optional):
            The form of the specular gradient method.
            Supported forms: ``'specular gradient'``, ``'implicit'``, ``'stochastic'``, ``'hybrid'``.
        tol (float, optional):
            Tolerance for iterations.
        zero_tol (float, optional):
            A small threshold used to determine if the denominator ``alpha + beta`` is close to zero for numerical stability.
        max_iter (int, optional):
            Maximum number of iterations.
        f_j (sequence of callable | callable | None, optional):
            The component function of ``f``.
            Used for the stochastic and hybrid forms to compute a random component of the objective function.

            * If a sequence of callables is provided, each callable should accept a single argument (the variable `x`).

            * If a single callable is provided, it should accept two arguments: the variable `x` and an index `j`, and return the `j`-th component function value at `x`.
        m (int, optional):
            The number of component functions.
            Used for the stochastic and hybrid forms.
        switch_iter (int | None, optional):
            The iteration to switch from a method to another for the hybrid form.
            Used for the hybrid form only.
        record_history (bool, optional):
            Whether to record the history of variables and function values.
        print_bar (bool, optional):
            Whether to print the progress bar.

    Returns:
        The result of the optimization containing the solution, function value, number of iterations, runtime, and history.
    
    Raises:
        ValueError:
            If ``h`` is not positive.
        TypeError:
            If an unknown ``form`` is provided.
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
        record_history=record_history,
        print_bar=print_bar,
    )
