import numpy as np
from typing import Any, Callable, Sequence, TypeAlias, cast

from .. import backend
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
