"""Compose a search direction and a step-size rule in one iteration loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from ..calculation import derivative, gradient as specular_gradient_value
from .direction import _as_point, _norm, make_direction
from .step_size import LineSearchError, make_step_size


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Last accepted point and numerical termination information.

    ``iteration`` counts accepted updates. History contains ``variables`` and
    ``values`` including the initial point, and one ``step_sizes`` entry per
    update. All three arrays are empty when history recording is disabled.
    ``success`` means the computed gradient norm met ``tol``; it does not
    certify a global minimizer.
    """

    solution: float | np.ndarray
    func_val: float
    iteration: int
    runtime: float
    history: dict[str, np.ndarray]
    stop_reason: str
    success: bool
    method: str = "speg"

    def last_record(self) -> tuple[float | np.ndarray, float, float]:
        """Return the last accepted point, objective value, and elapsed seconds."""
        return self.solution, self.func_val, self.runtime

    def get_history(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return recorded points, objective values, and elapsed seconds."""
        return self.history["variables"], self.history["values"], self.runtime


def _scalar(value: Any, *, name: str, finite: bool = True) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real scalar")
    result = float(array)
    if finite and not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _advance(x: Any, d: Any, gamma: float) -> np.ndarray:
    """Form each affine update with one rounding, including cancellation."""
    point, direction = np.asarray(x), np.asarray(d)
    values = []
    for value, component in zip(point.flat, direction.flat):
        try:
            values.append(math.fma(gamma, float(component), float(value)))
        except OverflowError:
            values.append(math.copysign(math.inf, float(component)))
    return np.asarray(values).reshape(point.shape)


def minimize(
    objective_function: Callable,
    initial_point: Any,
    *,
    direction: str | Callable = "speg",
    step_size: str | float | Callable = "constant",
    gradient: Callable | None = None,
    line_search_gradient: Callable | None = None,
    h: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    record_history: bool = True,
    step_options: Mapping[str, Any] | None = None,
) -> OptimizationResult:
    """Iterate ``x_next = x + step_size(n, x, d) * d`` with ``d=direction(n,x)``.

    ``direction`` selects SPEG or a callable accepting ``(n, x)``. A callable
    ``step_size`` accepts ``(n, x, d)``; wrap a schedule accepting only ``n``
    with :func:`make_step_size`. Iteration indices start at one.

    ``gradient`` supplies the raw specular gradient for SPEG and termination.
    When omitted, the selected calculation backend approximates it with mesh
    ``h`` (automatic by default). Scalar and nonempty 1D real states are
    supported; the optimizer itself is an eager NumPy/Python loop.

    Ordinary line searches use ordinary centered differences by default.
    The ``specular_`` rules use the same specular gradient as the direction.
    ``line_search_gradient`` overrides either choice. The classical Wolfe
    theory does not automatically apply to a specular substitution.

    A failed search or invalid proposed update leaves the last accepted point
    intact. Invalid arguments and callback shape errors raise exceptions.
    """
    if not callable(objective_function):
        raise TypeError("objective_function must be callable")
    if gradient is not None and not callable(gradient):
        raise TypeError("gradient must be callable")
    if line_search_gradient is not None and not callable(line_search_gradient):
        raise TypeError("line_search_gradient must be callable")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)):
        raise TypeError("max_iter must be a nonnegative integer")
    if max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    tol = _scalar(tol, name="tol")
    if tol < 0:
        raise ValueError("tol must be nonnegative")
    if h is not None:
        h = _scalar(h, name="h")
        if h <= 0:
            raise ValueError("h must be positive")
    if not isinstance(record_history, bool):
        raise TypeError("record_history must be a bool")
    x = _as_point(initial_point, name="initial_point")
    shape = np.shape(x)
    options = dict(step_options) if step_options is not None else {}

    def value_at(point):
        # Callbacks receive their own point so they cannot mutate accepted data.
        value = objective_function(_as_point(point))
        return _scalar(value, name="objective value", finite=False)

    def gradient_at(point):
        point = _as_point(point)
        if gradient is not None:
            value = gradient(point)
        elif np.ndim(point) == 0:
            value = derivative(objective_function, point, h=h)
        else:
            # Pass the original callable to preserve JAX tracing / Numba compilation.
            value = specular_gradient_value(objective_function, point, h=h)
        result = _as_point(value, name="gradient")
        if np.shape(result) != shape:
            raise ValueError("gradient shape must match initial_point")
        return result

    direction_fn = make_direction(direction, gradient=gradient_at, h=h)
    builtin_direction = isinstance(direction, str)
    if callable(step_size):
        if options:
            raise ValueError("step_options belong to a named or numeric step_size")
        if line_search_gradient is not None:
            raise ValueError("configure the gradient on the callable step_size itself")
        step_fn = step_size
        builtin_step = False
    else:
        # Never pass SPEG's raw gradient to an ordinary Wolfe rule implicitly.
        search_gradient = line_search_gradient
        is_specular_search = (
            isinstance(step_size, str) and step_size.strip().lower().startswith("specular_")
        )
        if search_gradient is None and is_specular_search:
            search_gradient = gradient_at
        if any(key in options for key in ("f", "gradient")):
            raise ValueError("use objective_function and line_search_gradient explicitly")
        step_fn = make_step_size(step_size, f=value_at, gradient=search_gradient, **options)
        builtin_step = True

    start = time.perf_counter()
    fx = value_at(x)
    if not math.isfinite(fx):
        raise ValueError("objective value at initial_point must be finite")
    points = [np.asarray(x).copy()] if record_history else []
    values = [fx] if record_history else []
    steps: list[float] = []
    iteration = 0
    success = False
    reason = "max_iter reached"

    # The endpoint check also recognizes convergence on the last allowed update.
    for n in range(1, max_iter + 2):
        if max_iter == 0:
            break
        q = gradient_at(x)
        if _norm(q) <= tol:
            success, reason = True, "gradient norm below tolerance"
            break
        if iteration == max_iter:
            break
        if builtin_direction:
            d = direction_fn(n, x, gradient_value=q)
        else:
            d = direction_fn(n, x)
        if _norm(d) == 0.0:
            reason = "zero direction with gradient above tolerance"
            break
        try:
            if builtin_step:
                gamma = step_fn(n, x, d, f_value=fx)
            else:
                gamma = step_fn(n, _as_point(x), _as_point(d))
            gamma = _scalar(gamma, name="step size")
            if gamma <= 0:
                raise ValueError("step size must be positive")
        except LineSearchError as exc:
            reason = f"line search failed: {exc}"
            break
        trial = _advance(x, d, gamma)
        if not np.all(np.isfinite(trial)):
            reason = "non-finite proposed point"
            break
        if np.array_equal(trial, x):
            reason = "step does not change the point at working precision"
            break
        trial = _as_point(trial)
        trial_value = value_at(trial)
        if not math.isfinite(trial_value):
            reason = "non-finite objective at proposed point"
            break
        x, fx = trial, trial_value
        iteration += 1
        if record_history:
            points.append(np.asarray(x).copy())
            values.append(fx)
            steps.append(gamma)

    history = {
        "variables": np.asarray(points, dtype=float).reshape((-1,) + shape),
        "values": np.asarray(values, dtype=float),
        "step_sizes": np.asarray(steps, dtype=float),
    }
    return OptimizationResult(
        solution=_as_point(x),
        func_val=fx,
        iteration=iteration,
        runtime=time.perf_counter() - start,
        history=history,
        stop_reason=reason,
        success=success,
        method=direction if isinstance(direction, str) else "custom",
    )


def specular_gradient(
    objective_function: Callable,
    initial_point: Any,
    step_size: str | float | Callable = "constant",
    max_iter: int = 1000,
    tol: float = 1e-6,
    *,
    h: float | None = None,
    gradient: Callable | None = None,
    line_search_gradient: Callable | None = None,
    record_history: bool = True,
    step_options: Mapping[str, Any] | None = None,
    **step_parameters: Any,
) -> OptimizationResult:
    """Run SPEG, the normalized negative specular-gradient method.

    Named step parameters can be supplied directly, e.g. ``a=0.1``, or in
    ``step_options``. Duplicate parameters are rejected. All iteration and
    termination behavior is provided by :func:`minimize`.
    """
    options = dict(step_options) if step_options is not None else {}
    overlap = options.keys() & step_parameters.keys()
    if overlap:
        raise TypeError(f"duplicate step parameters: {', '.join(sorted(overlap))}")
    options.update(step_parameters)
    return minimize(
        objective_function,
        initial_point,
        direction="speg",
        step_size=step_size,
        gradient=gradient,
        line_search_gradient=line_search_gradient,
        h=h,
        max_iter=max_iter,
        tol=tol,
        record_history=record_history,
        step_options=options,
    )


__all__ = ["OptimizationResult", "minimize", "specular_gradient"]
