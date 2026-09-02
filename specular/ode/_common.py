"""Shared types, validation, and runtime helpers for scalar ODE methods."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import Any

import numpy as np

from ._result import FloatArray


type RealScalar = int | float | np.integer[Any] | np.floating[Any]
type ScalarField = Callable[[float, float], RealScalar]


class _FieldEvaluationCounter:
    """Count evaluations performed through a scalar field."""

    __slots__ = ("_field", "number_of_field_evaluations")

    def __init__(self, field: ScalarField) -> None:
        self._field = field
        self.number_of_field_evaluations = 0

    def __call__(self, t: float, u: float) -> RealScalar:
        self.number_of_field_evaluations += 1
        return self._field(t, u)


def _finite_real(value: object, *, name: str) -> float:
    """Convert a real numeric scalar to a finite Python float."""

    try:
        array = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be a real scalar") from exc

    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real scalar")

    result = float(array)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return a strictly positive integer, excluding booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _field_value(
    F: ScalarField,
    t: float,
    u: float,
    *,
    step: int,
) -> float:
    """Evaluate and validate the scalar field ``F``."""

    return _finite_real(
        F(t, u),
        name=f"F({t!r}, {u!r}) at step {step}",
    )


def _time_grid(
    t_0: float,
    T: float,
    n_steps: int,
) -> tuple[FloatArray, FloatArray]:
    """Construct and validate a representable float64 time grid."""

    weights = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)
    if t_0 < 0.0 < T:
        t_values = (1.0 - weights) * t_0 + weights * T
    else:
        t_values = t_0 + weights * (T - t_0)
    t_values[0] = t_0
    t_values[-1] = T

    step_sizes = np.diff(t_values)
    if np.any(~np.isfinite(step_sizes)) or np.any(step_sizes <= 0.0):
        raise ValueError(
            "the requested uniform time grid is not representable in float64"
        )
    return t_values, step_sizes
