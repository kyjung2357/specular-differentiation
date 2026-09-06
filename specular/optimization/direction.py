"""Search directions and their common selection wrapper."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import calculation


type Point = float | NDArray[np.float64]
type Direction = Callable[..., Point]


def _as_point(value: Any, *, name: str = "x") -> Point:
    """Return a finite scalar or a private nonempty one-dimensional array."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar or a real 1D vector") from exc
    if array.dtype.kind not in "iuf" or (
        isinstance(value, (list, tuple))
        and any(isinstance(item, (bool, np.bool_)) for item in value)
    ):
        raise TypeError(f"{name} must be a real scalar or a real 1D vector")
    if array.ndim > 1 or (array.ndim == 1 and array.size == 0):
        raise ValueError(f"{name} must be a scalar or a nonempty 1D vector")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return float(result) if result.ndim == 0 else result


def _norm(value: Point) -> float:
    """Evaluate the Euclidean norm without intermediate range failures."""
    if np.ndim(value) == 0:
        return abs(float(value))
    return math.hypot(*value)


def _negative_unit(value: Point) -> Point:
    """Normalize a nonzero gradient, preserving genuine zero gradients."""
    if np.ndim(value) == 0:
        return 0.0 if value == 0.0 else -math.copysign(1.0, value)
    scale = float(np.max(np.abs(value)))
    if scale == 0.0:
        return np.zeros_like(value)
    # A negligible component may round to zero when component scales differ.
    with np.errstate(under="ignore"):
        scaled = value / scale
        return -scaled / _norm(scaled)


def make_direction(
    method: str | Callable[[int, Point], Any] = "speg",
    *,
    f: Callable[[Point], Any] | None = None,
    gradient: Callable[[Point], Any] | None = None,
    h: Any = None,
) -> Direction:
    r"""Create ``direction(n, x, *, gradient_value=None)``.

    ``"speg"`` (also ``"specular_gradient"``) returns the negative unit
    specular gradient, or zero when the gradient is exactly zero. Supply a
    raw ``gradient(x)`` callback or an objective ``f`` for numerical specular
    differentiation using the currently selected backend. ``h=None`` keeps
    that backend's automatic differentiation interval.

    A callable ``method(n, x)`` supplies a custom direction without additional
    normalization. Iterations are numbered from one. All returned directions
    must be finite and match the scalar or vector shape of ``x`` exactly.
    Callbacks receive a private copy of vector points.

    ``gradient_value`` optionally reuses a previously evaluated raw gradient
    for SPEG. Custom methods do not use this cache.
    """
    if not isinstance(method, str) and not callable(method):
        raise TypeError("method must be a direction name or a callable")
    if isinstance(method, str) and method not in {"speg", "specular_gradient"}:
        raise ValueError(f"unknown direction method: {method!r}")
    if f is not None and not callable(f):
        raise TypeError("f must be callable")
    if gradient is not None and not callable(gradient):
        raise TypeError("gradient must be callable")
    validated_h = None if h is None else calculation._positive_step(h)
    if isinstance(method, str) and f is None and gradient is None:
        raise ValueError("SPEG requires f or a raw gradient callback")

    def direction(n: int, x: Any, *, gradient_value: Any = None) -> Point:
        if isinstance(n, (bool, np.bool_)):
            raise TypeError("n must be an integer")
        try:
            iteration = operator.index(n)
        except TypeError as exc:
            raise TypeError("n must be an integer") from exc
        if iteration < 1:
            raise ValueError("n must be at least one")
        point = _as_point(x)
        shape = np.shape(point)
        if callable(method):
            value = _as_point(method(iteration, point), name="direction")
        else:
            if gradient_value is not None:
                raw = gradient_value
            elif gradient is not None:
                raw = gradient(point)
            elif shape == ():
                raw = calculation.derivative(f, point, h=validated_h)
            else:
                raw = calculation.gradient(f, point, h=validated_h)
            value = _as_point(raw, name="gradient")
        if np.shape(value) != shape:
            name = "direction" if callable(method) else "gradient"
            raise ValueError(f"{name} must have exactly the same shape as x")
        return value if callable(method) else _negative_unit(value)

    return direction


__all__ = ["make_direction"]
