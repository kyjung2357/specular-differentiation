"""Backend-neutral public calculation interface."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .backends._registry import _get_selected_backend


def _positive_step(h: Any) -> float:
    """Normalize a concrete, finite, positive real step size."""

    try:
        value = np.asarray(h)
    except (TypeError, ValueError) as exc:
        raise TypeError("h must be a concrete real scalar") from exc

    if value.ndim != 0 or value.dtype.kind not in "iuf":
        raise TypeError("h must be a concrete real scalar")

    step = float(value)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("h must be finite and greater than zero")
    return step


def derivative(f: Any, x: Any, h: Any = None) -> Any:
    """Evaluate a specular derivative with the selected backend.

    If ``h`` is omitted, it is selected from the backend dtype and the scale
    of ``x``. An explicit ``h`` must be a concrete, finite, positive real
    scalar and is validated before ``f`` is evaluated.
    """

    validated_h = None if h is None else _positive_step(h)
    backend = _get_selected_backend()
    return backend.derivative(f, x, validated_h)


def gradient(f: Any, x: Any, h: Any = None) -> Any:
    """Evaluate a specular gradient with the selected backend.

    If ``h`` is omitted, a separate step is selected for each coordinate from
    the backend dtype and coordinate scale. An explicit ``h`` must be a
    concrete, finite, positive real scalar and is validated before ``f`` is
    evaluated.
    """

    validated_h = None if h is None else _positive_step(h)
    backend = _get_selected_backend()
    return backend.gradient(f, x, validated_h)


def jacobian(f: Any, x: Any, h: Any = None) -> Any:
    """Evaluate a specular Jacobian with the selected backend.

    If ``h`` is omitted, a separate step is selected for each coordinate from
    the backend dtype and coordinate scale. An explicit ``h`` must be a
    concrete, finite, positive real scalar and is validated before ``f`` is
    evaluated.
    """

    validated_h = None if h is None else _positive_step(h)
    backend = _get_selected_backend()
    return backend.jacobian(f, x, validated_h)


__all__ = ["derivative", "gradient", "jacobian"]
