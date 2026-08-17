"""Backend-neutral public calculation interface."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .backends._registry import _get_selected_backend


def _positive_scalar(value: Any, *, name: str) -> float:
    """Normalize a concrete, finite, positive real scalar."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a concrete real scalar") from exc

    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a concrete real scalar")

    result = float(array)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _positive_step(h: Any) -> float:
    """Normalize a concrete, finite, positive real step size."""

    return _positive_scalar(h, name="h")


def _divide_by_scalar(value: Any, scalar: float) -> Any:
    """Divide a scalar or array-like input without coercing backend arrays."""

    try:
        return value / scalar
    except TypeError:
        return np.asarray(value) / scalar


def scaled_mean(
    alpha: Any,
    beta: Any,
    sigma: Any = 1.0,
) -> Any:
    r"""Evaluate the scaled angular mean elementwise.

    This is
    :math:`\mathcal C_\sigma(\alpha,\beta)
    =\sigma\mathcal C(\alpha/\sigma,\beta/\sigma)` for a concrete,
    finite, positive scalar ``sigma``. The selected backend evaluates
    :math:`\mathcal C` and determines the result type and floating-point
    range.
    """

    scale = _positive_scalar(sigma, name="sigma")
    backend = _get_selected_backend()
    result = backend._C(
        _divide_by_scalar(alpha, scale),
        _divide_by_scalar(beta, scale),
    )
    return scale * result


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


__all__ = ["scaled_mean", "derivative", "gradient", "jacobian"]
