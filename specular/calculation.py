"""Backend-neutral public calculation interface."""

from __future__ import annotations

from typing import Any

from .backends._registry import _get_backend_module


def _A(a: Any, b: Any, c: Any) -> Any:
    """Evaluate the defining secant kernel with the selected backend."""

    backend = _get_backend_module()
    return backend._A(a, b, c)


def _B(alpha: Any, beta: Any) -> Any:
    """Evaluate the angular slope mean with the selected backend."""

    backend = _get_backend_module()
    return backend._B(alpha, beta)


def _C(alpha: Any, beta: Any) -> Any:
    """Evaluate the algebraic slope mean with the selected backend."""

    backend = _get_backend_module()
    return backend._C(alpha, beta)


def derivative(f: Any, x: Any, h: Any = 1e-6) -> Any:
    """Evaluate a specular derivative with the selected backend."""

    backend = _get_backend_module()
    return backend.derivative(f, x, h)


def gradient(f: Any, x: Any, h: Any = 1e-6) -> Any:
    """Evaluate a specular gradient with the selected backend."""

    backend = _get_backend_module()
    return backend.gradient(f, x, h)


def jacobian(f: Any, x: Any, h: Any = 1e-6) -> Any:
    """Evaluate a specular Jacobian with the selected backend."""

    backend = _get_backend_module()
    return backend.jacobian(f, x, h)


__all__ = ["derivative", "gradient", "jacobian"]
