"""Specular differentiation."""

from importlib import metadata as _metadata
from typing import TYPE_CHECKING, Any

from .backends import (
    BackendName,
    available_backends,
    get_backend,
    set_backend,
    use_backend,
)
from .calculation import derivative, gradient, jacobian, scaled_mean


try:
    __version__ = _metadata.version("specular-differentiation")
except _metadata.PackageNotFoundError:
    __version__ = "0+unknown"

del _metadata

if TYPE_CHECKING:
    from .optimization import OptimizationResult, minimize, specular_gradient
    from .ode import (
        ODEResult,
        ellipse_scheme,
        euler_scheme_1,
        euler_scheme_2,
        euler_scheme_5,
    )


_ODE_EXPORTS = frozenset(
    {
        "ODEResult",
        "ellipse_scheme",
        "euler_scheme_1",
        "euler_scheme_2",
        "euler_scheme_5",
    }
)


_OPTIMIZATION_EXPORTS = frozenset({"OptimizationResult", "minimize", "specular_gradient"})


def __getattr__(name: str) -> Any:
    """Load application APIs only when their top-level names are requested."""
    if name in _ODE_EXPORTS or name in _OPTIMIZATION_EXPORTS:
        from importlib import import_module

        module = ".ode" if name in _ODE_EXPORTS else ".optimization"
        value = getattr(import_module(module, __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exported application names in interactive discovery."""
    return sorted(set(globals()) | _ODE_EXPORTS | _OPTIMIZATION_EXPORTS)


__all__ = [
    "__version__",
    "BackendName",
    "available_backends",
    "get_backend",
    "set_backend",
    "use_backend",
    "scaled_mean",
    "derivative",
    "gradient",
    "jacobian",
    "OptimizationResult",
    "minimize",
    "specular_gradient",
    "ODEResult",
    "ellipse_scheme",
    "euler_scheme_1",
    "euler_scheme_2",
    "euler_scheme_5",
]
