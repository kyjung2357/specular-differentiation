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


def __getattr__(name: str) -> Any:
    """Load the ODE API only when a top-level ODE name is requested."""
    if name in _ODE_EXPORTS:
        from importlib import import_module

        value = getattr(import_module(".ode", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exported ODE names in interactive discovery."""
    return sorted(set(globals()) | _ODE_EXPORTS)


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
    "ODEResult",
    "ellipse_scheme",
    "euler_scheme_1",
    "euler_scheme_2",
    "euler_scheme_5",
]
