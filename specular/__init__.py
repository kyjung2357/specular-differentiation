from importlib import import_module
from typing import TYPE_CHECKING

from .backend import (
    backend_info,
    change_backend,
)

from .calculation import (
    A,
    derivative,
    directional_derivative,
    partial_derivative,
    gradient,
    jacobian,
)

__version__ = "1.2.1"
__license__ = "MIT"
__author__ = "Kiyuob Jung"
__email__ = "kyjung@msu.edu"

if TYPE_CHECKING:
    from . import ode as ode
    from . import optimization as optimization
    from .ode import (
        classical_scheme,
        Euler_scheme,
        trigonometric_scheme,
        Heun_scheme,
        ellipse_scheme,
    )
    from .optimization import (
        BFGS_method,
        LineSearch,
        StepSize,
        gradient_method,
    )

_LAZY_ATTRS = {
    "ode": ("specular.ode", None),
    "classical_scheme": ("specular.ode", "classical_scheme"),
    "Euler_scheme": ("specular.ode", "Euler_scheme"),
    "trigonometric_scheme": ("specular.ode", "trigonometric_scheme"),
    "Heun_scheme": ("specular.ode", "Heun_scheme"),
    "ellipse_scheme": ("specular.ode", "ellipse_scheme"),
    "optimization": ("specular.optimization", None),
    "BFGS_method": ("specular.optimization", "BFGS_method"),
    "LineSearch": ("specular.optimization", "LineSearch"),
    "StepSize": ("specular.optimization", "StepSize"),
    "gradient_method": ("specular.optimization", "gradient_method"),
}


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "backend_info",
    "change_backend",
    "A",
    "derivative",
    "directional_derivative",
    "partial_derivative",
    "gradient",
    "jacobian",
    "ode",
    "classical_scheme",
    "Euler_scheme",
    "trigonometric_scheme",
    "Heun_scheme",
    "ellipse_scheme",
    "optimization",
    "BFGS_method",
    "LineSearch",
    "StepSize",
    "gradient_method",
    "__version__",
]


def __dir__():
    return sorted(__all__)
