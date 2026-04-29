from . import backend
from . import ode
from . import optimization

from .backend import(
    backend_info,
    change_backend
)

from .calculation import (
    A,
    derivative,
    directional_derivative,
    partial_derivative,
    gradient,
    jacobian
)

from .ode import (
    classical_scheme,
    Euler_scheme,
    trigonometric_scheme
)

from .optimization import (
    StepSize,
    gradient_method
)

__version__ = "1.0.9"
__license__ = "MIT"
__author__ = "Kiyuob Jung"
__email__ = "kyjung@msu.edu"

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
    "optimization",
    "StepSize",
    "gradient_method",
    "__version__"
]