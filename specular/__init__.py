from . import ode
from . import optimization

from .calculation import (
    A,
    derivative,
    directional_derivative,
    partial_derivative,
    gradient
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

__version__ = "0.11.6"
__license__ = "MIT"
__author__ = "Kiyuob Jung"
__email__ = "kyjung@msu.edu"

__all__ = [
    "A",
    "derivative",
    "directional_derivative",
    "partial_derivative",
    "gradient",
    "ode",
    "classical_scheme",
    "Euler_scheme",
    "trigonometric_scheme",
    "optimization",
    "StepSize",
    "gradient_method",
    "__version__"
]