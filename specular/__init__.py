from . import ode
from . import optimization

from .calculation import (
    A,
    derivative,
    directional_derivative,
    partial_derivative,
    gradient
)

from .optimization import (
    StepSize, 
    gradient_method
)

__version__ = "0.5.2"
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
    "optimization",
    "StepSize",
    "gradient_method",
    "__version__"
]