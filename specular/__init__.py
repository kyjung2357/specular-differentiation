from .calculation import (
    derivative,
    directional_derivative,
    partial_derivative,
    gradient
)

from . import ode
from . import optimization

__version__ = "0.3.7"
__license__ = "MIT"
__author__ = "Kiyuob Jung"
__email__ = "kyjung@msu.edu"

__all__ = [
    "derivative",
    "directional_derivative", 
    "partial_derivative",
    "gradient", 
    "ode",
    "optimization",
    "__version__"
]