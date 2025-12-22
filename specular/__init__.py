from .core import (
    A,
    derivative,
    directional_derivative,
    partial_derivative,
    gradient
)

from . import ode
from . import tools

__version__ = "0.3.2"

__all__ = [
    "A",
    "derivative",
    "directional_derivative", 
    "partial_derivative",
    "gradient", 
    "ode", 
    "tools"
]