from .result import ODEResult
from .solver import (
    classical_scheme,
    Euler_scheme,
    trigonometric_scheme
)

__all__ = [
    "classical_scheme",
    "Euler_scheme",
    "trigonometric_scheme",
    "ODEResult"
]