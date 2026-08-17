"""Scalar specular schemes for ordinary differential equations."""

from ._result import ODEResult
from .solver import (
    ellipse_scheme,
    euler_scheme_1,
    euler_scheme_2,
    euler_scheme_5,
)


__all__ = [
    "ODEResult",
    "euler_scheme_1",
    "euler_scheme_2",
    "euler_scheme_5",
    "ellipse_scheme",
]
