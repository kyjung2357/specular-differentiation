"""Scalar specular ellipse schemes for ordinary differential equations."""

from ._solver import (
    ODEResult,
    ellipse_scheme,
    ellipse_scheme_3rd_order,
    ellipse_scheme_4th_order,
)


__all__ = [
    "ODEResult",
    "ellipse_scheme",
    "ellipse_scheme_3rd_order",
    "ellipse_scheme_4th_order",
]
