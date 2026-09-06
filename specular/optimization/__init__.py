"""Composable specular optimization directions, step sizes, and solvers."""

from .direction import make_direction
from .step_size import LineSearchError, make_step_size
from .solver import OptimizationResult, minimize, specular_gradient

__all__ = [
    "make_direction",
    "make_step_size",
    "LineSearchError",
    "OptimizationResult",
    "minimize",
    "specular_gradient",
]
