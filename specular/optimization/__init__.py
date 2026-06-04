from .solver import (
    gradient_descent,
    specular_gradient,
    adam,
    bfgs,
    specular_bfgs,
    specular_modified_bfgs,
    gradient_method,
    BFGS_method
)

from .result import OptimizationResult
from .step_size.step_schedule import StepSchedule
from .step_size.line_search import LineSearch

__all__ = [
    # API functions
    "gradient_descent", 
    "specular_gradient", 
    "adam", 
    "bfgs", 
    "specular_bfgs", 
    "specular_modified_bfgs", 
    
    # Backwards compatibility
    "gradient_method", 
    "BFGS_method",
    
    # Core components
    "OptimizationResult", 
    "StepSchedule", 
    "LineSearch"
]
