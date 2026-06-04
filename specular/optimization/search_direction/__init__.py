from .base import SearchDirection
from .first_order import GradientDirection, NormalizedGradientDirection
from .adam import AdamDirection
from .bfgs import BFGSDirection, SpecularModifiedBFGSDirection

def get_direction_finder(method: str, options: dict) -> SearchDirection:
    """
    Factory function to instantiate the correct SearchDirection 
    based on the method name.
    """
    if "Adam" in method:
        return AdamDirection(
            beta1=options.get('beta1', 0.9),
            beta2=options.get('beta2', 0.999),
            eps=options.get('eps', 1e-8)
        )
    elif "specular_modified_BFGS" in method:
        return SpecularModifiedBFGSDirection()
    elif "BFGS" in method:
        return BFGSDirection()
    elif "specular_gradient" in method:
        # H-SPEG, S-SPEG, and SPEG use normalized specular gradient
        return NormalizedGradientDirection()
    elif "gradient_descent" in method or "gradient" in method:
        return GradientDirection()
    else:
        raise ValueError(f"No search direction implemented for method '{method}'")
