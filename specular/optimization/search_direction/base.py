import numpy as np

class SearchDirection:
    """Base class for search direction algorithms."""
    
    def __call__(self, k: int, x: np.ndarray | float, grad: np.ndarray | float) -> np.ndarray | float:
        """
        Compute the search direction at iteration k.
        
        Parameters:
            k: Current iteration number (1-indexed).
            x: Current position.
            grad: Gradient at the current position.
            
        Returns:
            The search direction d_k.
        """
        raise NotImplementedError("Subclasses must implement __call__")
