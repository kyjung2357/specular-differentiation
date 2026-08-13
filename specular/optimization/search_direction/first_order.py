import numpy as np
from .base import SearchDirection

class GradientDirection(SearchDirection):
    """Standard steepest descent direction: d_k = -g_k"""

    def __call__(self, k: int, x: np.ndarray | float, grad: np.ndarray | float) -> np.ndarray | float:
        if np.isscalar(grad) or np.asarray(grad).ndim == 0:
            return -float(grad)
        return -np.asarray(grad, dtype=float)

class NormalizedGradientDirection(SearchDirection):
    """Normalized steepest descent direction: d_k = -g_k / ||g_k||"""

    def __call__(self, k: int, x: np.ndarray | float, grad: np.ndarray | float) -> np.ndarray | float:
        if np.isscalar(grad) or np.asarray(grad).ndim == 0:
            val = float(grad)
            if val == 0.0:
                return 0.0
            return -1.0 if val > 0 else 1.0

        grad_arr = np.asarray(grad, dtype=float)
        norm = np.linalg.norm(grad_arr)
        if norm == 0.0:
            return np.zeros_like(grad_arr)
        return -grad_arr / norm
