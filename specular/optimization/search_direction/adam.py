import numpy as np
from .base import SearchDirection

class AdamDirection(SearchDirection):
    """
    Adam optimization direction.
    Maintains moving averages of the gradient and its square.
    """
    
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        
        self.m = None
        self.v = None

    def __call__(self, k: int, x: np.ndarray | float, grad: np.ndarray | float) -> np.ndarray | float:
        is_scalar = np.isscalar(grad) or np.asarray(grad).ndim == 0
        
        if is_scalar:
            g = np.array([float(grad)])
        else:
            g = np.asarray(grad, dtype=float).ravel()

        if self.m is None:
            self.m = np.zeros_like(g)
            self.v = np.zeros_like(g)

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1.0 - self.beta1 ** k)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1.0 - self.beta2 ** k)

        # Search direction (note: standard Adam uses this as the actual step, 
        # but here we return it as d_k to be scaled by step schedule t_k)
        d_k = -m_hat / (np.sqrt(v_hat) + self.eps)

        if is_scalar:
            return float(d_k[0])
        return d_k.reshape(np.asarray(grad).shape)
