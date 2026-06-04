import numpy as np
from .base import SearchDirection

class BFGSDirection(SearchDirection):
    """
    Standard BFGS search direction.
    Maintains an approximation of the inverse Hessian matrix H.
    """
    
    def __init__(self):
        self.H = None
        self.prev_x = None
        self.prev_grad = None

    def __call__(self, k: int, x: np.ndarray | float, grad: np.ndarray | float) -> np.ndarray | float:
        is_scalar = np.isscalar(x) or np.asarray(x).ndim == 0
        
        if is_scalar:
            x_vec = np.array([float(x)])
            g_vec = np.array([float(grad)])
        else:
            x_vec = np.asarray(x, dtype=float).ravel()
            g_vec = np.asarray(grad, dtype=float).ravel()

        n = len(x_vec)

        if self.H is None:
            self.H = np.eye(n)
        else:
            # BFGS update
            s = x_vec - self.prev_x
            y = g_vec - self.prev_grad
            
            rho_inv = np.dot(y, s)
            if rho_inv > 1e-10:  # Avoid division by zero and ensure positive definiteness
                rho = 1.0 / rho_inv
                I = np.eye(n)
                
                V = I - rho * np.outer(s, y)
                self.H = V @ self.H @ V.T + rho * np.outer(s, s)

        self.prev_x = x_vec.copy()
        self.prev_grad = g_vec.copy()

        d_k = -self.H @ g_vec

        if is_scalar:
            return float(d_k[0])
        return d_k.reshape(np.asarray(x).shape)


class SpecularModifiedBFGSDirection(BFGSDirection):
    """
    Specular Modified BFGS search direction.
    (To be implemented with custom y_k logic)
    """
    def __call__(self, k: int, x: np.ndarray | float, grad: np.ndarray | float) -> np.ndarray | float:
        # TODO: Implement specular specific modifications to y_k or H update here.
        # Fallback to standard BFGS for now
        return super().__call__(k, x, grad)
