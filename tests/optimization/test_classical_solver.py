import pytest
import torch
import numpy as np
import specular
from specular.optimization.classical_solver import gradient_descent_method, Adam, BFGS

# ==========================================
# 1. Test Setup: Objective Functions
# ==========================================

# Objective function: f(x) = x1^2 + x2^2 (Minimum at [0, 0])
def quadratic_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x**2)

def quadratic_np(x: np.ndarray) -> float:
    return float(np.sum(x**2))

# Initial point
x_0 = [1.0, 1.0]

# ==========================================
# 2. Test Gradient Descent
# ==========================================
def test_gradient_descent_convergence():
    """Test if GD converges using 'square_summable_not_summable' step size."""
    # StepSize 설정
    step_size = specular.StepSize(
        name='square_summable_not_summable', 
        parameters=[0.5, 0.0] 
    )
    
    res = gradient_descent_method(
        f_torch=quadratic_torch,
        x_0=x_0,
        step_size=step_size,
        max_iter=100
    )
    
    assert res.method == "gradient descent"
    
    assert res.func_val < 0.1 
    
    hist_vars, hist_vals, _ = res.history()
    assert len(hist_vals) == 101

def test_gradient_descent_constant():
    """Test GD with 'constant' step size."""
    step_size = specular.StepSize(name='constant', parameters=0.1)
    
    res = gradient_descent_method(
        f_torch=quadratic_torch,
        x_0=x_0,
        step_size=step_size,
        max_iter=50
    )
    
    np.testing.assert_allclose(res.solution, [0.0, 0.0], atol=1e-3)

# ==========================================
# 3. Test Adam
# ==========================================
def test_adam_convergence():
    """Test Adam with 'constant' step size."""
    step_size = specular.StepSize(name='constant', parameters=0.1)
    
    res = Adam(
        f_torch=quadratic_torch,
        x_0=x_0,
        step_size=step_size,
        max_iter=50
    )
    
    assert res.method == "Adam"
    np.testing.assert_allclose(res.solution, [0.0, 0.0], atol=1e-2)

def test_adam_geometric_decay():
    """Test Adam with 'geometric_series' step size."""
    step_size = specular.StepSize(
        name='geometric_series', 
        parameters=[0.1, 0.99]
    )
    
    res = Adam(
        f_torch=quadratic_torch,
        x_0=x_0,
        step_size=step_size,
        max_iter=50
    )
    assert res.func_val < 0.1

# ==========================================
# 4. Test BFGS (SciPy Wrapper)
# ==========================================
def test_bfgs_convergence():
    """Test BFGS."""
    res = BFGS(
        f_np=quadratic_np,
        x_0=np.array(x_0),
        max_iter=50,
        tol=1e-5
    )
    
    assert res.method == "BFGS"
    np.testing.assert_allclose(res.solution, [0.0, 0.0], atol=1e-6)

# ==========================================
# 5. Test High Dimension
# ==========================================
def test_high_dimension():
    """Check higher dimensional inputs."""
    dim = 10
    x_large = np.ones(dim)
    
    def f_torch_large(x):
        return torch.sum(x**2)
    
    step_size = specular.StepSize(name='constant', parameters=0.1)
    
    res = gradient_descent_method(f_torch_large, x_large, step_size, max_iter=10)
    
    assert res.solution.shape == (dim,)
    assert res.func_val < 5.0

# ==========================================
# 6. Test Result Class Methods
# ==========================================
def test_result_methods():
    """Check last_record and history methods."""
    step_size = specular.StepSize(name='constant', parameters=0.1)
    res = gradient_descent_method(quadratic_torch, x_0, step_size, max_iter=5)

    x, f, runtime = res.last_record()
    assert isinstance(f, float)
    assert runtime >= 0
    
    h_vars, h_vals, _ = res.history()
    assert len(h_vals) == 6