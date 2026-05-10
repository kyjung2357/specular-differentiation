import pytest
import sys
import types
import torch
import numpy as np
import specular
from specular.optimization.classical_solver import gradient_descent_method, Adam, BFGS
from specular.optimization.line_search import LineSearch
from specular._typing import Vector

# ==========================================
# 1. Test Setup: Objective Functions
# ==========================================

# Objective function: f(x) = x1^2 + x2^2 (Minimum at [0, 0])
def quadratic_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x**2)

def quadratic_np(x: Vector) -> float:
    x_arr = np.asarray(x, dtype=float)
    return float(np.sum(x_arr**2))

# Initial point
x_0 = [1.0, 1.0]

# ==========================================
# 2. Test Gradient Descent
# ==========================================
def test_gradient_descent_convergence():
    """Test if GD converges using 'square_summable_not_summable' step size."""
    step_size = specular.optimization.StepSchedule(
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
    step_size = specular.optimization.StepSchedule(name='constant', parameters=0.1)
    
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
    step_size = specular.optimization.StepSchedule(name='constant', parameters=0.1)
    
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
    step_size = specular.optimization.StepSchedule(
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

def test_bfgs_native_armijo_fallback():
    """Test native BFGS fallback for line searches unsupported by SciPy BFGS."""
    res = BFGS(
        f_np=quadratic_np,
        x_0=np.array(x_0),
        max_iter=50,
        tol=1e-5,
        line_search="armijo",
        grad_np=lambda x: 2.0 * x,
    )

    assert res.method == "BFGS (armijo)"
    np.testing.assert_allclose(res.solution, [0.0, 0.0], atol=1e-6)

def test_bfgs_strong_wolfe_uses_scipy(monkeypatch):
    """Test that strong Wolfe delegates directly to SciPy BFGS."""
    calls = {}
    H_0 = np.eye(2)

    def fake_minimize(fun, x0, method, jac, callback, options):
        calls["method"] = method
        calls["jac"] = jac
        calls["options"] = options
        callback(np.zeros_like(x0))
        return types.SimpleNamespace(x=np.zeros_like(x0), fun=0.0, nit=1)

    optimize_module = types.SimpleNamespace(minimize=fake_minimize)
    scipy_module = types.SimpleNamespace(optimize=optimize_module)
    monkeypatch.setitem(sys.modules, "scipy", scipy_module)
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize_module)

    res = BFGS(
        f_np=quadratic_np,
        x_0=np.array(x_0),
        max_iter=50,
        tol=1e-5,
        line_search="strong_wolfe",
        grad_np=lambda x: 2.0 * x,
        c_1=1e-3,
        c_2=0.8,
        H_0=H_0,
    )

    assert calls["method"] == "BFGS"
    assert calls["options"]["c1"] == 1e-3
    assert calls["options"]["c2"] == 0.8
    np.testing.assert_allclose(calls["options"]["hess_inv0"], H_0)
    assert res.method == "BFGS"
    np.testing.assert_allclose(res.solution, [0.0, 0.0], atol=1e-6)

def test_bfgs_rejects_scipy_line_search_name():
    """Test that BFGS only accepts line-search names, not backend names."""
    with pytest.raises(ValueError, match="Invalid line search 'scipy'"):
        BFGS(
            f_np=quadratic_np,
            x_0=np.array(x_0),
            line_search="scipy",
        )

def test_line_search_requires_explicit_name():
    """Test that LineSearch does not silently default to Armijo."""
    with pytest.raises(ValueError) as exc_info:
        LineSearch()

    message = str(exc_info.value)
    assert "Line search name is required" in message
    assert "exact" in message
    assert "armijo" in message
    assert "wolfe" in message
    assert "strong_wolfe" in message

# ==========================================
# 5. Test High Dimension
# ==========================================
def test_high_dimension():
    """Check higher dimensional inputs."""
    dim = 10
    x_large = np.ones(dim)
    
    def f_torch_large(x):
        return torch.sum(x**2)
    
    step_size = specular.optimization.StepSchedule(name='constant', parameters=0.1)
    
    res = gradient_descent_method(f_torch_large, x_large, step_size, max_iter=10)
    
    solution = np.asarray(res.solution)
    assert solution.shape == (dim,)
    assert res.func_val < 5.0

# ==========================================
# 6. Test Result Class Methods
# ==========================================
def test_result_methods():
    """Check last_record and history methods."""
    step_size = specular.optimization.StepSchedule(name='constant', parameters=0.1)
    res = gradient_descent_method(quadratic_torch, x_0, step_size, max_iter=5)

    x, f, runtime = res.last_record()
    assert isinstance(f, float)
    assert runtime >= 0
    
    h_vars, h_vals, _ = res.history()
    assert len(h_vals) == 6
