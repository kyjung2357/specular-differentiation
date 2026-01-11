import pytest
import numpy as np
import specular
from specular.optimization.solver import gradient_method

# ==========================================
# 1. Test Setup: Common Functions
# ==========================================

# Target: f(x) = |x| (Minimum at 0) - Nonsmooth
def f_abs_scalar(x):
    return abs(x)

# Target: f(x) = sum(x^2) (Minimum at 0) - Smooth
def f_quad_vector(x):
    return float(np.sum(np.array(x)**2))

# Component functions for Stochastic test
# f(x) = x1^2 + x2^2
# f1(x) = x1^2, f2(x) = x2^2
def f_comp_1(x):
    return x[0]**2

def f_comp_2(x):
    return x[1]**2

# ==========================================
# 2. Scalar Tests (n=1)
# ==========================================

def test_scalar_specular_gradient():
    """Test standard specular gradient method on 1D function."""
    step_size = specular.StepSize('constant', 0.1)
    
    res = gradient_method(
        f=f_abs_scalar,
        x_0=1.0,
        step_size=step_size,
        form='specular gradient',
        max_iter=20
    )
    
    assert res.method == 'specular gradient'
    assert abs(res.solution) < 0.2

def test_scalar_implicit():
    """Test implicit specular gradient method on 1D function."""
    step_size = specular.StepSize('constant', 0.1)
    
    res = gradient_method(
        f=f_abs_scalar,
        x_0=-1.0,
        step_size=step_size,
        form='implicit',
        max_iter=20
    )
    
    assert res.method == 'implicit specular gradient'
    assert abs(res.solution) < 0.2

# ==========================================
# 3. Vector Tests (n > 1)
# ==========================================

def test_vector_specular_gradient():
    """Test specular gradient on 2D vector."""
    step_size = specular.StepSize('constant', 0.1)
    x_0 = [1.0, 1.0]
    
    res = gradient_method(
        f=f_quad_vector,
        x_0=x_0,
        step_size=step_size,
        form='specular gradient',
        max_iter=50
    )
    
    assert res.method == 'specular gradient'
    np.testing.assert_allclose(res.solution, [0.0, 0.0], atol=0.1)

# ==========================================
# 4. Stochastic & Hybrid Tests
# ==========================================

def test_vector_stochastic():
    """Test stochastic form with component functions list."""
    step_size = specular.StepSize('square_summable_not_summable', [0.5, 1.0])
    x_0 = [1.0, 1.0]
    f_components = [f_comp_1, f_comp_2]
    
    res = gradient_method(
        f=f_quad_vector,
        x_0=x_0,
        step_size=step_size,
        form='stochastic',
        f_j=f_components,
        max_iter=100
    )
    
    assert res.method == 'stochastic specular gradient'
    assert res.func_val < 0.5 

def test_vector_hybrid():
    """Test hybrid form (switch from standard to stochastic)."""
    step_size = specular.StepSize('constant', 0.05)
    x_0 = [1.0, 1.0]
    f_components = [f_comp_1, f_comp_2]
    
    res = gradient_method(
        f=f_quad_vector,
        x_0=x_0,
        step_size=step_size,
        form='hybrid',
        f_j=f_components,
        switch_iter=5,
        max_iter=20
    )
    
    assert res.method == 'hybrid specular gradient'
    assert res.func_val < f_quad_vector(x_0)

# ==========================================
# 5. Error Handling Tests
# ==========================================

def test_invalid_h():
    """Check error when h <= 0."""
    step = specular.StepSize('constant', 0.1)
    with pytest.raises(ValueError, match="Mesh size 'h' must be positive"):
        gradient_method(f_abs_scalar, 1.0, step, h=-1.0)

def test_unknown_form():
    """Check error for invalid form string."""
    step = specular.StepSize('constant', 0.1)
    with pytest.raises(TypeError, match="Unknown form"):
        gradient_method(f_abs_scalar, 1.0, step, form='magic method')

def test_stochastic_missing_fj():
    """Check error when f_j is missing in stochastic mode."""
    step = specular.StepSize('constant', 0.1)
    with pytest.raises(ValueError, match="must be provided"):
        gradient_method(f_quad_vector, [1, 1], step, form='stochastic', f_j=None)

def test_f_j_callable_signature_error():
    """Check error if f_j callable has wrong signature."""
    step = specular.StepSize('constant', 0.1)
    
    bad_fj = lambda x: x 
    
    with pytest.raises(ValueError, match="must accept at least 2 arguments"):
        gradient_method(
            f_quad_vector, [1, 1], step, 
            form='stochastic', 
            f_j=bad_fj, 
            m=2
        )