import pytest
import numpy as np

# JAX가 없으면 테스트 건너뛰기
jax = pytest.importorskip("jax")
import jax.numpy as jnp
import specular
import specular.jax as sjax

# 정밀도 문제 방지를 위해 x64 활성화
jax.config.update("jax_enable_x64", True)

# ==========================================
# Test Functions
# ==========================================

def f_quad(x):
    """Simple quadratic function: sum(x^2). Min at 0."""
    return jnp.sum(x**2)

def f_j_quad(x, idx):
    """Component function for stochastic test."""
    # f(x) = x[0]^2 + x[1]^2
    # f_j(x, 0) = x[0]^2, f_j(x, 1) = x[1]^2
    # JAX compatible indexing
    return x[idx]**2

# ==========================================
# 1. Specular Gradient (Standard)
# ==========================================

def test_jax_specular_gradient_convergence():
    """Test standard specular gradient convergence on GPU/JAX."""
    step_size = specular.StepSize('constant', 0.1)
    x_0 = [1.0, 1.0]
    
    res = sjax.gradient_method(
        f=f_quad,
        x_0=x_0,
        step_size=step_size,
        form='specular gradient',
        max_iter=50
    )
    
    assert "JAX specular gradient" in res.method
    np.testing.assert_allclose(res.solution, [0.0, 0.0], atol=0.1)
    assert res.iteration == 50

def test_jax_history_length():
    """Check if history includes x_0 and matches max_iter."""
    step_size = specular.StepSize('constant', 0.1)
    x_0 = [1.0, 1.0]
    max_iter = 10
    
    res = sjax.gradient_method(
        f=f_quad,
        x_0=x_0,
        step_size=step_size,
        max_iter=max_iter,
        record_history=True
    )
    
    # History length should be max_iter + 1 (initial + updates)
    assert len(res.all_history['values']) == max_iter + 1
    # First value should be f(x_0)
    assert res.all_history['values'][0] == pytest.approx(2.0) # 1^2 + 1^2

# ==========================================
# 2. Stochastic Form
# ==========================================

def test_jax_stochastic():
    """Test stochastic form with component function."""
    # Decaying step size for stochastic convergence
    step_size = specular.StepSize('not_summable', 0.5) 
    x_0 = [1.0, 1.0]
    
    res = sjax.gradient_method(
        f=f_quad,
        x_0=x_0,
        step_size=step_size,
        form='stochastic',
        f_j=f_j_quad, # Function accepting (x, idx)
        m=2,
        max_iter=100,
        seed=42
    )
    
    assert "stochastic" in res.method
    # Stochastic is noisy, use loose tolerance
    assert res.func_val < 0.2

# ==========================================
# 3. Hybrid Form
# ==========================================

def test_jax_hybrid_continuity():
    """Test hybrid form and history continuity."""
    step_size = specular.StepSize('constant', 0.05)
    x_0 = [1.0, 1.0]
    max_iter = 20
    switch_iter = 10
    
    res = sjax.gradient_method(
        f=f_quad,
        x_0=x_0,
        step_size=step_size,
        form='hybrid',
        f_j=f_j_quad,
        m=2,
        switch_iter=switch_iter,
        max_iter=max_iter
    )
    
    assert "hybrid" in res.method
    # Check total iterations
    assert res.iteration == max_iter
    
    # Check history shape consistency
    hist_x = res.all_history['variables']
    assert hist_x.shape == (max_iter + 1, 2)
    
    # Check if value decreased
    assert res.func_val < 2.0

# ==========================================
# 4. Step Size Rules (JAX Compatible)
# ==========================================

def test_jax_step_size_rules():
    """Test all step size rules implicitly."""
    x_0 = [0.5]
    f = lambda x: x[0]**2
    
    rules = [
        ('constant', 0.1),
        ('not_summable', 0.1),
        ('square_summable_not_summable', [0.1, 1.0]),
        ('geometric_series', [0.1, 0.9])
    ]
    
    for rule, params in rules:
        step = specular.StepSize(rule, params)
        res = sjax.gradient_method(f, x_0, step, max_iter=5)
        assert res.runtime >= 0

def test_jax_user_defined_step():
    """Test user-defined JAX step function."""
    # Define a JAX-compatible step function
    def custom_step(k):
        return 0.1 / jnp.sqrt(k)
    
    step = specular.StepSize('user_defined', custom_step)
    x_0 = [1.0]
    
    res = sjax.gradient_method(
        f=lambda x: x[0]**2,
        x_0=x_0,
        step_size=step,
        max_iter=10
    )
    
    assert res.func_val < 1.0

# ==========================================
# 5. Error Handling
# ==========================================

def test_jax_error_invalid_backend():
    """Check TypeError for unknown form."""
    step = specular.StepSize('constant', 0.1)
    with pytest.raises(TypeError, match="Unknown form"):
        sjax.gradient_method(f_quad, [1.0], step, form='magic_method')

def test_jax_error_missing_fj():
    """Check ValueError if f_j is missing for stochastic."""
    step = specular.StepSize('constant', 0.1)
    with pytest.raises(ValueError, match="must be provided"):
        sjax.gradient_method(f_quad, [1.0], step, form='stochastic', f_j=None)