import pytest
import numpy as np

jax = pytest.importorskip("jax")
import jax.numpy as sdj
import specular.jax as sjax

jax.config.update("jax_enable_x64", True)

# ==========================================
# 1. Test A (Core Logic with JAX)
# ==========================================
def test_jax_A_scalar():
    """Test A with JAX scalar inputs."""
    res = sjax.A(1.0, 1.0)
    assert res == pytest.approx(1.0)
    
    assert isinstance(res, (jax.Array, float))

def test_jax_A_vector():
    """Test A with JAX vector inputs (broadcasting)."""
    alpha = sdj.array([1.0, 0.0])
    beta = sdj.array([1.0, 1.0])
    
    result = sjax.A(alpha, beta)
    
    expected_0 = 1.0
    expected_1 = np.sqrt(2) - 1
    
    np.testing.assert_allclose(result, [expected_0, expected_1], rtol=1e-5)

# ==========================================
# 2. Test Derivative
# ==========================================
def test_jax_derivative_scalar():
    """f(x) = x^2, f'(2) = 4."""
    f = lambda x: x**2

    res = sjax.derivative(f, 2.0)
    assert res == pytest.approx(4.0, rel=1e-4)

def test_jax_derivative_nonsmooth():
    """f(x) = |x|, f'(0) = 0 (Specular property)."""
    f = lambda x: sdj.abs(x)
    res = sjax.derivative(f, 0.0)
    assert res == pytest.approx(0.0, abs=1e-6)

# ==========================================
# 3. Test JIT Compilation (Crucial for JAX)
# ==========================================
def test_jax_jit_compatibility():
    """
    Ensure specular functions can be JIT compiled.
    """
    f = lambda x: x**2
    x = 2
    
    jit_derivative = jax.jit(sjax.derivative, static_argnums=(0,))
    
    res = jit_derivative(f, x)
    assert float(res) == pytest.approx(4.0, rel=1e-4)

# ==========================================
# 4. Test Gradient (vmap verification)
# ==========================================
def test_jax_gradient():
    """f(x) = sum(x^2). Gradient is 2x."""
    f = lambda x: sdj.sum(x**2)
    x = sdj.array([1.0, 2.0, 3.0])
    
    grad = sjax.gradient(f, x)
    
    assert grad.shape == (3,)
    np.testing.assert_allclose(grad, [2.0, 4.0, 6.0], rtol=1e-4)

def test_jax_gradient_jit():
    """Test gradient with JIT."""
    f = lambda x: sdj.sum(x**2)
    
    jit_grad = jax.jit(sjax.gradient, static_argnums=(0,))
    
    x = sdj.array([1.0, 2.0])
    res = jit_grad(f, x)
    np.testing.assert_allclose(res, [2.0, 4.0], rtol=1e-4)

# ==========================================
# 5. Test Jacobian (Matrix output + vmap)
# ==========================================
def test_jax_jacobian():
    """f(x, y) = [x^2, x + y]."""
    def f(v):
        x, y = v[0], v[1]
        return sdj.array([x**2, x + y])
    
    x = sdj.array([2.0, 1.0])
    
    J = sjax.jacobian(f, x)
    
    expected_J = np.array([
        [4.0, 0.0],
        [1.0, 1.0]
    ])
    
    assert J.shape == (2, 2)
    np.testing.assert_allclose(J, expected_J, rtol=1e-4)

def test_jax_jacobian_scalar_output():
    """Check shape consistency when output is effectively scalar."""
    f = lambda x: sdj.sum(x)
    x = sdj.array([1.0, 2.0, 3.0])
    
    J = sjax.jacobian(f, x)
    
    assert J.shape == (1, 3)
    np.testing.assert_allclose(J, [[1.0, 1.0, 1.0]], rtol=1e-4)

# ==========================================
# 6. Test Input Types (ArrayLike)
# ==========================================
def test_jax_input_types():
    """Verify it accepts lists and standard floats."""
    f = lambda x: sdj.sum(x**2)
    x = [1.0, 2.0]
    
    grad = sjax.gradient(f, x=x) # type: ignore
    np.testing.assert_allclose(grad, [2.0, 4.0], rtol=1e-4)