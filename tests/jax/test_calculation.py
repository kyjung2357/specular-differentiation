import pytest
import numpy as np

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import specular.jax as jsp

jax.config.update("jax_enable_x64", True)

# ==========================================
# 1. Test Derivative
# ==========================================
def test_jax_derivative_smooth():
    """$f(x) = x^2$, $f'(2) = 4$."""
    f = lambda x: x**2

    res = jsp.derivative(f, 2.0)
    assert res == pytest.approx(4.0, rel=1e-4)

def test_jax_derivative_nonsmooth():
    """$f(x) = |x|$, $f'(0) = 0$."""
    f = lambda x: jnp.abs(x)
    res = jsp.derivative(f, 0.0)
    assert res == pytest.approx(0.0, abs=1e-6)

# ==========================================
# 2. Test JIT Compilation (Crucial for JAX)
# ==========================================
def test_jax_jit_compatibility():
    """
    Ensure specular functions can be JIT compiled.
    """
    f = lambda x: x**2
    x = 2
    
    jit_derivative = jax.jit(jsp.derivative, static_argnums=(0,))
    
    res = jit_derivative(f, x)
    assert float(res) == pytest.approx(4.0, rel=1e-4)

# ==========================================
# 3. Test Gradient (vmap verification)
# ==========================================
def test_jax_gradient_smooth():
    """$f(x_1, x_2, x_3) = x_1^2 + x_2^2 + x_3^2$."""
    f = lambda x: jnp.sum(x**2)
    x = jnp.array([1.0, 2.0, 3.0])
    
    grad = jsp.gradient(f, x)
    
    assert grad.shape == (3,)
    np.testing.assert_allclose(grad, [2.0, 4.0, 6.0], rtol=1e-4)

def test_jax_gradient_nonsmooth():
    """$f(x_1, x_2) = x_1^2 + max(0.0, x_2)$."""
    f = lambda x: x[0]**2 + jnp.maximum(0.0, x[1])

    x = jnp.array([2.0, 0.0])

    res = jsp.gradient(f, x)

    assert res[0] == pytest.approx(4.0, rel=1e-4)
    assert res[1] == pytest.approx(np.sqrt(2) - 1.0, rel=1e-4)
    
def test_jax_gradient_jit():
    """Test gradient with JIT."""
    f = lambda x: jnp.sum(x**2)
    
    jit_grad = jax.jit(jsp.gradient, static_argnums=(0,))
    
    x = jnp.array([1.0, 2.0])
    res = jit_grad(f, x)
    np.testing.assert_allclose(res, [2.0, 4.0], rtol=1e-4)

# ==========================================
# 5. Test Jacobian (Matrix output + vmap)
# ==========================================
def test_jax_jacobian():
    """f(x_1, x_2) = [x_1^2, x_1 + x_2]."""
    def f(x):
        x_1, x_2 = x[0], x[1]
        return jnp.array([x_1**2, x_1 + x_2])
    
    x = jnp.array([2.0, 1.0])
    
    J = jsp.jacobian(f, x)
    
    expected_J = np.array([
        [4.0, 0.0],
        [1.0, 1.0]
    ])
    
    assert J.shape == (2, 2)
    np.testing.assert_allclose(J, expected_J, rtol=1e-4)

def test_jax_jacobian_scalar_output():
    """Check shape consistency when output is effectively scalar."""
    f = lambda x: jnp.sum(x)
    x = jnp.array([1.0, 2.0, 3.0])
    
    J = jsp.jacobian(f, x)
    
    assert J.shape == (1, 3)
    np.testing.assert_allclose(J, [[1.0, 1.0, 1.0]], rtol=1e-4)

# ==========================================
# 6. Test Input Types (ArrayLike)
# ==========================================
def test_jax_input_types():
    """Verify it accepts lists and standard floats."""
    f = lambda x: jnp.sum(x**2)
    x = [1.0, 2.0]
    
    grad = jsp.gradient(f, x=x) # type: ignore
    np.testing.assert_allclose(grad, [2.0, 4.0], rtol=1e-4)