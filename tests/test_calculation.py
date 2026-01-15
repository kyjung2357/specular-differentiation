import pytest
import math
import numpy as np
import specular

# ==========================================
# 1. Test Function A (Core Logic)
# ==========================================
def test_A_scalar():
    """Test scalar inputs for A."""
    expected = 1.0
    assert specular.A(1.0, 1.0) == pytest.approx(expected)
    
    expected = math.sqrt(2) - 1
    assert specular.A(0.0, 1.0) == pytest.approx(expected)

# ==========================================
# 2. Test Derivative (Scalar Input)
# ==========================================
def test_derivative_scalar_output():
    """f: R -> R (Smooth function: x^2)."""
    # f(x) = x**2
    f = lambda x: x**2
    assert specular.derivative(f, x=2.0) == pytest.approx(4.0, rel=1e-4)

def test_derivative_specular_property():
    """f: R -> R (Nonsmooth function: |x|)."""
    # f(x) = |x|
    f = lambda x: abs(x)
    assert specular.derivative(f, x=0.0) == pytest.approx(0.0, abs=1e-6)

def test_derivative_vector_output():
    """f: R -> R^2 (Parametric curve)."""
    # f(x) = (x, x**2)
    f = lambda x: [x, x**2]
    
    result = specular.derivative(f, x=2.0)
    
    np.testing.assert_allclose(result, [1.0, 4.0], rtol=1e-4)

# ==========================================
# 3. Test Directional Derivative
# ==========================================
def test_directional_derivative():
    """f: R^2 -> R."""
    # f(x_1, x_2) = x_1**2 + x_2**2
    f = lambda v: v[0]**2 + v[1]**2
    x = [1.0, 1.0]
    
    assert specular.directional_derivative(f, x, v=[1.0, 0.0]) == pytest.approx(2.0, rel=1e-4)

# ==========================================
# 4. Test Partial Derivative
# ==========================================
def test_partial_derivative():
    """Test partial derivative with 1-based indexing."""
    # f(x_1, x_2, x_3) = x_1 + 2x_2 + 3x_3
    f = lambda x: x[0] + 2*x[1] + 3*x[2]
    x = [0, 0, 0]
    
    assert specular.partial_derivative(f, x, i=2) == pytest.approx(2.0, rel=1e-4)
    assert specular.partial_derivative(f, x, i=3) == pytest.approx(3.0, rel=1e-4)

def test_partial_derivative_error():
    """Check index out of bounds error."""
    # f(x_1, x_2) = x_1
    f = lambda x: x[0]
    x = [1.0, 2.0]

    with pytest.raises(ValueError):
        specular.partial_derivative(f, x, i=3)

# ==========================================
# 5. Test Gradient
# ==========================================
def test_gradient():
    """f: R^3 -> R."""
    # f(x_1, x_2, x_3) = x_1 + x_2 + x_3
    f = lambda x: np.sum(np.square(x))
    x = [1.0, 2.0, 3.0]
    
    grad = specular.gradient(f, x)
    
    np.testing.assert_allclose(grad, [2.0, 4.0, 6.0], rtol=1e-4)

# ==========================================
# 6. Test Jacobian
# ==========================================
def test_jacobian():
    """f: R^2 -> R^2."""
    # f(x_1, x_2) = (x**2, x_1 + x_2)
    f = lambda x: [x[0]**2, x[0] + x[1]]
    x = [2.0, 1.0]
    
    J = specular.jacobian(f, x)
    
    expected_J = np.array([
        [4.0, 0.0],
        [1.0, 1.0]
    ])
    
    assert J.shape == (2, 2)
    np.testing.assert_allclose(J, expected_J, rtol=1e-4)

def test_jacobian_scalar_output():
    """Check if Jacobian works for scalar output (should be 1xN matrix)."""
    # f(x_1, x_2, x_3) = x_1 + x_2 + x_3
    f = lambda x: np.sum(x)
    x = [1.0, 2.0, 3.0]
    
    J = specular.jacobian(f, x)
    assert J.shape == (1, 3)
    np.testing.assert_allclose(J, [[1.0, 1.0, 1.0]], rtol=1e-4)