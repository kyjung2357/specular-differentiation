import pytest
import math
import importlib.util
import os
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

def test_derivative_scalar_diagnostics():
    f = lambda x: x**2
    result = specular.derivative(f, x=1.0, quasi_Fermat=True, monotonicity=True)
    assert isinstance(result, list)
    assert len(result) == 3

# ==========================================
# 3. Test Directional Derivative
# ==========================================
def test_directional_derivative():
    """f: R^2 -> R."""
    # f(x_1, x_2) = x_1**2 + x_2**2
    f = lambda v: v[0]**2 + v[1]**2
    x = [1.0, 1.0]
    
    assert specular.directional_derivative(f, x, v=[1.0, 0.0]) == pytest.approx(2.0, rel=1e-4)

def test_directional_derivative_zero_direction():
    f = lambda x: x[0] ** 2
    assert specular.directional_derivative(f, x=[1.0], v=[0.0]) == 0.0

def test_directional_derivative_rejects_matrix_input():
    f = lambda x: x[0] ** 2
    with pytest.raises(TypeError):
        specular.directional_derivative(f, x=[[1.0]], v=[[1.0]])

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

def test_partial_derivative_rejects_scalar_input():
    f = lambda x: x
    with pytest.raises(TypeError):
        specular.partial_derivative(f, x=1.0, i=1)

# ==========================================
# 5. Test Gradient
# ==========================================
def test_gradient():
    """f: R^3 -> R."""
    # f(x_1, x_2, x_3) = x_1^2 + x_2^2 + x_3^2
    f = lambda x: np.sum(np.square(x))
    x = [1.0, 2.0, 3.0]
    
    grad = specular.gradient(f, x)
    
    np.testing.assert_allclose(grad, [2.0, 4.0, 6.0], rtol=1e-4)

def test_gradient_small_linear_not_zero():
    f = lambda x: 0.001 * x[0]
    np.testing.assert_allclose(specular.gradient(f, [0.0]), [0.001], rtol=1e-8, atol=1e-12)

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

# ==========================================
# 7. Test A edge cases
# ==========================================
def test_A_zero_denominator():
    assert specular.A(0.0, 0.0) == 0.0

def test_A_near_zero_denominator():
    assert specular.A(1.0, -1.0) == 0.0

# ==========================================
# 8. Test Derivative error cases
# ==========================================
def test_derivative_invalid_h():
    f = lambda x: x
    with pytest.raises(ValueError):
        specular.derivative(f, x=1.0, h=-1.0)
    with pytest.raises(ValueError):
        specular.derivative(f, x=1.0, h=0.0)

def test_derivative_vector_input():
    f = lambda x: x[0]
    with pytest.raises(TypeError):
        specular.derivative(f, x=[1.0, 2.0])

# ==========================================
# 9. Test Directional Derivative error cases
# ==========================================
def test_directional_derivative_shape_mismatch():
    f = lambda v: v[0]**2 + v[1]**2
    with pytest.raises(ValueError):
        specular.directional_derivative(f, x=[1.0, 2.0], v=[1.0, 0.0, 0.0])

def test_directional_derivative_invalid_h():
    f = lambda v: v[0]**2
    with pytest.raises(ValueError):
        specular.directional_derivative(f, x=[1.0], v=[1.0], h=0.0)

# ==========================================
# 10. Test Gradient quasi_Fermat and monotonicity
# ==========================================
def test_gradient_quasi_fermat():
    f = lambda x: np.sum(np.square(x))
    x = [1.0, 2.0, 3.0]
    result = specular.gradient(f, x, quasi_Fermat=True)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (3,)
    assert result[1].shape == (3,)

def test_gradient_monotonicity():
    f = lambda x: np.sum(np.square(x))
    x = [1.0, 2.0, 3.0]
    result = specular.gradient(f, x, monotonicity=True)
    assert isinstance(result, list)
    assert len(result) == 2

def test_gradient_quasi_fermat_and_monotonicity():
    f = lambda x: np.sum(np.square(x))
    x = [1.0, 2.0, 3.0]
    result = specular.gradient(f, x, quasi_Fermat=True, monotonicity=True)
    assert isinstance(result, list)
    assert len(result) == 3

# ==========================================
# 11. Test Backend
# ==========================================
def test_available_backends_contains_cpu_numpy():
    assert "cpu_numpy" in specular.backend._AVAILABLE_BACKENDS

def test_current_backend_is_available():
    assert specular.backend._CURRENT_BACKEND in specular.backend._AVAILABLE_BACKENDS

def test_change_backend_valid():
    original = specular.backend._CURRENT_BACKEND
    specular.change_backend("cpu_numpy")
    assert specular.backend._CURRENT_BACKEND == "cpu_numpy"
    specular.change_backend(original)

def test_change_backend_invalid():
    with pytest.raises(ValueError):
        specular.change_backend("garbage_backend")

def test_change_backend_cpu_jax_routes_calculation():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    original = specular.backend._CURRENT_BACKEND
    try:
        specular.change_backend("cpu_jax")
        result = specular.derivative(lambda x: x**2, x=2.0)
        assert float(result) == pytest.approx(4.0)

        relu = lambda x: jnp.maximum(x, 0)
        assert float(specular.derivative(relu, x=0.0)) == pytest.approx(math.sqrt(2) - 1)
    finally:
        specular.change_backend(original)

def test_backend_info_output(capsys):
    specular.backend_info()
    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    assert len(lines) == 3
    assert "supported backends" in lines[0]
    assert "available backends" in lines[1]
    assert "current backend" in lines[2]


# ==========================================
# 12. Multi-backend tests
# ==========================================
BACKENDS = ["cpu_numpy"]
if (os.cpu_count() or 1) > 1 and importlib.util.find_spec("numba") is not None:
    BACKENDS.append("cpu_numba")
if importlib.util.find_spec("jax") is not None:
    BACKENDS.append("cpu_jax")

@pytest.fixture(autouse=True)
def reset_backend():
    original = specular.backend._CURRENT_BACKEND
    yield
    specular.change_backend(original)

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_derivative_backends(backend_name):
    specular.change_backend(backend_name)
    f = lambda x: x**2
    assert specular.derivative(f, x=2.0) == pytest.approx(4.0, rel=1e-4)

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_gradient_backends(backend_name):
    specular.change_backend(backend_name)
    if backend_name == "cpu_jax":
        jnp = pytest.importorskip("jax.numpy")
        f = lambda x: jnp.sum(jnp.square(x))
    else:
        f = lambda x: np.sum(np.square(x))
    grad = specular.gradient(f, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(grad, [2.0, 4.0, 6.0], rtol=1e-4)

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_jacobian_backends(backend_name):
    specular.change_backend(backend_name)
    if backend_name == "cpu_jax":
        jnp = pytest.importorskip("jax.numpy")
        f = lambda x: jnp.array([x[0]**2, x[0] + x[1]])
    else:
        f = lambda x: [x[0]**2, x[0] + x[1]]
    J = specular.jacobian(f, [2.0, 1.0])
    np.testing.assert_allclose(J, [[4.0, 0.0], [1.0, 1.0]], rtol=1e-4)
