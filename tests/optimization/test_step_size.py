import pytest
from specular.optimization.step_size import StepSize

# ==========================================
# 1. Valid Cases
# ==========================================

def test_constant_step_size():
    """Test 'constant' rule: h_k = a"""
    # a = 0.5
    step = StepSize(name='constant', parameters=0.5)
    
    # k=1, k=100
    assert step(1) == 0.5
    assert step(100) == 0.5

def test_not_summable():
    """Test 'not_summable' rule: h_k = a / sqrt(k)"""
    # a = 2.0
    step = StepSize(name='not_summable', parameters=2.0)
    
    # k=1 -> 2.0 / 1 = 2.0
    assert step(1) == pytest.approx(2.0)
    # k=4 -> 2.0 / 2 = 1.0
    assert step(4) == pytest.approx(1.0)

def test_square_summable():
    """Test 'square_summable_not_summable' rule: h_k = a / (b + k)"""
    # a = 10, b = 2
    step = StepSize(name='square_summable_not_summable', parameters=[10.0, 2.0])
    
    # k=1 -> 10 / (2 + 1) = 3.333...
    assert step(1) == pytest.approx(10/3)
    # k=8 -> 10 / (2 + 8) = 1.0
    assert step(8) == pytest.approx(1.0)

def test_geometric_series():
    """Test 'geometric_series' rule: h_k = a * r^k"""
    # a = 1.0, r = 0.5
    step = StepSize(name='geometric_series', parameters=[1.0, 0.5])
    
    # k=1 -> 1.0 * 0.5^1 = 0.5
    assert step(1) == pytest.approx(0.5)
    # k=2 -> 1.0 * 0.5^2 = 0.25
    assert step(2) == pytest.approx(0.25)

def test_user_defined():
    """Test 'user_defined' callable."""
    # Custom rule: h_k = 1 / k^2
    custom_rule = lambda k: 1.0 / (k**2)
    step = StepSize(name='user_defined', parameters=custom_rule)
    
    assert step(1) == 1.0
    assert step(2) == 0.25

# ==========================================
# 2. Error Cases
# ==========================================

def test_invalid_name():
    with pytest.raises(ValueError, match="Invalid step size"):
        StepSize(name="magic_step", parameters=1.0)

def test_constant_error():
    with pytest.raises(ValueError, match="positive number required"):
        StepSize('constant', 0.0)
        
    with pytest.raises(ValueError):
        StepSize('constant', -1.0)
        
    with pytest.raises(TypeError):
        StepSize('constant', [1.0])

def test_square_summable_error():
    with pytest.raises(ValueError, match="Invalid length"):
        StepSize('square_summable_not_summable', [1.0])

    # a <= 0
    with pytest.raises(ValueError):
        StepSize('square_summable_not_summable', [-1.0, 1.0])

    # b < 0
    with pytest.raises(ValueError):
        StepSize('square_summable_not_summable', [1.0, -0.1])

def test_geometric_error():
    # r >= 1
    with pytest.raises(ValueError):
        StepSize('geometric_series', [1.0, 1.1])
    
    # r <= 0
    with pytest.raises(ValueError):
        StepSize('geometric_series', [1.0, -0.5])

def test_user_defined_error():
    with pytest.raises(TypeError):
        StepSize('user_defined', "not a function") # type: ignore