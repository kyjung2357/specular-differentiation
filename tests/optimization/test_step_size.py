import pytest
from specular.optimization import StepSchedule

# ==========================================
# 1. Valid Cases
# ==========================================

def test_constant_step_size():
    """Test 'constant' rule: h_k = a"""
    # a = 0.5
    step = StepSchedule(name='constant', a=0.5)
    
    # k=1, k=100
    assert step(1) == 0.5
    assert step(100) == 0.5

def test_not_summable():
    """Test 'not_summable' rule: h_k = a / sqrt(k)"""
    # a = 2.0
    step = StepSchedule(name='not_summable', a=2.0)
    
    # k=1 -> 2.0 / 1 = 2.0
    assert step(1) == pytest.approx(2.0)
    # k=4 -> 2.0 / 2 = 1.0
    assert step(4) == pytest.approx(1.0)

def test_square_summable():
    """Test 'square_summable_not_summable' rule: h_k = a / (b + k)"""
    # a = 10, b = 2
    step = StepSchedule(name='square_summable_not_summable', a=10.0, b=2.0)
    
    # k=1 -> 10 / (2 + 1) = 3.333...
    assert step(1) == pytest.approx(10/3)
    # k=8 -> 10 / (2 + 8) = 1.0
    assert step(8) == pytest.approx(1.0)

def test_geometric_series():
    """Test 'geometric_series' rule: h_k = a * r^k"""
    # a = 1.0, r = 0.5
    step = StepSchedule(name='geometric_series', a=1.0, r=0.5)
    
    # k=1 -> 1.0 * 0.5^1 = 0.5
    assert step(1) == pytest.approx(0.5)
    # k=2 -> 1.0 * 0.5^2 = 0.25
    assert step(2) == pytest.approx(0.25)

def test_user_defined():
    """Test 'user_defined' callable."""
    # Custom rule: h_k = 1 / k^2
    custom_rule = lambda k: 1.0 / (k**2)
    step = StepSchedule(name='user_defined', user_defined_rule=custom_rule)
    
    assert step(1) == 1.0
    assert step(2) == 0.25

# ==========================================
# 2. Error Cases
# ==========================================

def test_invalid_name():
    with pytest.raises(ValueError, match="Invalid step size"):
        StepSchedule(name="magic_step", a=1.0)

def test_constant_error():
    with pytest.raises(ValueError):
        StepSchedule('constant', a=0.0)
        
    with pytest.raises(ValueError):
        StepSchedule('constant', a=-1.0)

def test_square_summable_error():
    with pytest.raises(ValueError):
        StepSchedule('square_summable_not_summable')

    # a <= 0
    with pytest.raises(ValueError):
        StepSchedule('square_summable_not_summable', a=-1.0, b=1.0)

    # b < 0
    with pytest.raises(ValueError):
        StepSchedule('square_summable_not_summable', a=1.0, b=-0.1)

def test_geometric_error():
    # r >= 1
    with pytest.raises(ValueError):
        StepSchedule('geometric_series', a=1.0, r=1.1)
    
    # r <= 0
    with pytest.raises(ValueError):
        StepSchedule('geometric_series', a=1.0, r=-0.5)