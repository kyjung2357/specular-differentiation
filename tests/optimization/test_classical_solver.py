import pytest
import numpy as np
from specular.optimization.solver import gradient_descent, adam, bfgs, specular_gradient, stochastic_specular_gradient, hybrid_specular_gradient
from specular.optimization.result import OptimizationResult

def quadratic_np(x):
    return float(np.sum(np.asarray(x)**2))

def f_comp_1(x):
    return float(x[0]**2)

def f_comp_2(x):
    return float(x[1]**2)

x_0 = np.array([1.0, 1.0])

def test_gradient_descent_convergence():
    res = gradient_descent(
        objective_function=quadratic_np, 
        initial_point=x_0, 
        step_size='square_summable_not_summable', 
        a=0.5, 
        b=0.0, 
        max_iter=100, 
        print_bar=False
    )
    assert res.method == "gradient_descent"
    assert res.func_val < 0.1
    hist_vars, hist_vals, _ = res.get_history()
    assert len(hist_vals) <= 101

def test_adam_convergence():
    res = adam(
        objective_function=quadratic_np, 
        initial_point=x_0, 
        step_size='constant', 
        a=0.1, 
        max_iter=100, 
        print_bar=False
    )
    assert res.method == "Adam"
    np.testing.assert_allclose(res.solution, [0.0, 0.0], atol=1e-2)

def test_bfgs_convergence():
    res = bfgs(
        objective_function=quadratic_np, 
        initial_point=x_0, 
        max_iter=100, 
        print_bar=False
    )
    assert res.method == "BFGS"
    np.testing.assert_allclose(res.solution, [0.0, 0.0], atol=1e-3)

def test_specular_gradient_convergence():
    res = specular_gradient(
        objective_function=quadratic_np, 
        initial_point=x_0, 
        step_size='constant', 
        a=0.1, 
        max_iter=100, 
        print_bar=False
    )
    assert res.method == "specular_gradient"
    assert res.func_val < 0.1

def test_stochastic_specular_gradient():
    res = stochastic_specular_gradient(
        objective_function=quadratic_np, 
        initial_point=x_0, 
        step_size='square_summable_not_summable', 
        a=1.0, 
        b=0.0, 
        f_j=[f_comp_1, f_comp_2], 
        max_iter=50, 
        print_bar=False, 
        tol=1e-6
    )
    assert res.method == "stochastic_specular_gradient"
    assert res.func_val < 2.0

def test_hybrid_specular_gradient():
    res = hybrid_specular_gradient(
        objective_function=quadratic_np, 
        initial_point=x_0, 
        step_size='not_summable', 
        a=0.5, 
        f_j=[f_comp_1, f_comp_2], 
        switch_iter=5, 
        max_iter=15, 
        print_bar=False, 
        tol=1e-6
    )
    assert res.method == "hybrid_specular_gradient"
    assert res.func_val < 2.0

def test_result_methods():
    res = gradient_descent(
        objective_function=quadratic_np, 
        initial_point=x_0, 
        step_size='constant', 
        a=0.1, 
        max_iter=5, 
        print_bar=False
    )
    sol, fval, runtime = res.last_record()
    np.testing.assert_allclose(sol, res.solution)
    assert fval == res.func_val
    h_vars, h_vals, _ = res.get_history()
    assert len(h_vals) <= 6
