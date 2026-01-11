import pytest
import numpy as np
import specular

def test_Euler_scheme_Type1_simple():
    """
    Test 'Euler_scheme' (Type 1) with a simple linear ODE:
      u' = -2u, u(0) = 1.0
      Analytical solution: u(t) = exp(-2t)
    """
    def F(t, u):
        return -2.0 * u 

    def exact_sol(t):
        return np.exp(-2.0 * t)

    t_0 = 0.0
    u_0 = 1.0
    T = 2.5
    h = 0.01

    res = specular.Euler_scheme(
        of_Type='1', 
        F=F, 
        t_0=t_0, 
        u_0=u_0, 
        T=T, 
        h=h
    )

    times, u_values = res.history() 
    
    expected_len = int((T - t_0) / h) + 1
    assert len(times) == len(u_values)
    assert len(times) >= expected_len - 1 

    assert u_values[0] == u_0

    max_error = res.total_error(exact_sol, norm='max')
    assert max_error < 0.05

def test_ODEResult_methods():
    """
    Check if ODEResult helper methods (table, etc.) run without errors.
    """
    t = np.linspace(0, 1, 10)
    u = t**2
    history = {"variables": t, "values": u}
    
    res = specular.ode.ODEResult("Euler", 0.1, history) 

    t_out, u_out = res.history()
    np.testing.assert_array_equal(t_out, t)

    exact = lambda x: x**2
    assert res.total_error(exact) == pytest.approx(0.0)
    
    df = res.table(exact_sol=exact)
    assert df.shape == (10, 3)