# 2. API Reference

The specular package consists of the following modules and subpackages.

## [2.1. Calculation](calculation.md)

* [`specular.calculation`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/calculation.py): five primary functions to calculate specular differentiation, depending on the dimension of input.

## [2.2. ODE](ode.md)

* [`specular.ode.solver`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/ode/solver.py): the explicit Euler, implicit Euler, Crank-Nicolson, specular trigonometric, specular Euler schemes.

* [`specular.ode.result`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/ode/result.py): the `ODEResult` class to store the results.

## [2.3. Optimization](optimization.md)

* [`specular.optimization.step_size`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/optimization/step_size.py): the `StepSize` class to define step size $h_k$.

* [`specular.optimization.solver`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/optimization/solver.py): the specular gradient method.

* [`specular.optimization.classical_solver`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/optimization/classical_solver.py): the gradient descent method, Adam, and BFGS. **Lazy importing**.

* [`specular.optimization.result`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/optimization/result.py): the `OptimizationResult` class to store the results.

## [2.4. JAX Backend](jax.md)

* [`specular.jax.calculation`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/jax/calculation.py): JAX implementations of the calculations in `specular.calculation`. **Lazy importing**.

* [`specular.jax.optimization.solver`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/jax/optimization/solver.py): JAX implementations of the numerical schemes in `specular.optimization.solver`. **Lazy importing**.