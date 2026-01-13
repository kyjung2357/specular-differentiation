
# 2. API Reference

The specular package consists of the following modules and subpackages.

## [2.1. Calculation](calculation.md)

* The [`specular.calculation`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/calculation.py) module provides five primary functions to calculate specular differentiation, depending on the dimension of input.

    | Function | Space | Description |Input Type | Output Type |
    | :--- | :--- | :--- | :--- | :--- |
    | `derivative` | $\mathbb{R} \to \mathbb{R}^m$ | specular derivative | `float` | `float`, `np.ndarray`
    | `directional_derivative` | $\mathbb{R}^n \to \mathbb{R}$ | specular directional derivative in direction $v \in \mathbb{R}^n$ | `np.ndarry` | `float` |
    | `partial_derivative` | $\mathbb{R}^n \to \mathbb{R}$ | specular partial derivative w.r.t. $v = x_i$ | `np.ndarray` | `float`
    | `gradient` | $\mathbb{R}^n \to \mathbb{R}$ | specular gradient vector | `np.ndarray` | `np.ndarray` |
    | `jacobian` | $\mathbb{R}^n \to \mathbb{R}^m$ | specular jacobian matrix | `np.ndarray` | `np.ndarray` |

## [2.2 ODE](ode.md)

* Let the source function $F:[t_0, T] \times ℝ \to ℝ$ be given, and the initial data $u_0:ℝ \to ℝ$ be given. 
Consider the initial value problem:

$$
u'(t) = F(t, u(t))
$$ 

with the initial condition $u(t_0) = u_0(t_0)$.

* To solve the problem numerically, the subpackage [`specular.ode.solver`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/ode/solver.py) provides the following numerical schemes:

  * the *specular Euler* scheme (Type 1 ~ 6)
  * the *specular trigonometric* scheme
  * the explicit Euler scheme
  * the implicit Euler scheme
  * the Crank-Nicolson scheme

* The [`specular.ode.result`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/ode/result.py) module provides the `ODEResult` class to store the results.

## [2.3 Optimization](optimization.md)

* Consider the optimization problem:

$$
\min_{x \in \mathbb{R}^n} f(x),
$$

where $f:\mathbb{R}^n \to \mathbb{R}$ is convex.

* To solve the problem numerically, the subpackage [`specular.optimization.solver`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/optimization/solver.py) provides the following methods:

  * the *specular gradient (SPEG)* method
  * the *stochastic specular gradient (S-SPEG)* method
  * the *hybrid specular gradient (H-SPEG)* method

* Given an initial point $x_0 \in \mathbb{R}^n$, method takes the form: 

$$
x_{k+1} = x_k - h_k s_k,
$$

where $h_k > 0$ is the step size and $s_k$ is the specular gradient for each $k \in \mathbb{N}$.

* The [`specular.optimization.step_size`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/optimization/step_size.py) module provides the `StepSize` class to define step size $h_k$.

    | Name | Rule (Formula) | Type | Input | Description |
    | :--- | :--- | :--- | :--- | :--- |
    | `constant` | $h_k = a$ | `float` | `a` | Fixed step size for all $k$. |
    | `not_summable` | $h_k = a / \sqrt{k}$ | `float` | `a` | $\lim_{k \to \infty }h_k = 0$, but $\sum h_k = \infty$. |
    | `square_summable_not_summable` | $h_k = a / (b + k)$ | `list` | `[a, b]` | $\sum h_k^2 < \infty$ and $\sum h_k = \infty$. |
    | `geometric_series` | $h_k = a \cdot r^k$ | `list` | `[a, r]` | Exponentially decaying step size. |
    | `user_defined` | Custom | `Callable` | `f(k)` | User-provided function of iteration $k$. |

* The [`specular.optimization.result`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/optimization/result.py) module provides the `OptimizationResult` class to store the results.

## [2.4 JAX Backend](jax.md)

* The [`specular.jax.calculation`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/jax/calculation.py) module provides JAX implementations of the calculations in `specular.calculation`.

* The [`specular.jax.optimization.solver`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/jax/optimization/solver.py) module provides JAX implementations of the numerical schemes in `specular.optimization.solver`.