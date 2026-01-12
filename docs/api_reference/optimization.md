# 3. Optimization

Consider the optimization problem:

$$
\min_{x \in \mathbb{R}^n} f(x),
$$

where $f:\mathbb{R}^n \to \mathbb{R}$ is convex.
To solve the problem numerically, the module The [`optimization`](/specular/optimization/solver.py) subpackage  provides the following methods:

* the *specular gradient (SPEG)* method
* the *stochastic specular gradient (S-SPEG)* method
* the *hybrid specular gradient (H-SPEG)* method

Given an initial point $x_0 \in \mathbb{R}^n$, method takes the form: 
$$
x_{k+1} = x_k - h_k s_k,
$$
where $h_k > 0$ is the step size and $s_k$ is the specular gradient for each $k \in \mathbb{N}$.


## 3.1 Step Size Rules (`StepSize`)

The `StepSize` class defines how the step size $h_k$ evolves during optimization:
$x_{k+1} = x_k - h_k s_k$

### Available Options

| Name | Rule (Formula) | Type | Input | Description |
| :--- | :--- | :--- | :--- | :--- |
| `constant` | $h_k = a$ | `float` | `a` | Fixed step size for all $k$. |
| `not_summable` | $h_k = a / \sqrt{k}$ | `float` | `a` | $\lim_{k \to \infty }h_k = 0$, but $\sum h_k = \infty$. |
| `square_summable_not_summable` | $h_k = a / (b + k)$ | `list` | `[a, b]` | $\sum h_k^2 < \infty$ and $\sum h_k = \infty$. |
| `geometric_series` | $h_k = a \cdot r^k$ | `list` | `[a, r]` | Exponentially decaying step size. |
| `user_defined` | Custom | `Callable` | `f(k)` | User-provided function of iteration $k$. |

---

### Quick Example
```python
from specular.optimization.step_size import StepSize

# Use a square-summable rule: h_k = 10 / (2 + k)
step = StepSize(name='square_summable_not_summable', parameters=[10.0, 2.0])
h_1 = step(1)
print(h_1)
# Output: 3.3333333333333335
```

## 3.2 The specular gradient method

### The one dimensional case
```python
import specular 

# Objective function: f(x) = |x|
def f(x):
    return abs(x)

step_size = specular.StepSize('constant', 0.1) 

# Specular gradient method
res = specular.gradient_method(f=f, x_0=1.0, step_size=step_size, form='specular gradient', max_iter=20)
```

### Higher dimensional cases

```python
import specular 

# Objective function: f(x) = sum(x^2)
def f(x):
    return float(np.sum(np.array(x)**2))

# Component functions for Stochastic test
# f(x) = x1^2 + x2^2
# f1(x) = x1^2, f2(x) = x2^2
def f_comp_1(x):
    return x[0]**2

def f_comp_2(x):
    return x[1]**2

f_components = [f_comp_1, f_comp_2]

x_0 = [1.0, 1.0]
step_size = specular.StepSize('square_summable_not_summable', [0.5, 1.0]) 

# Specular gradient method
res1 = specular.gradient_method(f=f, x_0=x_0, step_size=step_size, form='specular gradient', max_iter=50)

# Stochastic specular gradient method
res2 = specular.gradient_method(f=f, x_0=x_0, step_size=step_size, form='stochastic', f_j=f_components, max_iter=100)

# hybrid specular gradient method
res3 = specular.gradient_method(f=f_quad_vector, x_0=x_0, step_size=step_size, form='hybrid', f_j=f_components, switch_iter=5, max_iter=20)
```

### 3.3 `OptimizationResult`

The class `OptimizationResult` collects the optimization results.
To get history of optimization, call `history()`.

```python
import specular 

# Objective function: f(x) = sum(x^2)
def f(x):
    return float(np.sum(np.array(x)**2))

x_0 = [1.0, 1.0]
step_size = specular.StepSize('square_summable_not_summable', [0.5, 1.0]) 

# Specular gradient method
res_x, res_f, res_time = specular.gradient_method(f=f, x_0=x_0, step_size=step_size, form='specular gradient', max_iter=50).history()
```

## Continue reading

### [API Reference](./README.md)

* [1. Calculation](./calculation.md)
* [2. ODE](./ode.md)
* [4. JAX backend](./jax.md)

### [Examples](/examples/README.md)

* [2026-Jung](/examples/ode/2026-Jung/main.ipynb)
* [2024-Jung-Oh](/examples/optimization/2024-Jung-Oh/main.py)
* [2026-Jung](/examples/optimization/2026-Jung/main.py)