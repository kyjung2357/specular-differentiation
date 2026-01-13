# 2.3. Optimization

::: specular.optimization.step_size.StepSize
    options:
      show_root_heading: false
      members:
        - __init__
        - __call__




### Quick Example
```python
from specular.optimization.step_size import StepSize

# Use a square-summable rule: h_k = 10 / (2 + k)
step = StepSize(name='square_summable_not_summable', parameters=[10.0, 2.0])
h_1 = step(1)
print(h_1)
# Output: 3.3333333333333335
```

## 2.3.2 The specular gradient method

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

### 2.3.3 `OptimizationResult`

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