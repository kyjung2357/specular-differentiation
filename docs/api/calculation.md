# Calculation

Source code in [`specular.calculation.py`](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/calculation.py)

This module provides five primary functions to calculate specular differentiation, depending on the dimension of input.

| Function | Space | Description |Input Type | Output Type |
| :--- | :--- | :--- | :--- | :--- |
| `derivative` | $\mathbb{R} \to \mathbb{R}^m$ | specular derivative | `float` | `float`, `np.ndarray`
| `directional_derivative` | $\mathbb{R}^n \to \mathbb{R}$ | specular directional derivative in direction $v \in \mathbb{R}^n$ | `np.ndarry` | `float` |
| `partial_derivative` | $\mathbb{R}^n \to \mathbb{R}$ | specular partial derivative w.r.t. $v = x_i$ | `np.ndarray` | `float`
| `gradient` | $\mathbb{R}^n \to \mathbb{R}$ | specular gradient vector | `np.ndarray` | `np.ndarray` |
| `jacobian` | $\mathbb{R}^n \to \mathbb{R}^m$ | specular jacobian matrix | `np.ndarray` | `np.ndarray` |

This module provides implementations of specular directional derivatives, specular partial derivatives, specular derivatives, specular gradients, and specular Jacobians.

The calculations are based on the function $\mathcal{A}:\mathbb{R}^2 \to \mathbb{R}$ defined by

$$
\mathcal{A}(\alpha, \beta) =
\begin{cases}
    \frac{\alpha \beta - 1 + \sqrt{(1 + \alpha^2)(1 + \beta^2)}}{\alpha + \beta} & \text{if } \alpha + \beta \neq 0, \\
    0 & \text{otherwise.}
\end{cases}
$$

The parameters $\alpha$ and $\beta$ are intended to represent right and left derivatives.
In the code, computations are based on the finite difference approximation of one-sided (directional) derivatives:

$$
\alpha \approx \frac{f(x + hv) - f(x)}{h}
\qquad \text{and} \qquad
\beta \approx \frac{f(x) - f(x - hv)}{h},
$$

where $f : \mathbb{R}^n \to \mathbb{R}$ is a function, $h > 0$ is a real number, and $x, v \in \mathbb{R}^n$ are vectors.

NumPy and Numba approximate this data from function values using one-sided finite differences, while JAX computes it using automatic differentiation at shifted points.

## One-dimensional Euclidean Space ($n=1$)

In $ℝ$, the *specular derivative* can be calculated using the function `derivative`.

```python
import specular

def f(x):
    return max(x, 0.0)

print(specular.derivative(f, x=0.0))
```

```text
0.41421356237309515
```

## The $n$-dimensional Euclidean space ($n>1$)

In $ℝ^n$, the *specular directional derivative* of a function $f: ℝ^n \to ℝ$ at a point $x \in ℝ^n$ in the direction $v \in ℝ^n$ can be calculated using the function `directional_derivative`.

```python
import specular
import math

f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
print(specular.directional_derivative(f, x=[0.0, 0.1, -0.1], v=[1.0, -1.0, 2.0]))
```

```text
-2.1213203434708223
```

Let $e_1, e_2, \ldots, e_n$ be the standard basis of $ℝ^n$.
For each $i \in ℕ$ with $1 \leq i \leq n$, the *specular partial derivative* with respect to a variable $x_i$ can be calculated using the function `partial_derivative`, which yields the same result as `directional_derivative` with direction $v=e_i$.

```python
import specular
import math

def f(x):
    return math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

print(specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=2))
print(specular.directional_derivative(f, x=[0.1, 2.3, -1.2], v=[0.0, 1.0, 0.0]))
```

```text
0.8859268982863702
0.8859268982863702
```

Also, the *specular gradient* can be calculated using `gradient`.

```python
import specular
import numpy as np

def f(x):
    return np.linalg.norm(x)

print(specular.gradient(f, x=[0.1, 2.3, -1.2]))
print(specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=1))
print(specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=2))
print(specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=3))
```

```text
[ 0.03851856  0.8859269  -0.46222273]
0.03851856078540371
0.8859268982863702
-0.4622227292028128
```

## API Reference

- [`specular.calculation.A`](calculation/A.md) <span class="api-kind">module</span>
- [`specular.calculation.derivative`](calculation/derivative.md) <span class="api-kind">module</span>
- [`specular.calculation.directional_derivative`](calculation/directional_derivative.md) <span class="api-kind">module</span>
- [`specular.calculation.partial_derivative`](calculation/partial_derivative.md) <span class="api-kind">module</span>
- [`specular.calculation.gradient`](calculation/gradient.md) <span class="api-kind">module</span>
- [`specular.calculation.jacobian`](calculation/jacobian.md) <span class="api-kind">module</span>
