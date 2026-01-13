# 2.1. Calculation

The [`calculation`](../specular/calculation.py) module provides five primary functions to calculate specular differentiation, depending on the dimension of input.

---

## Quick Reference

| Function | Space | Description |Input Type | Output Type |
| :--- | :--- | :--- | :--- | :--- |
| `derivative` | $\mathbb{R} \to \mathbb{R}^m$ | specular derivative | `float` | `float`, `np.ndarray`
| `directional_derivative` | $\mathbb{R}^n \to \mathbb{R}$ | specular directional derivative in direction $v \in \mathbb{R}^n$ | `np.ndarry` | `float` |
| `partial_derivative` | $\mathbb{R}^n \to \mathbb{R}$ | specular partial derivative w.r.t. $v = x_i$ | `np.ndarray` | `float`
| `gradient` | $\mathbb{R}^n \to \mathbb{R}$ | specular gradient vector | `np.ndarray` | `np.ndarray` |
| `jacobian` | $\mathbb{R}^n \to \mathbb{R}^m$ | specular jacobian matrix | `np.ndarray` | `np.ndarray` |

---

## 1.1 One-dimensional Euclidean Space ($n=1$)

In $ℝ$, the *specular derivative* can be calculated using the function `derivative`.

```python
import specular

def f(x):
    return max(x, 0.0)

specular.derivative(f, x=0.0)
# Output: 0.41421356237309515
```

## 1.2 the $n$-dimensional Euclidean space ($n>1$)

In $ℝ^n$, the *specular directional derivative* of a function $f: ℝ^n \to ℝ$ at a point $x \in ℝ^n$ in the direction $v \in ℝ^n$ can be calculated using the function `directional_derivative`.

```python
import specular
import math 

f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
specular.directional_derivative(f, x=[0.0, 0.1, -0.1], v=[1.0, -1.0, 2.0])
# Output: -2.1213203434708223
```

Let $e_1, e_2, \ldots, e_n$ be the standard basis of $ℝ^n$.
For each $i \in ℕ$ with $1 \leq i \leq n$, the *specular partial derivative* with respect to a variable $x_i$ can be calculated using the function `partial_derivative`, which yields the same result as `directional_derivative` with direction $v=e_i$.

```python
import specular
import math

def f(x):
    return math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=2)
# Output: 0.8859268982863702
specular.directional_derivative(f, x=[0.1, 2.3, -1.2], v=[0.0, 1.0, 0.0])
# Output: 0.8859268982863702
```

Also, the *specular gradient* can be calculated using `gradient`.

```python
import specular
import numpy as np

def f(x):
    return np.linalg.norm(x)

specular.gradient(f, x=[0.1, 2.3, -1.2])
# Output: [ 0.03851856  0.8859269  -0.46222273]
specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=1)
# Output: 0.03851856078540371
specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=2)
# Output: 0.8859268982863702
specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=3)
# Output: -0.4622227292028128
```