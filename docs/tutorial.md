# Specular Differentiation tutorial

Denote by ℕ the set of all positive integers. 
For each $n \in \mathbb{N}$, denote by the $n$-dimensional Euclidean space.

## 1. Calculation of specular differentiation

In `core.py`, there are four modules to calculate specular differentiation, depending on the dimension. 

### 1.1 the one-dimensional Euclidean space

In $\mathbb{R}$, the *specular derivative* can be calculated using the function `derivative`.

```python
>>> import specular
>>> 
>>> def f(x):
>>>     return max(x, 0.0)
>>> 
>>> specular.derivative(f, x=0.0)
0.41421356237309515
```

### 1.2 the $n$-dimensional Euclidean space ($n>1$)

In $\mathbb{R}^n$, the *specular directional derivative* of a function $f: \mathbb{R}^n \to \mathbb{R}$ at a point $x \in \mathbb{R}^n$ in the direction $v \in \mathbb{R}^n$ can be calculated using the function `directional_derivative`.

```python
>>> import specular
>>> import math 
>>>
>>> f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
>>> specular.directional_derivative(f, x=[0.0, 0.1, -0.1], v=[1.0, -1.0, 2.0])
-2.1213203434708223
```

Let $e_1, e_2, \ldots, e_n$ be the standard basis of $\mathbb{R}^n$.
For each $i \in \mathbb{N}$ with $1 \leq i \leq n$, the *specular partial derivative* with respect to a variable $x_i$ can be calculated using the function `partial_derivative`, which yields the same result as `directional_derivative` with direction $v=e_i$.

```python
>>> import specular
>>> import math
>>>
>>> def f(x):
>>>     return math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
>>>
>>> specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=2)
0.8859268982863702
>>> specular.directional_derivative(f, x=[0.1, 2.3, -1.2], v=[0.0, 1.0, 0.0])
0.8859268982863702
```

Also, the *specular gradient* can be calculated using `gradient`.

```python
>>> import specular
>>> import numpy as np
>>>
>>> def f(x):
>>>     return np.linalg.norm(x)
>>> 
>>> specular.gradient(f, x=[0.1, 2.3, -1.2])
[ 0.03851856  0.8859269  -0.46222273]
>>> specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=1)
0.03851856078540371
>>> specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=2)
0.8859268982863702
>>> specular.partial_derivative(f, x=[0.1, 2.3, -1.2], i=3)
-0.4622227292028128
```




