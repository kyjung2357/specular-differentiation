# Specular Differentiation tutorial

## 1. Calculation of specular differentiation

In `specular_derivative.py`, there are five modules. 

### 1.1 the one-dimensional Euclidean space

In $\mathbb{R}$, the specular derivative can be calculated using the function `specular_derivative`, which yields the same result as `specular_directional_derivative` with direction $v=1$.

```python
>>> import specular_diff as sd
>>> 
>>> def f(x):
>>>     return max(x, 0.0)
>>> 
>>> sd.specular_derivative(f, x=0.0)
0.41421356237309515
>>> sd.specular_directional_derivative(f, x=0.0, v=1.0)
0.41421356237309515
```

### 1.2 the $n$-dimensional Euclidean space ($n>1$)

In $\mathbb{R}^n$, the specular partial derivative with respect to a variable $x_i$ ($1 \leq i \leq n$) can be calculated using the function `specular_partial_derivative`, which yields the same result as `specular_directional_derivative` with direction $v=e_i$, where $\left\{ e_i \right\}_{i=1}^n$ is the standard basis of $\mathbb{R}^n$.

```python
>>> import math
>>> from specular_diff as sd
>>>
>>> def f(x):
>>>     return math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
>>>
>>> sd.specular_partial_derivative(f, x=[0.1, 2.3, -1.2], i=2)
-0.4622227292028128
>>> sd.specular_directional_derivative(f, x=[0.1, 2.3, -1.2], i=[0.0, 1.0, 0.0])
-0.4622227292028128
```

A specular partial derivative of a function on $\mathbb{R}^n$ can be calculated in two ways.


Three-dimensional input:
>>> import math
    
    >>> f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    >>> specular_partial_derivative(f, x=[0.1, 2.3, -1.2], i=2)
    -0.4622227292028128

    >>> f = lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    >>> specular_directional_derivative(f, x=[0.0, 0.1, -0.1], v=[1.0, -1.0, 2.0])
    -2.1213203434708223


>>> import numpy as np

    >>> f = lambda x: np.linalg.norm(x)
    >>> specular_gradient(f, x=[1.4, -3.47, 4.57, 9.9])
    array([ 0.12144298, -0.3010051 ,  0.39642458,  0.85877534])

