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

In $\mathbb{R}^n$, the specular partial derivative with respect to a variable $x_i$ ($1 \leq i \leq n$) can be calculated using the function `specular_partial_derivative`, which yields the same result as `specular_directional_derivative` with direction $v=e_i$, where $e_1, e_2, \ldots, e_n$ are the standard basis of $\mathbb{R}^n$.

```python
>>> import specular_diff as sd
>>> import math
>>>
>>> def f(x):
>>>     return math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
>>>
>>> sd.specular_partial_derivative(f, x=[0.1, 2.3, -1.2], i=2)
0.8859268982863702
>>> sd.specular_directional_derivative(f, x=[0.1, 2.3, -1.2], i=[0.0, 1.0, 0.0])
0.8859268982863702
```

Also, the specular gradient can be calculated using `specular_gradient`.
```python
>>> import specular_diff as sd
>>> import math
>>>
>>> def f(x):
>>>     return math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
>>> sd.specular_gradient(f, x=[0.1, 2.3, -1.2])
[ 0.03851856  0.8859269  -0.46222273]
```


