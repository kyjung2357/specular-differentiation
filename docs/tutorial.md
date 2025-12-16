# Specular Differentiation tutorial

## Calculation of specular differentiation

In `specular_derivative.py`, there are five modules. 

### the one-dimensional Euclidean space

A specular derivative of a function on the one-dimensional Euclidean space can be calculated as follows. 

```python
>>> import specular_diff as sd
>>> 
>>> def f(x):
>>>     return max(x, 0)
>>> 
>>> sd.specular_derivative(f, x=0)
0.41421356237309515
```

### the $n$-dimensional Euclidean space ($n>1$)

