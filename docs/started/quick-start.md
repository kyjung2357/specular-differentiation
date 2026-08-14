# 1.2. Quick start

The following simple example calculates the specular derivative of the [ReLU function](https://en.wikipedia.org/wiki/Rectified_linear_unit) $f(x) = max(0, x)$ at the origin.

```python
import specular

ReLU = lambda x: max(x, 0)
specular.derivative(ReLU, x=0)
```

```text
0.41421356237309503
```
