# Calculation

The public calculation API provides the scaled angular mean and the three
finite-difference operators used for the four finite-dimensional real map
types.

| Function | Map type | Result |
| :--- | :--- | :--- |
| `scaled_mean` | $\mathbb R^2\times(0,\infty)\to\mathbb R$ | scalar or broadcast array |
| `derivative` | $\mathbb R\to\mathbb R$ or $\mathbb R\to\mathbb R^m$ | scalar or vector |
| `gradient` | $\mathbb R^n\to\mathbb R$ | vector of shape `(n,)` |
| `jacobian` | $\mathbb R^n\to\mathbb R^m$ | matrix of shape `(m, n)` |

All functions use centered samples at `x` and `x +/- h`, with the center value
evaluated once. If `h` is omitted, the backend chooses
`eps(dtype)**(1/3) * max(1, abs(x))`; gradients and Jacobians use a separate
step for each coordinate. An explicit `h` must be a concrete, finite, positive
real scalar and is validated before the callback is evaluated. Values that are
positive but too small or too large to form distinct, finite samples at `x`
are rejected as well when `x` is concrete. Under JAX transformations, a
traced `x` cannot be inspected before execution, so an ineffective sample may
instead appear as a non-finite result.

## Examples

The scaled angular mean is evaluated elementwise by the selected backend.
Its `sigma` argument is a concrete, finite, positive scalar. Evaluation uses
the defining rescaling

\[
\mathcal C_\sigma(\alpha,\beta)
=\sigma\mathcal C(\alpha/\sigma,\beta/\sigma),
\]

so its floating-point range and return type follow the selected backend.
Under JAX transformations, `alpha` and `beta` may be traced while `sigma`
must remain static. The exact diagonal and antidiagonal identities
\(\mathcal C_\sigma(\alpha,\alpha)=\alpha\) and
\(\mathcal C_\sigma(\alpha,-\alpha)=0\) are preserved even when the direct
rescaling would overflow or underflow. Other extreme inputs remain limited by
the selected backend dtype's representable range.

### Scaled angular mean

```python
import specular

print(specular.scaled_mean(1.0, -0.5, sigma=2.0))
```

A vector-valued derivative produces a one-dimensional vector.

### Vector derivative

```python
import numpy as np
import specular

f = lambda x: np.array([x, x * x])
print(specular.derivative(f, 2.0))
```

### Gradient

```python
import numpy as np
import specular

f = lambda x: np.sum(x * x)
print(specular.gradient(f, [1.0, 2.0, 3.0]))
```

### Jacobian

```python
import numpy as np
import specular

f = lambda x: np.array([x[0] + 2 * x[1], 3 * x[0] - x[1]])
print(specular.jacobian(f, [1.0, 2.0]))
```

## API reference

::: specular.calculation.scaled_mean
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: true
      show_source: true

::: specular.calculation.derivative
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: true
      show_source: true

::: specular.calculation.gradient
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: true
      show_source: true

::: specular.calculation.jacobian
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: true
      show_source: true
