# 1.3. Backend Usage

To use the **JAX** backend, install the JAX extra and select the backend explicitly:

```python
import jax.numpy as jnp
import specular

specular.change_backend("cpu_jax")

ReLU = lambda x: jnp.maximum(x, 0)
specular.derivative(ReLU, 0.0)
```

```text
Array(0.41421354, dtype=float32)
```

To enable 64-bit precision (double precision), update the **JAX** configuration as follows:

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import specular

specular.change_backend("cpu_jax")

ReLU = lambda x: jnp.maximum(x, 0)
specular.derivative(ReLU, 0.0)
```

```text
Array(0.41421356, dtype=float64)
```

The JAX backend is not a bitwise-equivalent implementation of the NumPy backend:
it uses automatic differentiation at shifted points, while NumPy/Numba use
one-sided finite differences. This distinction is intentional.
