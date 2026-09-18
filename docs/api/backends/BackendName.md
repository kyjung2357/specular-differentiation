---
title: specular.backends.BackendName
---

# specular.backends.BackendName

The public type alias for supported calculation backend names:

```python
from typing import Literal

type BackendName = Literal["numpy", "numba", "jax"]
```

`Literal` comes from `typing`. `BackendName` describes the allowed string values;
it is not a function or a class constructor. It is also available as
`specular.BackendName`.

## Values

| Value | Meaning |
| :--- | :--- |
| `"numpy"` | Default backend, included with the package. |
| `"numba"` | Optional backend using Numba-compiled callbacks. |
| `"jax"` | Optional backend using JAX arrays and transformations. |

## Notes

The alias includes every supported name, regardless of which optional packages
are installed. Use [`available_backends`](available_backends.md) to probe the
current environment. Runtime name validation takes place when a backend is
selected by [`set_backend`](set_backend.md) or by entering
[`use_backend`](use_backend.md).

## Examples

```python
from specular.backends import BackendName, set_backend

backend: BackendName = "numpy"
set_backend(backend)
```

See the [backend user guide](../../user-guide/backends.md) for installation and
differences between the backends.

[Source code](https://github.com/kyjung2357/specular-differentiation/blob/main/specular/backends/_registry.py).
