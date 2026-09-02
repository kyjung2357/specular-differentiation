# Backends

NumPy is always the default backend. Numba and JAX are optional and are loaded
only when they are selected or probed with `available_backends()`.

```python
import specular

print(specular.get_backend())
print(specular.available_backends())
```

```text
numpy
('numpy', 'numba', 'jax')
```

The available tuple depends on the optional packages installed in the current
environment. Probing availability does not change the selected backend.

## Persistent selection

`set_backend()` selects `"numpy"`, `"numba"`, or `"jax"` in the current
execution context.

```python
import specular

specular.set_backend("numba")
result = specular.derivative(lambda x: x * x, 2.0)
```

The setting is isolated between asynchronous contexts. A newly created OS
thread starts from the NumPy default.

## Temporary selection

`use_backend()` restores the previous backend when its scope ends, including
when an exception is raised.

```python
import specular

with specular.use_backend("jax"):
    result = specular.derivative(lambda x: x * x, 2.0)

print(specular.get_backend())
# numpy
```

The same object can be used as a decorator.

```python
@specular.use_backend("numba")
def run():
    return specular.gradient(lambda x: (x * x).sum(), [1.0, 2.0])
```

The decorator form supports ordinary synchronous and asynchronous functions.
For a generator or async generator, put a `with use_backend(...)` block inside
the generator body instead.

## Backend behavior

All three backends use the same centered function samples and the same
specular increment kernel. They differ in execution and result representation:

| Backend | Dependency | Result family | Callback requirement |
| :--- | :--- | :--- | :--- |
| NumPy | core | Python scalar or NumPy array | NumPy-compatible callable |
| Numba | `numba` extra | Python scalar or NumPy array | Numba-compilable callable |
| JAX | `jax` extra | JAX array | JAX-transformable callable |

Numba compiles and caches ordinary Python callbacks. As with Numba's
`nopython` mode, referenced global and closure values are captured when that
callback is first compiled; pass changing data through the callback argument
or use a new callback object.

JAX normally uses 32-bit floating-point values unless its 64-bit mode is
enabled before calculations. Double precision is recommended when numerical
agreement with the NumPy and Numba float64 backends is required:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

Under `jax.jit`, `h` must be closed over by the compiled function or marked as
a static argument; a dynamically traced step is rejected before the callback
is traced. XLA may flush subnormal values to zero on some devices, so exact
subnormal parity is not part of the cross-backend contract. In the normal
range, compare results with tolerances appropriate to the selected dtype.

## API reference

::: specular.backends.get_backend
    options:
      show_root_heading: true

::: specular.backends.available_backends
    options:
      show_root_heading: true

::: specular.backends.set_backend
    options:
      show_root_heading: true

::: specular.backends.use_backend
    options:
      show_root_heading: true
