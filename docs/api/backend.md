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
`BackendName` is the public type alias for the three accepted names:
`Literal["numpy", "numba", "jax"]`.

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
the generator body and exit it before yielding. A generator suspends the
context at `yield`, so yielding inside the block would also leave the caller
using the temporary backend until the generator resumes or closes.

```python
def derivatives(points):
    for point in points:
        with specular.use_backend("numba"):
            result = specular.derivative(lambda x: x * x, point)
        yield result
```

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

The JAX angular-mean kernels supply analytic differentiation rules for finite
real inputs, including equal and opposite slopes. `jax.grad` and `jax.jvp`
therefore differentiate the smooth mean through these cases. For example:

```python
with specular.use_backend("jax"):
    slope = jax.grad(lambda alpha: specular.scaled_mean(alpha, 1.0))(1.0)
    print(slope)  # 0.5
```

Differentiating a call to `derivative`, `gradient`, or `jacobian` differentiates
the finite-difference approximation and its callback; it does not turn the
approximation into an exact higher derivative of the original function.
Automatic differentiation requires differentiable callbacks and representable
intermediate derivatives. No derivative is promised for invalid inputs or
extended-real infinity cases.

Coordinate sampling uses bounded batches rather than a dense `n` by `n`
identity matrix, which limits temporary storage for large gradients. A
Jacobian still requires storage for its returned `m` by `n` matrix.

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
