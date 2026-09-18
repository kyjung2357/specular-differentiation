---
title: specular.backends.use_backend
api_reference: true
---

::: specular.backends.use_backend
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Temporarily select a backend, restoring the previous selection when the scope
ends, including when an exception is raised.

## Parameters

| Name | Type | Description |
| :--- | :--- | :--- |
| `name` | `str` | One of `"numpy"`, `"numba"`, or `"jax"`. Names are case-sensitive. |

## Returns

**Context manager and function decorator** — Entering the context selects the
requested backend. In `with use_backend(name) as selected:`, `selected` is the
validated [`BackendName`](BackendName.md). Exiting restores the previous backend.

## Notes

Validation and backend loading occur when entering the context or calling a
decorated function, rather than when constructing the context manager. Invalid
names and missing dependencies raise the same errors as
[`set_backend`](set_backend.md). Optional backends must be
[installed first](../../user-guide/backends.md#installation).

An object returned by `use_backend` can be reused after its context exits, but
re-entering the same object while it is active raises `RuntimeError`. Nested
overrides are supported by creating a separate context manager for each scope.

The decorator supports ordinary synchronous and asynchronous functions.
Decorating a generator or async-generator function raises `TypeError`. In a
generator, put a `with` block inside the body and leave it before yielding;
otherwise the caller also observes the temporary backend while the generator
is suspended.

This function is also available as `specular.use_backend`.

## Examples

```python
import specular

previous = specular.get_backend()
with specular.use_backend("numpy") as selected:
    print(selected)  # numpy
    value = specular.derivative(lambda x: x * x, 2.0)
assert specular.get_backend() == previous
```

The decorator applies the temporary selection to each invocation:

```python
@specular.use_backend("numpy")
def square_derivative(x):
    return specular.derivative(lambda t: t * t, x)

print(square_derivative(2.0))
```

See the [backend user guide](../../user-guide/backends.md) for optional backend
examples and callback requirements.
