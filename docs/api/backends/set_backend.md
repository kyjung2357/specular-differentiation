---
title: specular.backends.set_backend
api_reference: true
---

::: specular.backends.set_backend
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Select a backend for subsequent calculations in the current execution context.

## Parameters

| Name | Type | Description |
| :--- | :--- | :--- |
| `name` | `str` | One of `"numpy"`, `"numba"`, or `"jax"`. Names are case-sensitive. |

## Returns

**`None`** — The selected backend is stored in the current context.

## Notes

The backend is imported and validated before the selection changes. A missing
optional dependency is reported immediately as an `ImportError`, leaving the
previous selection in place. Install the corresponding extra before selecting
Numba or JAX; see the [backend user guide](../../user-guide/backends.md#installation).

A non-string `name` raises `TypeError`; an unsupported string raises `ValueError`.
Other errors from backend loading propagate to the caller.

The selection persists in the current context until changed again. Changes in
an asynchronous context do not overwrite the selection in other contexts.
A fresh execution context starts with NumPy; a thread that explicitly inherits
a context inherits its selection. For a temporary override that restores the previous
selection, use [`use_backend`](use_backend.md).

This function is also available as `specular.set_backend`.

## Examples

```python
import specular

specular.set_backend("numpy")
value = specular.derivative(lambda x: x * x, 2.0)
print(specular.get_backend(), value)
```
