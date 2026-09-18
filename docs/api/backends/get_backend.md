---
title: specular.backends.get_backend
api_reference: true
---

::: specular.backends.get_backend
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Return the backend selected in the current execution context.

## Parameters

None.

## Returns

**`BackendName`** — `"numpy"`, `"numba"`, or `"jax"`. The default is `"numpy"`.

## Notes

Reading the selection does not import the selected backend or probe optional
dependencies. Selection is local to the current context; asynchronous contexts
can change it independently. A fresh execution context starts with NumPy;
a thread that explicitly inherits a context inherits that context's selection.

This function is also available as `specular.get_backend`.

## Examples

```python
import specular

specular.set_backend("numpy")
print(specular.get_backend())  # numpy
```

See also [`set_backend`](set_backend.md), [`use_backend`](use_backend.md), and
the [backend user guide](../../user-guide/backends.md).
