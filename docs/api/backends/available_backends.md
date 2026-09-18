---
title: specular.backends.available_backends
api_reference: true
---

::: specular.backends.available_backends
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Return the supported backends that can be imported in the current environment.

## Parameters

None.

## Returns

**`tuple[BackendName, ...]`** — Available names in the order `"numpy"`, `"numba"`,
`"jax"`, omitting backends whose required optional dependencies are missing.

## Notes

This call probes all three backends and imports their calculation modules when
available. It does not change the selected backend. Successful imports are
cached; ordinary `import specular` does not perform this probe.

A missing required optional dependency excludes that backend from the returned
tuple. Other import errors or invalid backend implementations propagate to the
caller instead of being treated as a missing installation.

NumPy is included with the package. Install the `numba` or `jax` extra to use the
corresponding optional backend; see the
[backend user guide](../../user-guide/backends.md#installation).

This function is also available as `specular.available_backends`.

## Examples

```python
import specular

selected = specular.get_backend()
print(specular.available_backends())
assert specular.get_backend() == selected
```

With both optional backends installed, the printed tuple is
`('numpy', 'numba', 'jax')`. A NumPy-only installation prints `('numpy',)`.
