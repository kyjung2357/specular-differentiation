# Quick start

## Import Specular Differentiation

Import `specular` and check the installed package version.

```python
--8<-- "examples/quick_start.py:version"
```

### Compute a specular derivative

Define the ReLU function and evaluate its specular derivative at the kink `x = 0`.
The result is $\sqrt{2}-1$, approximately `0.41421356237309503`.

```python
--8<-- "examples/quick_start.py:derivative"
```

## Backends

### Inspect the backends

`get_backend()` returns the backend selected for the current execution
context. `available_backends()` returns the installed backends that can be
selected. NumPy is the default.

```python
--8<-- "examples/quick_start.py:backend-status"
```

### Install an optional backend

NumPy is included in the standard installation. To run the optional Numba
examples below, install the Numba extra:

```bash
pip install "specular-differentiation[numba]"
```

### Select a backend persistently

`set_backend()` changes the selected backend until it is changed again in the
current execution context. A `try`/`finally` block makes the restoration
explicit.

```python
--8<-- "examples/quick_start.py:persistent-backend"
```

### Select a backend temporarily

`use_backend()` limits the selection to a `with` block and restores the
previous backend automatically when the block ends.

```python
--8<-- "examples/quick_start.py:temporary-backend"
```