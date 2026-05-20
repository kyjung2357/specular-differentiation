# 1.1. User installation

**Standard Installation (NumPy backend)**

The package is available on PyPI:

```bash
pip install specular-differentiation
```

Check the version:

```python
import specular

print("version: ", specular.__version__)
```

```text
version:  1.3.0
```

**ODE solvers**

```bash
pip install "specular-differentiation[ode]"
```

**Optimization routines**

```bash
pip install "specular-differentiation[optimization]"
```

**Numba backend**

```bash
pip install "specular-differentiation[numba]"
```

If Numba is installed and available, the package may use the Numba-accelerated CPU backend.
Enable it explicitly with `specular.change_backend("cpu_numba")`.

**JAX backend**

By default, the package uses the NumPy backend (CPU).
To enable hardware acceleration, you can install the package with the JAX backend (GPU/TPU).
This adds the following dependencies:

* **[JAX](https://docs.jax.dev/en/latest/index.html)** (`jax`, `jaxlib` >= 0.10):

```bash
pip install "specular-differentiation[jax]"
```

> [!NOTE]
> This feature is experimental for now. See [2.4 Backend](../api/backend.md).

**PyTorch backend**

```bash
pip install "specular-differentiation[torch]"
```

> [!NOTE]
> This feature is experimental for now. See [2.4 Backend](../api/backend.md).

**Developer installation**

To install all dependencies including tests, docs, and examples.
This adds the following dependencies:

* optional extras: `ode`, `optimization`, `numba`, `jax`, and `torch`
* **[SciPy](https://scipy.org/)** (`scipy` >= 1.17)
* **[PyTorch](https://pytorch.org/)** (`torch` >= 2.12)
* **[Pytest](https://docs.pytest.org/en/stable/)** (`pytest` >= 9.0)

```bash
pip install -e ".[dev]"
```
