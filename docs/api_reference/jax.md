# 4. JAX backend

See the [official homepage](https://docs.jax.dev/en/latest/index.html).

## 4.1 For now

* This feature is currently experimental and undergoing verification.
* Benchmarks indicate significant speedups compared to the NumPy backend.
* Full GPU support and optimization are planned but not yet finalized.
* Requirement: Your objective function must use `jax.numpy` instead of standard `numpy` to avoid errors.

## 4.2 Why JAX?

* The JAX is chosen as the primary acceleration backend due to its high compatibility and similar syntax to NumPy. 
* The core calculation logic is planned to be ported to other backends like *PyTorch* or *TensorFlow* to provide native GPU/TPU support and broader ecosystem integration.

## 4.3 Example

For a detailed comparison of the algorithms, please refer to the following scripts:
* **`docs/jax/jax_example.py`**: A basic implementation using the JAX backend.
* **`examples/optimization/2026-Jung/main.py`**: The full experimental setup used in the paper.

> [!NOTE]
> Preliminary tests show that the computation time is **shorter than BFGS**; however, the exact theoretical reasons for this performance gain are still being investigated and have not yet been fully proven.

## Continue reading

### [API Reference](./README.md)

* [1. Calculation](./calculation.md)
* [2. ODE](./ode.md)
* [3. Optimization](./optimization.md)

### [Examples](/examples/README.md)

* [2026-Jung](/examples/ode/2026-Jung/main.ipynb)
* [2024-Jung-Oh](/examples/optimization/2024-Jung-Oh/main.py)
* [2026-Jung](/examples/optimization/2026-Jung/main.py)