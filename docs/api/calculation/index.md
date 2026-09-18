# specular.calculation

Backend-aware specular differentiation and scaled angular means.

| Function | Description |
| :--- | :--- |
| [`scaled_mean`](scaled_mean.md) | Evaluate the scaled angular mean elementwise. |
| [`derivative`](derivative.md) | Differentiate a scalar-input, scalar- or vector-valued function. |
| [`gradient`](gradient.md) | Compute the specular gradient of a scalar-valued function. |
| [`jacobian`](jacobian.md) | Compute the specular Jacobian of a vector-valued function. |

These functions are also available directly as `specular.scaled_mean`,
`specular.derivative`, `specular.gradient`, and `specular.jacobian`.

See [Calculation in the User Guide](../../user-guide/calculation.md) for
usage and finite-difference step selection, and
[`specular.backends`](../backends/index.md) for backend selection.
