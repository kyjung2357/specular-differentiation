# specular.optimization

Composable direction rules, step size rules, and an iteration wrapper for
scalar or vector optimization. SPEG combines the normalized negative
specular gradient with a step size rule.

| Function or class | Description |
| :--- | :--- |
| [`specular_gradient`](specular_gradient.md) | Run SPEG with a selected step size rule. |
| [`minimize`](minimize.md) | Combine a direction and a step size rule in one solver. |
| [`make_direction`](make_direction.md) | Create a SPEG or custom direction callable. |
| [`make_step_size`](make_step_size.md) | Create a schedule or a bounded line search. |
| [`OptimizationResult`](OptimizationResult.md) | Inspect the accepted solution, history, and stopping reason. |
| [`LineSearchError`](LineSearchError.md) | Identify a search that could not accept a step. |

For worked examples, step size formulas, and the distinction between
classical and specular search derivatives, see the
[optimization user guide](../../user-guide/optimization.md).

The optimizer uses the selected
[calculation backend](../backends/index.md) for numerical specular
differentiation, with an eager Python loop and NumPy state.
