---
title: specular.optimization.LineSearchError
api_reference: true
---

::: specular.optimization.LineSearchError
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false
      merge_init_into_class: false

Exception raised when a line search cannot accept a finite, positive step.
It is a subclass of `RuntimeError`.

## Notes

Searches created with [`make_step_size`](make_step_size.md) raise this
exception for failures such as a non-descent initial slope, an exhausted
budget, a search bound reached without acceptable curvature, or stagnation
at floating-point precision. A failed search does not return an unchecked
fallback step.

[`minimize`](minimize.md) and [`specular_gradient`](specular_gradient.md)
catch this exception from a step rule and return an
[`OptimizationResult`](OptimizationResult.md) with `success=False` and a
`stop_reason` beginning with `"line search failed:"`. The last accepted
point remains intact. Invalid arguments or callback shape errors use their
own exceptions and are not treated as a failed line search.

## Examples

```python
from specular.optimization import LineSearchError, make_step_size

search = make_step_size(
    "armijo",
    f=lambda x: x * x,
    gradient=lambda x: 2.0 * x,
)

try:
    search(1, x=1.0, d=1.0)
except LineSearchError as error:
    print(error)
# Line search requires a descent direction (negative raw slope)
```
