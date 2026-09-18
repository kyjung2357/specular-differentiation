# specular.backends

Select the calculation backend for the current execution context. NumPy is the
default; Numba and JAX are optional. All entries below are also available directly
from `specular`.

| Name | Description |
| :--- | :--- |
| [`get_backend`](get_backend.md) | Return the currently selected backend name. |
| [`available_backends`](available_backends.md) | Probe which supported backends can be imported. |
| [`set_backend`](set_backend.md) | Select a backend for subsequent calculations. |
| [`use_backend`](use_backend.md) | Temporarily select a backend in a scope or decorated function. |
| [`BackendName`](BackendName.md) | Type alias for the three supported backend names. |

See the [backend user guide](../../user-guide/backends.md) for installation,
backend selection, callback requirements, and numerical behavior.
