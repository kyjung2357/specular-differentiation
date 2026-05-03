import os
import subprocess

_SUPPORTED_BACKENDS = {"cpu_numpy", "cpu_numba", "cpu_jax", "gpu_jax", "cpu_tensorflow", "gpu_tensorflow", "cpu_pytorch", "gpu_pytorch"}
_AVAILABLE_BACKENDS = {"cpu_numpy"}
_REQUESTED_BACKEND = os.environ.get("SPECULAR_BACKEND")
_CURRENT_BACKEND = _REQUESTED_BACKEND or "cpu_numpy"
_BACKEND_ORDER = ["cpu_numpy", "cpu_numba", "cpu_tensorflow", "gpu_tensorflow", "cpu_pytorch", "gpu_pytorch", "cpu_jax", "gpu_jax"]
_DEFAULT_BACKEND_ORDER = ["cpu_numba", "cpu_numpy"]
_INSTALL_HINTS = {
    "cpu_numba":      "pip install 'specular-differentiation[numba]'",
    "cpu_jax":        "pip install 'specular-differentiation[jax]'",
    "gpu_jax":        "pip install 'specular-differentiation[jax]' (requires CUDA-compatible GPU)",
    "cpu_tensorflow": "pip install tensorflow",
    "gpu_tensorflow": "pip install tensorflow (requires CUDA-compatible GPU)",
    "cpu_pytorch":    "pip install torch",
    "gpu_pytorch":    "pip install torch (requires CUDA-compatible GPU)",
}

if _CURRENT_BACKEND not in _SUPPORTED_BACKENDS:
    raise ValueError(f"Invalid SPECULAR_BACKEND={_CURRENT_BACKEND!r}. Choose from: {', '.join(_BACKEND_ORDER)}")

def _has_numba():
    """Return True if Numba is installed and register the CPU Numba backend when usable."""
    try:
        import numba
        if (os.cpu_count() or 1) > 1:
            _AVAILABLE_BACKENDS.add("cpu_numba")
        return True
    except ImportError:
        return False

def _has_nvidia_gpu():
    """Return True if an NVIDIA GPU is accessible via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def _has_jax():
    """Return True if JAX is installed and register available JAX backends."""
    try:
        import jax
        _AVAILABLE_BACKENDS.add("cpu_jax")
        if _has_nvidia_gpu():
            _AVAILABLE_BACKENDS.add("gpu_jax")
        return True
    except ImportError:
        return False

def _has_tensorflow():
    """Return True if TensorFlow is installed and register available TensorFlow backends."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        _AVAILABLE_BACKENDS.add("cpu_tensorflow")
        if gpus:
            _AVAILABLE_BACKENDS.add("gpu_tensorflow")
        return True
    except ImportError:
        return False

def _has_pytorch():
    """Return True if PyTorch is installed and register available PyTorch backends."""
    try:
        import torch
        _AVAILABLE_BACKENDS.add("cpu_pytorch")
        if torch.cuda.is_available():
            _AVAILABLE_BACKENDS.add("gpu_pytorch")
        return True
    except ImportError:
        return False

def _probe_backend(name):
    """Probe a backend lazily and register it if its dependency is available."""
    if name == "cpu_numpy":
        return True
    if name == "cpu_numba":
        return _has_numba()
    if name in {"cpu_jax", "gpu_jax"}:
        return _has_jax()
    if name in {"cpu_tensorflow", "gpu_tensorflow"}:
        return _has_tensorflow()
    if name in {"cpu_pytorch", "gpu_pytorch"}:
        return _has_pytorch()
    return False

def _choose_default_backend():
    """Choose the fastest available backend when SPECULAR_BACKEND is unset."""
    global _CURRENT_BACKEND

    if _REQUESTED_BACKEND is not None:
        return

    for candidate in _DEFAULT_BACKEND_ORDER:
        if candidate in _AVAILABLE_BACKENDS:
            _CURRENT_BACKEND = candidate
            return

_has_numba()

if _REQUESTED_BACKEND is None:
    _choose_default_backend()
else:
    _probe_backend(_CURRENT_BACKEND)

if _CURRENT_BACKEND not in _AVAILABLE_BACKENDS:
    raise ValueError(f"SPECULAR_BACKEND={_CURRENT_BACKEND!r} is not available on this machine. Available: {', '.join(b for b in _BACKEND_ORDER if b in _AVAILABLE_BACKENDS)}")

def backend_info():
    """Print the supported, available, and current backends.

    Example::

        >>> import specular
        >>> specular.backend_info()
        supported backends: cpu_numpy, cpu_numba, cpu_tensorflow, gpu_tensorflow, cpu_pytorch, gpu_pytorch, cpu_jax, gpu_jax
        available backends: cpu_numpy, cpu_numba
        current backend   : cpu_numpy
    """
    _has_numba()
    _has_tensorflow()
    _has_pytorch()
    _has_jax()

    print(f"supported backends: {', '.join(_BACKEND_ORDER)}")
    print(f"available backends: {', '.join(b for b in _BACKEND_ORDER if b in _AVAILABLE_BACKENDS)}")
    print(f"current backend   : {_CURRENT_BACKEND}")

def _ensure_backend_available(new_backend):
    return new_backend in _AVAILABLE_BACKENDS or _probe_backend(new_backend)

def change_backend(new_backend):
    """Change the active backend for the current session.

    Parameters
    ----------
    new_backend : str
        Name of the backend to switch to. Must be one of the available
        backends on this machine (see ``backend_info()``).

    Raises
    ------
    ValueError
        If ``new_backend`` is not in ``_AVAILABLE_BACKENDS``.

    Example::

        >>> import specular
        >>> specular.change_backend("cpu_pytorch")
    """
    global _CURRENT_BACKEND
    
    if new_backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"{new_backend!r} is not a valid backend. "
            f"Choose from: {', '.join(_BACKEND_ORDER)}"
        )

    if _ensure_backend_available(new_backend):
        _CURRENT_BACKEND = new_backend
        return

    hint = _INSTALL_HINTS.get(new_backend, "")
    raise ValueError(
        f"{new_backend!r} is supported but not available on this machine. "
        + (f"Run: {hint}" if hint else "")
    )
