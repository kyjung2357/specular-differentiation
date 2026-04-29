import os
import subprocess

_SUPPORTED_BACKENDS = {"cpu_numpy", "cpu_numba", "cpu_jax", "gpu_jax"}
_AVAILABLE_BACKENDS = {"cpu_numpy"}
_CURRENT_BACKEND = os.environ.get("SPECULAR_BACKEND", "cpu_numpy")
_BACKEND_ORDER = ["cpu_numpy", "cpu_numba", "cpu_jax", "gpu_jax"]

if _CURRENT_BACKEND not in _SUPPORTED_BACKENDS:
    raise ValueError(f"Invalid SPECULAR_BACKEND={_CURRENT_BACKEND!r}. Choose from {', '.join(sorted(_SUPPORTED_BACKENDS))}")

def _has_numba():
    """Return True if numba is installed."""
    global _CURRENT_BACKEND
    try:
        import numba
        if (os.cpu_count() or 1) > 1:
            _AVAILABLE_BACKENDS.add("cpu_numba")
            if _CURRENT_BACKEND == "cpu_numpy":
                _CURRENT_BACKEND = "cpu_numba"
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
    try:
        import jax
        _AVAILABLE_BACKENDS.add("cpu_jax")
        if _has_nvidia_gpu():
            _AVAILABLE_BACKENDS.add("gpu_jax")
        return True
    except ImportError:
        return False
              
def _detect_available_backends():
    """Populate _AVAILABLE_BACKENDS based on hardware and installed packages."""
    _has_numba()
    _has_jax()

_detect_available_backends()

if _CURRENT_BACKEND not in _AVAILABLE_BACKENDS:
    raise ValueError(f"SPECULAR_BACKEND={_CURRENT_BACKEND!r} is not available on this machine. Available: {', '.join(sorted(_AVAILABLE_BACKENDS))}")

def backend_info():
    """Print the supported, available, and current backends.

    Example::

        >>> import specular
        >>> specular.backend_info()
        supported backends: cpu_numpy, cpu_numba, cpu_jax, gpu_jax
        available backends: cpu_numpy, cpu_numba, cpu_jax
        current backend   : cpu_numpy
    """
    print(f"supported backends: {', '.join(_BACKEND_ORDER)}")
    print(f"available backends: {', '.join(b for b in _BACKEND_ORDER if b in _AVAILABLE_BACKENDS)}")
    print(f"current backend   : {_CURRENT_BACKEND}")

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
        >>> specular.change_backend("cpu_jax")
    """
    global _CURRENT_BACKEND
    if new_backend in _AVAILABLE_BACKENDS:
        _CURRENT_BACKEND = new_backend
    else:
        raise ValueError(f"{new_backend!r} is not available. Available: {', '.join(sorted(_AVAILABLE_BACKENDS))}")
