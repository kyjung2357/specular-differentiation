"""Backend discovery, selection, and lazy loading."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ContextDecorator
from contextvars import ContextVar, Token
from functools import lru_cache, wraps
from importlib import import_module
from inspect import (
    isasyncgenfunction,
    iscoroutinefunction,
    isgeneratorfunction,
)
from threading import Lock
from typing import Any, Final, Literal, Protocol, TypeVar, cast


type BackendName = Literal["numpy", "numba", "jax"]


_F = TypeVar("_F", bound=Callable[..., Any])


class _CalculationBackend(Protocol):
    """Structural contract implemented by every calculation backend."""

    def _A(self, a: Any, b: Any, c: Any) -> Any: ...

    def _B(self, alpha: Any, beta: Any) -> Any: ...

    def _C(self, alpha: Any, beta: Any) -> Any: ...

    def derivative(self, f: Any, x: Any, h: Any = None) -> Any: ...

    def gradient(self, f: Any, x: Any, h: Any = None) -> Any: ...

    def jacobian(self, f: Any, x: Any, h: Any = None) -> Any: ...


_BACKEND_MODULES: Final[dict[BackendName, str]] = {
    "numpy": "specular.backends.numpy.calculation",
    "numba": "specular.backends.numba.calculation",
    "jax": "specular.backends.jax.calculation",
}

_BACKEND_DEPENDENCIES: Final[dict[BackendName, tuple[str, ...]]] = {
    "numpy": ("numpy",),
    "numba": ("numba",),
    "jax": ("jax", "jaxlib"),
}

_REQUIRED_MEMBERS: Final = (
    "_A",
    "_B",
    "_C",
    "derivative",
    "gradient",
    "jacobian",
)

_current_backend: ContextVar[BackendName] = ContextVar(
    "specular_backend",
    default="numpy",
)


class _BackendUnavailableError(ImportError):
    """A supported backend cannot be imported in this environment."""


def _validate_backend(name: str) -> BackendName:
    """Return a supported backend name or raise a concise error."""

    if not isinstance(name, str):
        raise TypeError("backend name must be a string")
    if name not in _BACKEND_MODULES:
        choices = ", ".join(_BACKEND_MODULES)
        raise ValueError(
            f"unknown backend {name!r}; expected one of: {choices}"
        )
    return cast(BackendName, name)


@lru_cache(maxsize=None)
def _load_backend(name: BackendName) -> _CalculationBackend:
    """Import and validate a backend the first time it is selected."""

    module_name = _BACKEND_MODULES[name]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        dependencies = _BACKEND_DEPENDENCIES[name]
        missing_dependency = exc.name is not None and any(
            exc.name == dependency
            or exc.name.startswith(f"{dependency}.")
            for dependency in dependencies
        )
        if missing_dependency:
            dependency_names = " and ".join(map(repr, dependencies))
            dependency_label = (
                "dependency" if len(dependencies) == 1 else "dependencies"
            )
            raise _BackendUnavailableError(
                f"backend {name!r} requires the optional {dependency_label} "
                f"{dependency_names}; install "
                f"'specular-differentiation[{name}]'"
            ) from exc
        raise

    missing = tuple(
        member
        for member in _REQUIRED_MEMBERS
        if not callable(getattr(module, member, None))
    )
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"backend {name!r} does not implement required members: {names}"
        )
    return cast(_CalculationBackend, module)


def _get_selected_backend() -> _CalculationBackend:
    """Return the backend module selected in the current context."""

    return _load_backend(get_backend())


def get_backend() -> BackendName:
    """Return the backend selected in the current execution context."""

    return _current_backend.get()


def available_backends() -> tuple[BackendName, ...]:
    """Return supported backends that can be imported in this environment.

    Calling this function probes each backend, but it does not change the
    selected backend. Ordinary :mod:`specular` imports remain lazy.
    """

    available: list[BackendName] = []
    for name in _BACKEND_MODULES:
        try:
            _load_backend(name)
        except _BackendUnavailableError:
            continue
        available.append(name)
    return tuple(available)


def set_backend(name: str) -> None:
    """Select a backend for subsequent calls in the current context.

    The backend is imported immediately so a missing optional dependency is
    reported at configuration time instead of at the first calculation.
    """

    selected = _validate_backend(name)
    _load_backend(selected)
    _current_backend.set(selected)


class _BackendContext(ContextDecorator):
    """A backend override usable with ``with`` and as a decorator."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entry_lock = Lock()
        self._token: Token[BackendName] | None = None

    def _recreate_cm(self) -> _BackendContext:
        return type(self)(self._name)

    def __enter__(self) -> BackendName:
        with self._entry_lock:
            if self._token is not None:
                raise RuntimeError("backend context is already entered")
            selected = _validate_backend(self._name)
            _load_backend(selected)
            self._token = _current_backend.set(selected)
            return selected

    def __exit__(self, *exc_info: object) -> None:
        with self._entry_lock:
            if self._token is None:
                raise RuntimeError("backend context was not entered")
            _current_backend.reset(self._token)
            self._token = None

    def __call__(self, func: _F) -> _F:
        if isgeneratorfunction(func) or isasyncgenfunction(func):
            raise TypeError(
                "use_backend does not support decorating generator or "
                "async-generator functions"
            )

        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self._recreate_cm():
                    return await func(*args, **kwargs)

            return cast(_F, async_wrapper)

        return cast(_F, super().__call__(func))


def use_backend(name: str) -> _BackendContext:
    """Temporarily select a backend, restoring the previous choice afterward.

    The returned object is both a context manager and a function decorator.
    It can be reused after its context exits, but cannot be re-entered while
    active. Generator and async-generator functions cannot be decorated.
    """

    return _BackendContext(name)


__all__ = [
    "BackendName",
    "available_backends",
    "get_backend",
    "set_backend",
    "use_backend",
]
