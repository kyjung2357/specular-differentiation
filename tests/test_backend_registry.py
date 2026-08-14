"""Tests for backend discovery and selection."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest

import specular
from specular.backends import _registry


@pytest.fixture(autouse=True)
def _restore_numpy_backend():
    token = _registry._current_backend.set("numpy")
    try:
        yield
    finally:
        _registry._current_backend.reset(token)


def test_import_does_not_load_optional_backends() -> None:
    code = (
        "import sys; import specular; "
        "assert 'numba' not in sys.modules; "
        "assert 'jax' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_default_backend_is_numpy() -> None:
    assert specular.get_backend() == "numpy"


def test_available_backends_are_supported_and_numpy_is_available() -> None:
    available = specular.available_backends()

    assert available[0] == "numpy"
    assert set(available) <= {"numpy", "numba", "jax"}


def test_available_backends_probes_without_changing_selection() -> None:
    available = specular.available_backends()

    assert specular.get_backend() == "numpy"
    if "numba" in available:
        assert "numba" in sys.modules
    if "jax" in available:
        assert "jax" in sys.modules


def test_unknown_backend_is_rejected_without_changing_selection() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        specular.set_backend("unknown")

    assert specular.get_backend() == "numpy"


def test_non_string_backend_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        specular.set_backend(None)  # type: ignore[arg-type]


def test_context_manager_is_nested_and_restores_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_registry, "_load_backend", lambda name: object())

    with specular.use_backend("numba") as selected:
        assert selected == "numba"
        assert specular.get_backend() == "numba"
        with specular.use_backend("jax"):
            assert specular.get_backend() == "jax"
        assert specular.get_backend() == "numba"

    assert specular.get_backend() == "numpy"


def test_context_manager_restores_selection_after_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_registry, "_load_backend", lambda name: object())

    with pytest.raises(RuntimeError, match="stop"):
        with specular.use_backend("jax"):
            raise RuntimeError("stop")

    assert specular.get_backend() == "numpy"


def test_use_backend_also_acts_as_decorator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_registry, "_load_backend", lambda name: object())

    @specular.use_backend("jax")
    def selected_inside() -> str:
        return specular.get_backend()

    assert selected_inside() == "jax"
    assert selected_inside() == "jax"
    assert specular.get_backend() == "numpy"


def test_use_backend_decorates_async_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_registry, "_load_backend", lambda name: object())

    @specular.use_backend("jax")
    async def selected_inside() -> str:
        await asyncio.sleep(0)
        return specular.get_backend()

    assert asyncio.run(selected_inside()) == "jax"
    assert specular.get_backend() == "numpy"


def test_async_tasks_keep_independent_backend_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_registry, "_load_backend", lambda name: object())

    async def selected_inside(name: str) -> str:
        with specular.use_backend(name):
            await asyncio.sleep(0)
            return specular.get_backend()

    async def run_both() -> tuple[str, str]:
        first, second = await asyncio.gather(
            selected_inside("numba"),
            selected_inside("jax"),
        )
        return first, second

    assert asyncio.run(run_both()) == ("numba", "jax")
    assert specular.get_backend() == "numpy"


def test_use_backend_decorates_generator_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_registry, "_load_backend", lambda name: object())

    @specular.use_backend("numba")
    def selected_inside():
        yield specular.get_backend()

    assert list(selected_inside()) == ["numba"]
    assert specular.get_backend() == "numpy"


def test_preimported_facade_function_observes_later_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from specular import derivative

    class Backend:
        @staticmethod
        def derivative(f, x, h):
            return "selected backend"

    monkeypatch.setattr(_registry, "_load_backend", lambda name: Backend())
    specular.set_backend("jax")

    assert derivative(lambda value: value, 0.0) == "selected backend"


def test_set_backend_checks_availability_before_changing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> object:
        raise ImportError(f"{name} unavailable")

    monkeypatch.setattr(_registry, "_load_backend", unavailable)

    with pytest.raises(ImportError, match="jax unavailable"):
        specular.set_backend("jax")

    assert specular.get_backend() == "numpy"
