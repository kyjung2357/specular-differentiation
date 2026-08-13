import builtins
import importlib
import sys


def test_classical_solver_import_does_not_require_torch_or_scipy(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "scipy"}:
            raise ImportError(f"blocked optional dependency: {name}")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("specular.optimization", None)
    optimization_pkg = sys.modules.get("specular")
    if optimization_pkg is not None:
        monkeypatch.delattr(optimization_pkg, "optimization", raising=False)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("specular.optimization")

    assert hasattr(module, "gradient_descent")
    assert hasattr(module, "adam")
    assert hasattr(module, "bfgs")
