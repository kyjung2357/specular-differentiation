from pathlib import Path


DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"


def test_backend_docs_replace_old_jax_namespace_page():
    assert (DOCS_ROOT / "api" / "backend.md").exists()
    assert not (DOCS_ROOT / "api" / "jax.md").exists()


def test_docs_do_not_reference_removed_jax_namespace():
    stale_patterns = [
        "specular.jax",
        "sjax",
        "api/jax.md",
        "jax.md",
        "2.4. JAX",
    ]

    for path in DOCS_ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for pattern in stale_patterns:
            assert pattern not in text, f"{pattern!r} remains in {path}"
