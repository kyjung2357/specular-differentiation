"""Keep the README's documentation links inside the rendered documentation site."""

from mkdocs.utils import get_relative_url


# README links must also work on GitHub. Match their published origin explicitly:
# mkdocs serve replaces config.site_url with the local preview address.
_PUBLISHED_URL = "https://kyjung2357.github.io/specular-differentiation/"
_PAGES = {
    "quick-start/": "quick-start.md",
    "api/": "api/index.md",
    "examples/": "examples/index.md",
    "started/latex-macro/": "started/latex-macro.md",
}


def on_page_content(html: str, *, page, config, files) -> str:
    """Localize homepage section links without changing the README itself."""
    if page.file.src_uri != "index.md":
        return html
    for route, source in _PAGES.items():
        target = files.get_file_from_path(source)
        if target is None:
            raise ValueError(f"Documentation section page is missing: {source}")
        relative = get_relative_url(target.url, page.url)
        html = html.replace(
            f'href="{_PUBLISHED_URL}{route}"', f'href="{relative}"'
        )
    return html
