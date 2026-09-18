"""Place each API page's generated source after its curated reference text."""

import re


_SOURCE = re.compile(r'<details class="mkdocstrings-source">.*?</details>', re.DOTALL)


def on_page_content(html: str, *, page, **kwargs) -> str:
    """Keep mkdocstrings signatures and source in sync with the Python code."""
    if not page.meta.get("api_reference"):
        return html
    sources = _SOURCE.findall(html)
    if len(sources) != 1:
        raise ValueError(f"Expected one API source block in {page.file.src_uri}")
    return _SOURCE.sub("", html) + "\n" + sources[0]
