"""Build-time code block copy buttons for MkDocs pages."""

from __future__ import annotations

import re


_HIGHLIGHT_BLOCK = re.compile(
    r'(<div class="(?=[^"]*\bhighlight\b)[^"]*">)(?!\s*<button\b)'
)


def on_post_page(output: str, *, page, config) -> str:
    """Insert visible copy buttons into rendered code blocks."""
    return _HIGHLIGHT_BLOCK.sub(
        (
            r'\1<button class="code-copy-button" type="button" '
            r'aria-label="Copy code">Copy</button>'
        ),
        output,
    )
