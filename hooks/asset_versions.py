"""Refresh local CSS and JavaScript URLs whenever their contents change."""

from __future__ import annotations

from hashlib import sha256
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit


def _versioned_url(url: str, files) -> str:
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or parts.path.startswith("/"):
        return url

    asset = files.get_file_from_path(unquote(parts.path))
    if asset is None:
        # Plugins can add stylesheets generated later, outside the source files
        # (for example mkdocstrings' assets/_mkdocstrings.css).
        return url

    version = sha256(asset.content_bytes).hexdigest()[:16]
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key != "v"
    ]
    query.append(("v", version))
    return urlunsplit(parts._replace(query=urlencode(query)))


def on_files(files, *, config):
    """Version configured assets after file collection, before template rendering.

    MkDocs still copies each asset under its original filename. Its URL filter
    retains the query string when calculating links for nested pages. Replacing
    an existing version also keeps repeated development-server builds stable.
    """
    config.extra_css = [_versioned_url(url, files) for url in config.extra_css]
    for index, script in enumerate(config.extra_javascript):
        if isinstance(script, str):
            config.extra_javascript[index] = _versioned_url(script, files)
        else:
            # Preserve options such as type="module", defer, and async.
            script.path = _versioned_url(script.path, files)
    return files
