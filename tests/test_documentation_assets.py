"""Cache freshness contracts for the published documentation's custom assets."""

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from docs.hooks.asset_versions import on_files


def _files(contents):
    return SimpleNamespace(
        get_file_from_path=lambda path: (
            SimpleNamespace(content_bytes=contents[path]) if path in contents else None
        )
    )


def test_asset_changes_refresh_only_the_changed_asset_url():
    contents = {"stylesheets/extra.css": b"old CSS", "javascripts/nav.js": b"stable JS"}
    config = SimpleNamespace(
        extra_css=["stylesheets/extra.css"], extra_javascript=["javascripts/nav.js"]
    )
    files = _files(contents)
    on_files(files, config=config)
    original_css, original_js = config.extra_css[0], config.extra_javascript[0]

    on_files(files, config=config)
    assert config.extra_css == [original_css]
    assert config.extra_javascript == [original_js]

    contents["stylesheets/extra.css"] = b"new CSS"
    on_files(files, config=config)
    assert config.extra_css[0] != original_css
    assert config.extra_javascript == [original_js]
    assert urlsplit(config.extra_css[0]).path == "stylesheets/extra.css"
    assert len(parse_qs(urlsplit(config.extra_css[0]).query)["v"]) == 1


def test_remote_assets_and_script_options_are_preserved():
    external = ["https://cdn.example.org/math.js?v=3", "//cdn.example.org/theme.css"]
    script = SimpleNamespace(path="javascripts/nav.js?mode=full#start", defer=True, type="module")
    config = SimpleNamespace(extra_css=[external[1]], extra_javascript=[external[0], script])
    on_files(_files({"javascripts/nav.js": b"JS"}), config=config)

    assert config.extra_css == [external[1]]
    assert config.extra_javascript[0] == external[0]
    assert script.defer is True
    assert script.type == "module"
    parts = urlsplit(script.path)
    assert parts.path == "javascripts/nav.js"
    assert parts.fragment == "start"
    assert parse_qs(parts.query)["mode"] == ["full"]
    assert "v" in parse_qs(parts.query)


def test_plugin_generated_assets_are_left_to_their_provider():
    config = SimpleNamespace(extra_css=["assets/_mkdocstrings.css"], extra_javascript=[])
    on_files(_files({}), config=config)
    assert config.extra_css == ["assets/_mkdocstrings.css"]
