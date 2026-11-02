"""The page's files: no external requests, no markup built from values.

The rules for the comparison page apply here too (AGENTS.md): charts are inline
SVG, nothing is loaded from elsewhere, and every value is inserted as text.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from fraud_detect.dashboard.server import REPORT_CSS, STATIC_DIR, STATIC_FILES

HTML = (STATIC_DIR / "index.html").read_text("utf-8")
JS = (STATIC_DIR / "app.js").read_text("utf-8")
CSS = (STATIC_DIR / "dashboard.css").read_text("utf-8")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"


@pytest.mark.parametrize("text", [HTML, JS, CSS, REPORT_CSS.read_text("utf-8")])
def test_no_file_reaches_anything_external(text):
    urls = re.findall(r"(?:https?:)?//[a-z0-9.-]+\.[a-z]{2,}[^\s\"')]*", text, flags=re.I)
    assert [
        u for u in urls if u != SVG_NAMESPACE.removeprefix("http:") and u != SVG_NAMESPACE
    ] == []
    assert "@import" not in text and "url(" not in text


def test_the_html_loads_only_its_own_files():
    sources = re.findall(r'(?:src|href)="([^"]+)"', HTML)
    assert sorted(sources) == ["/app.js", "/dashboard.css", "/report.css"]
    assert all(s in STATIC_FILES for s in sources)


def test_the_html_needs_nothing_the_content_security_policy_forbids():
    assert "<style" not in HTML
    assert not re.search(r"<script(?![^>]*\bsrc=)", HTML)
    assert not re.search(r"\son[a-z]+=", HTML)
    assert not re.search(r"\sstyle=", HTML)


@pytest.mark.parametrize(
    "pattern",
    [
        r"\binnerHTML\b",
        r"\bouterHTML\b",
        r"\binsertAdjacentHTML\b",
        r"\bdocument\.write\b",
        r"\beval\(",
        r"\bnew Function\b",
        r"setAttribute\(\s*[\"']style",
    ],
)
def test_the_script_never_builds_markup_or_code_from_values(pattern):
    assert not re.search(pattern, JS)


def test_the_script_only_calls_its_own_server():
    fetched = re.findall(r"api\(\s*\"([^\"]+)\"", JS)
    assert fetched and all(path.startswith("/api/") for path in fetched)
    assert "fetch(path" in JS and JS.count("fetch(") == 1


def test_every_element_the_script_names_exists():
    named = set(re.findall(r"\$\(\"([a-z-]+)\"\)", JS))
    named |= set(re.findall(r"\bput\(\s*\"([a-z-]+)\"", JS))
    named |= set(
        re.findall(
            r"\"([a-z]+(?:-[a-z]+)+)\"",
            " ".join(re.findall(r"const scoped = \[[^\]]+\]", JS)),
        )
    )
    ids = set(re.findall(r'\bid="([a-z-]+)"', HTML))
    assert named and named <= ids, named - ids


def test_the_page_keeps_its_honest_furniture():
    assert "Group only" in HTML and "Do not upload, commit or host" in HTML
    assert "Read this first" in HTML
    assert HTML.index('id="read-first"') < HTML.index('id="methods"')
    assert "Sanity checks" in HTML
    # A zoomed axis must say so, and the budget filter must exist.
    assert "does not start at 0" in JS
    assert 'id="budget-select"' in HTML


def test_dark_mode_is_chosen_not_flipped():
    assert "prefers-color-scheme: dark" in CSS and ':root[data-theme="dark"]' in CSS
    for colour in ("#3987e5", "#d95926", "#199e70"):
        assert CSS.count(colour) == 2  # both the media query and the toggle scope


def test_every_verdict_kind_has_an_icon_and_a_style():
    from fraud_detect.dashboard import data  # noqa: PLC0415

    for kind in data.VERDICT_LABELS:
        assert f"{kind}:" in JS, kind
    for kind in (data.SUPPORTED, data.WORSE_SIGNIFICANT, data.WORSE_MEAN):
        assert f".verdict.{kind}" in CSS, kind


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_script_parses():
    done = subprocess.run(
        ["node", "--check", str(STATIC_DIR / "app.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
