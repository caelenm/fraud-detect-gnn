"""The pages' files: no external requests, no markup built from values.

The rules for the comparison page apply here too (AGENTS.md): charts are inline
SVG, nothing is loaded from elsewhere, and every value is inserted as text.
The dashboard is one page per section; app.js draws whichever sections a page
holds, so every element it names must exist on at least one of them.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from fraud_detect.dashboard.server import APP_PAGES, REPORT_CSS, STATIC_DIR, STATIC_FILES

PAGES = {path: (STATIC_DIR / name).read_text("utf-8") for path, name in APP_PAGES.items()}
OTHER_PAGES = {
    "/walkthrough": (STATIC_DIR / "walkthrough.html").read_text("utf-8"),
    "/explain": (STATIC_DIR / "explain.html").read_text("utf-8"),
}
ALL_PAGES = {**PAGES, **OTHER_PAGES}
JS = (STATIC_DIR / "app.js").read_text("utf-8")
CSS = (STATIC_DIR / "dashboard.css").read_text("utf-8")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
NAV = re.compile(r'<nav class="pages".*?</nav>', re.S)
SHARED = (
    "header-meta",
    "check-banner",
    "controls",
    "theme-toggle",
    "page-footer",
    "tooltip",
)


@pytest.mark.parametrize(
    "text", [*PAGES.values(), JS, CSS, REPORT_CSS.read_text("utf-8")]
)
def test_no_file_reaches_anything_external(text):
    urls = re.findall(r"(?:https?:)?//[a-z0-9.-]+\.[a-z]{2,}[^\s\"')]*", text, flags=re.I)
    assert [
        u for u in urls if u != SVG_NAMESPACE.removeprefix("http:") and u != SVG_NAMESPACE
    ] == []
    assert "@import" not in text and "url(" not in text


@pytest.mark.parametrize("path", PAGES)
def test_each_page_loads_only_its_own_files(path):
    html = PAGES[path]
    loaded = re.findall(r'(?:src|<link[^>]*href)="([^"]+)"', html)
    assert sorted(loaded) == ["/app.js", "/dashboard.css", "/report.css"]
    linked = re.findall(r'<a [^>]*href="([^"]+)"', html)
    assert linked and all(link in STATIC_FILES for link in linked), linked


@pytest.mark.parametrize("path", PAGES)
def test_each_page_needs_nothing_the_content_security_policy_forbids(path):
    html = PAGES[path]
    assert "<style" not in html
    assert not re.search(r"<script(?![^>]*\bsrc=)", html)
    assert not re.search(r"\son[a-z]+=", html)
    assert not re.search(r"\sstyle=", html)


def test_every_page_has_the_same_page_links_and_marks_itself():
    navs = {}
    for path, html in ALL_PAGES.items():
        found = NAV.findall(html)
        assert len(found) == 1, path
        nav = found[0]
        current = re.findall(r'<a href="([^"]+)"[^>]*aria-current="page"', nav)
        assert current == [path], (path, current)
        # The same links in the same order; only the marker and kept state differ.
        navs[path] = re.findall(r'<a href="([^"]+)"', nav)
    first = next(iter(navs.values()))
    assert sorted(first) == sorted(ALL_PAGES)
    assert all(links == first for links in navs.values()), navs


def test_every_page_app_js_draws_has_its_shared_furniture():
    for path, html in PAGES.items():
        for element in SHARED:
            assert f'id="{element}"' in html, (path, element)


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


def test_every_element_the_script_names_exists_on_some_page():
    named = set(re.findall(r"\$\(\"([a-z-]+)\"\)", JS))
    named |= set(re.findall(r"\bput\(\s*\"([a-z-]+)\"", JS))
    named |= set(re.findall(r"\bsetText\(\s*\"([a-z-]+)\"", JS))
    named |= set(
        re.findall(
            r"\"([a-z]+(?:-[a-z]+)+)\"",
            " ".join(re.findall(r"const scoped = \[[^\]]+\]", JS)),
        )
    )
    ids = set()
    for html in PAGES.values():
        ids |= set(re.findall(r'\bid="([a-z-]+)"', html))
    assert named and named <= ids, named - ids


def test_the_pages_keep_their_honest_furniture():
    overview, classify = PAGES["/"], PAGES["/classify"]
    assert "Read this first" in overview
    assert "Group only" in classify and "Do not upload, commit or host" in classify
    assert "Sanity checks" in PAGES["/checks"]
    # Every page that shows one dataset shows its caveats (Amazon's f19 warning).
    for path in ("/questions", "/methods", "/contrasts"):
        html = PAGES[path]
        assert 'id="dataset-caveats"' in html, path
        assert html.index('id="dataset-caveats"') < html.index("<section"), path
    # A zoomed axis must say so, and the budget filter must exist.
    assert "does not start at 0" in JS
    assert 'id="budget-select"' in PAGES["/methods"]


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
