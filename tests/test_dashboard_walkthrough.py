"""The "explain a node" walkthrough page: the dashboard's page rules, plus its glossary.

The page reads the same two API calls as /explain, so the data behind it is
tested in test_dashboard_explain.py; these tests cover the page's own files.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from fraud_detect.dashboard.server import STATIC_DIR, STATIC_FILES

HTML = (STATIC_DIR / "walkthrough.html").read_text("utf-8")
JS = (STATIC_DIR / "walkthrough.js").read_text("utf-8")
CSS = (STATIC_DIR / "walkthrough.css").read_text("utf-8")


@pytest.mark.parametrize("text", [HTML, JS, CSS])
def test_the_page_reaches_nothing_external(text):
    urls = re.findall(r"(?:https?:)?//[a-z0-9.-]+\.[a-z]{2,}[^\s\"')]*", text, flags=re.I)
    assert [u for u in urls if "www.w3.org/2000/svg" not in u] == []
    assert "@import" not in text and "url(" not in text


def test_the_page_loads_only_its_own_files_and_keeps_the_csp():
    loaded = re.findall(r'(?:src|<link[^>]*href)="([^"]+)"', HTML)
    assert sorted(loaded) == [
        "/dashboard.css",
        "/report.css",
        "/walkthrough.css",
        "/walkthrough.js",
    ]
    links = re.findall(r'<a [^>]*href="([^"]+)"', HTML)
    assert all(link in STATIC_FILES or link.startswith("#src-") for link in links), links
    assert "<style" not in HTML and not re.search(r"<script(?![^>]*\bsrc=)", HTML)
    assert not re.search(r"\son[a-z]+=", HTML) and not re.search(r"\sstyle=", HTML)


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
        r"\.style\.",
    ],
)
def test_the_script_never_builds_markup_from_values(pattern):
    assert not re.search(pattern, JS)


def test_the_script_calls_only_the_explain_api_and_names_real_elements():
    fetched = re.findall(r"api\(\s*\"([^\"]+)\"", JS)
    assert sorted(set(fetched)) == ["/api/explain/models", "/api/explain/node"]
    assert JS.count("fetch(") == 1
    named = set(re.findall(r"\$\(\"([a-z-]+)\"\)", JS))
    named |= set(re.findall(r"\bput\(\s*\"([a-z-]+)\"", JS))
    for listed in re.findall(r"for \(const id of \[([^\]]+)\]", JS):
        named |= set(re.findall(r"\"([a-z-]+)\"", listed))
    ids = set(re.findall(r'\bid="([a-z-]+)"', HTML))
    assert named and named <= ids, named - ids


def test_the_page_keeps_its_honest_furniture():
    assert "Group only" in HTML and "Do not upload, commit or host" in HTML
    assert "not the cause of fraud" in HTML
    assert "f19 is a probable label leak" in HTML
    assert "faithful by construction" not in HTML + JS
    assert "the important nodes" not in JS
    # Amazon's default is all 25 features, with the f19 note beside it.
    assert 'm.variant === "all"' in JS and "probable label leak" in JS


def sources() -> list[str]:
    block = re.search(r"const SOURCES = \[(.*?)\n  \];", JS, re.S).group(1)
    return re.findall(r'^    "(.*)",$', block, re.M)


def glossary() -> dict[str, list[int]]:
    block = re.search(r"const GLOSSARY = \{(.*?)\n  \};", JS, re.S).group(1)
    src = re.search(r"const SRC = \{([^}]*)\}", JS).group(1)
    names = dict(re.findall(r"(\w+): (\d+)", src))
    out = {}
    for key, refs in re.findall(r'^    "?([\w:]+)"?: \[.*\[([^\]]*)\]\],$', block, re.M):
        out[key] = [int(names[r.strip().removeprefix("SRC.")]) for r in refs.split(",")]
    return out


def test_every_glossary_card_cites_a_listed_source():
    n = len(sources())
    cards = glossary()
    assert n == 12 and len(cards) >= 20
    for key, refs in cards.items():
        assert refs and all(1 <= r <= n for r in refs), key
    # The page's own citation in step 6 points at the analyst studies.
    assert 'href="#src-12"' in HTML and "Jesus et al." in sources()[11]


def test_both_datasets_relations_and_labels_are_explained():
    cards = glossary()
    for key in (
        "amazon:UPU",
        "amazon:USU",
        "amazon:UVU",
        "amazon:f19",
        "amazon:feat",
        "amazon:label",
        "yelpchi:RUR",
        "yelpchi:RSR",
        "yelpchi:RTR",
        "yelpchi:feat",
        "yelpchi:label",
        "nbr",
        "deg",
        "gnnssl",
        "shap",
        "group",
        "influence",
        "threshold",
        "base",
    ):
        assert key in cards, key


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_script_parses():
    done = subprocess.run(
        ["node", "--check", str(STATIC_DIR / "walkthrough.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
