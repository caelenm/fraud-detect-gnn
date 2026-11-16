"""The GAT attention page: test nodes only, weights that add up, and no label of a
test neighbour ever leaves the server.

A fake dataset (tests/helpers.py), a small E10 run on it, then the real
scripts/gat_attention.py. Nothing here is real data. Each failure test breaks
one saved file and asserts that the page refuses the model.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil

import pandas as pd
import pytest
from helpers import fake_dataset

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from fraud_detect import columns as C  # noqa: E402
from fraud_detect.cli import main  # noqa: E402
from fraud_detect.config import REPO_ROOT, get_paths  # noqa: E402
from fraud_detect.dashboard import attention as at  # noqa: E402
from fraud_detect.dashboard import server as srv  # noqa: E402
from fraud_detect.dashboard.server import STATIC_DIR, STATIC_FILES  # noqa: E402

FS = at.FEATURE_SET
SMALL = [
    "--set", "gnn_classify.hidden=[8]", "--set", "gnn_classify.dropout=[0.0]",
    "--set", "gnn_classify.lr=[0.02]", "--set", "gnn_classify.max_epochs=6",
    "--set", "gnn_classify.patience=3", "--set", "gnn_classify.inner_folds=3",
    "--set", "gnn_classify.seeds=1",
]  # fmt: skip
TOP_K = 3  # small, so some nodes have more neighbours than are shown


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A fake dataset with E10 run and its attention read-out; the original
    attention folder is kept so each test can restore it."""
    tmp = tmp_path_factory.mktemp("attention")
    path, config = fake_dataset(tmp, n_nodes=300)
    assert main(["--config", str(path), "--set", f"feature_set={FS}", *SMALL]) == 0
    spec = importlib.util.spec_from_file_location(
        "gat_attention", REPO_ROOT / "scripts" / "gat_attention.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    overrides = [a for pair in zip(SMALL[::2], SMALL[1::2], strict=True)
                 for a in ("--set", pair[1])]  # fmt: skip
    args = ["--dataset", "fake", "--config", str(path), "--n-repeats", "1",
            "--top-k", str(TOP_K), *overrides]  # fmt: skip
    assert script.main(args) == 0
    p = get_paths({**config, "feature_set": FS})
    folder = p.feature_set_dir / at.OUTPUT_DIR
    shutil.copytree(folder, tmp / "attention_original")
    return config, p, folder, tmp / "attention_original"


@pytest.fixture
def env(built, monkeypatch):
    config, p, folder, original = built
    shutil.rmtree(folder)
    shutil.copytree(original, folder)
    monkeypatch.setattr(at, "DATASETS", ("fake",))
    return config, p, folder


def test_a_finished_read_out_loads_with_every_check_passed(env):
    config, _, _ = env
    model = at.load_model(config, "fake")
    assert all(c.ok for c in model.checks)
    assert {c.name for c in model.checks} >= {
        "The mapped nodes are exactly the test nodes",
        "No mapped node was trained on",
        "Attention weights add up",
        "The saved attention holds no labels",
        "The refit matches the run's fold model",
    }
    summary = model.summary()
    json.dumps(srv.clean(summary), allow_nan=False)
    assert summary["top_k"] == TOP_K and summary["heads"] >= 1
    assert {r["layer"] for r in summary["heat"]} == {1, 2}
    for row in summary["heat"]:
        assert 0 < row["focus"] <= 1 + 1e-9
        assert len(row["heads"]) == summary["heads"]


def test_every_card_is_a_test_node_and_its_weights_add_up(env):
    config, p, _ = env
    model = at.load_model(config, "fake")
    test = pd.read_csv(p.test_ids)
    train = set(pd.read_csv(p.train_ids)[C.NODE_ID])
    for seed in range(15):
        card = model.draw("any", seed)
        assert card["node_id"] in set(test[C.NODE_ID]) and card["node_id"] not in train
        label = int(test.set_index(C.NODE_ID).loc[card["node_id"], C.LABEL])
        assert card["label"] == label
        for layer in card["layers"]:
            for rel in layer["relations"]:
                weights = [n["alpha"] for n in rel["neighbours"]]
                assert len(weights) == rel["n_shown"] == min(rel["degree"], TOP_K)
                assert sum(weights) == pytest.approx(rel["shown_weight"], abs=1e-6)
                if 0 < rel["degree"] <= TOP_K:
                    assert rel["shown_weight"] == pytest.approx(1.0, abs=1e-5)
                assert weights == sorted(weights, reverse=True)


def test_only_training_neighbours_carry_a_label(env):
    config, p, _ = env
    model = at.load_model(config, "fake")
    train = pd.read_csv(p.train_ids).set_index(C.NODE_ID)[C.LABEL]
    test = set(pd.read_csv(p.test_ids)[C.NODE_ID])
    seen = set()
    for seed in range(30):
        card = model.draw("any", seed)
        for layer in card["layers"]:
            for rel in layer["relations"]:
                for n in rel["neighbours"]:
                    node, kind = n["node_id"], n["kind"]
                    seen.add(kind)
                    assert set(n) == {"node_id", "rank", "alpha", "heads", "kind"}
                    if node in train.index:
                        expected = at.TRAIN_FRAUD if train[node] else at.TRAIN_BENIGN
                    else:
                        expected = at.TEST if node in test else at.UNLABELLED
                    assert kind == expected
    assert at.TEST in seen  # the fake graph links test nodes too


def test_draws_are_reproducible_and_keep_to_their_kind(env):
    config, _, _ = env
    model = at.load_model(config, "fake")
    assert model.draw("fraud", 7)["node_id"] == model.draw("fraud", 7)["node_id"]
    for draw in ("fraud", "benign"):
        for seed in range(5):
            card = model.draw(draw, seed)
            assert card["label"] == (1 if draw == "fraud" else 0)
    for seed in range(5):
        card = model.draw("any", seed)
        assert card["flagged"] == (card["prob"] >= card["threshold"])


def test_bad_requests_are_refused(env):
    config, _, _ = env
    model = at.load_model(config, "fake")
    for draw, seed in (("everything", 1), ("any", -1), ("any", 1.5), ("any", True)):
        with pytest.raises(at.AttentionRequestError):
            model.draw(draw, seed)


def test_a_training_node_among_the_targets_stops_the_model(env):
    config, p, folder = env
    targets = pd.read_parquet(folder / "attention_targets.parquet")
    first = targets["target"].iloc[0]
    targets.loc[targets["target"] == first, "target"] = pd.read_csv(p.train_ids)[
        C.NODE_ID
    ].iloc[0]
    targets.to_parquet(folder / "attention_targets.parquet", index=False)
    with pytest.raises(at.AttentionUnavailable, match="trained on|test nodes"):
        at.load_model(config, "fake")


def test_weights_that_do_not_add_up_stop_the_model(env):
    config, _, folder = env
    edges = pd.read_parquet(folder / "attention_edges.parquet")
    edges.loc[0, "alpha"] += 0.2
    head_cols = [c for c in edges.columns if c.startswith("alpha_h")]
    edges.loc[0, head_cols] += 0.2
    edges.to_parquet(folder / "attention_edges.parquet", index=False)
    with pytest.raises(at.AttentionUnavailable, match="add up"):
        at.load_model(config, "fake")


def test_a_label_column_in_the_saved_files_stops_the_model(env):
    config, _, folder = env
    targets = pd.read_parquet(folder / "attention_targets.parquet")
    targets[C.LABEL] = 0
    targets.to_parquet(folder / "attention_targets.parquet", index=False)
    with pytest.raises(at.AttentionUnavailable, match="no labels|label-like"):
        at.load_model(config, "fake")


def test_a_failed_refit_check_stops_the_model(env):
    config, _, folder = env
    meta = json.loads((folder / "attention_meta.json").read_text("utf-8"))
    meta["check"]["matches"] = False
    (folder / "attention_meta.json").write_text(json.dumps(meta), "utf-8")
    with pytest.raises(at.AttentionUnavailable, match="refit"):
        at.load_model(config, "fake")


def test_a_missing_read_out_says_what_to_run(env):
    config, _, folder = env
    (folder / "attention_edges.parquet").unlink()
    with pytest.raises(at.AttentionUnavailable, match="gat_attention.py"):
        at.load_model(config, "fake")


def test_the_server_offers_loaded_models_and_refuses_unknown_ones(env, tmp_path):
    config, _, _ = env
    app = srv.App(config, tmp_path / "empty_db", use_model=False)
    app.load_attention()
    models = {m["key"]: m for m in app.api_attention_models()["models"]}
    assert models["fake"]["state"] == srv.READY
    json.dumps(srv.clean(app.api_attention_models()), allow_nan=False)
    card = app.api_attention_node({"model": "fake", "draw": "any", "seed": 3})
    assert card["seed"] == 3
    json.dumps(srv.clean(card), allow_nan=False)
    for body in ({"model": "nope"}, ["fake"], {"model": "fake", "draw": "x"}):
        with pytest.raises(srv.BadRequest):
            app.api_attention_node(body)


# ---- The page's files -----------------------------------------------------
HTML = (STATIC_DIR / "attention.html").read_text("utf-8")
JS = (STATIC_DIR / "attention.js").read_text("utf-8")
CSS = (STATIC_DIR / "attention.css").read_text("utf-8")


@pytest.mark.parametrize("text", [HTML, JS, CSS])
def test_the_page_reaches_nothing_external(text):
    urls = re.findall(r"(?:https?:)?//[a-z0-9.-]+\.[a-z]{2,}[^\s\"')]*", text, flags=re.I)
    assert [u for u in urls if "www.w3.org/2000/svg" not in u] == []
    assert "@import" not in text and "url(" not in text


def test_the_page_loads_only_its_own_files_and_keeps_the_csp():
    sources = re.findall(r'(?:src|href)="([^"]+)"', HTML)
    assert all(s in STATIC_FILES for s in sources), sources
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
    ],
)
def test_the_script_never_builds_markup_from_values(pattern):
    assert not re.search(pattern, JS)


def test_the_script_calls_only_its_own_server_and_names_real_elements():
    fetched = re.findall(r"api\(\s*\"([^\"]+)\"", JS)
    assert fetched and all(path.startswith("/api/attention/") for path in fetched)
    assert JS.count("fetch(") == 1
    named = set(re.findall(r"\$\(\"([a-z-]+)\"\)", JS)) | set(
        re.findall(r"\bput\(\s*\"([a-z-]+)\"", JS)
    )
    named |= set(
        re.findall(
            r"\"([a-z]+(?:-[a-z]+)+)\"",
            " ".join(re.findall(r"for \(const id of \[[^\]]+\]", JS)),
        )
    )
    ids = set(re.findall(r'\bid="([a-z-]+)"', HTML))
    assert named and named <= ids, named - ids


def test_the_page_keeps_its_honest_furniture():
    assert "Group only" in HTML and "Do not upload, commit or host" in HTML
    assert "Read this first" in HTML and "not why it decides" in HTML
    assert "refit" in HTML and "training nodes" in HTML


def test_every_page_links_to_this_one():
    for name in ("index.html", "explain.html", "walkthrough.html", "checks.html"):
        assert 'href="/attention"' in (STATIC_DIR / name).read_text("utf-8")
