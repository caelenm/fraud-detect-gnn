"""The explanations page: only test nodes, and every card adds up before it is shown.

A fake dataset (tests/helpers.py) through df_analyze_input, then an invented
explain folder with SHAP values that sum to an invented margin. Nothing here is
real data or a real model. Each failure test breaks one file and asserts that
the page refuses.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
from helpers import FAKE_RUN, fake_dataset

from fraud_detect import columns as C
from fraud_detect.cli import main
from fraud_detect.config import get_paths
from fraud_detect.dashboard import explain as ex
from fraud_detect.dashboard import server as srv
from fraud_detect.dashboard.server import STATIC_DIR, STATIC_FILES

FS = "pilot_m2_own_nbr"
BASE = -1.5
THRESHOLD = 0.4


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def write_explain(p, *, train_in_test: bool = False, influence: bool = True) -> list[str]:
    """An invented explain folder for the feature set in `p`. Returns the columns."""
    info = json.loads((p.df_analyze_input_dir / "feature_set.json").read_text("utf-8"))
    columns = list(info["columns"])
    test = pd.read_csv(p.test_ids)
    train = pd.read_csv(p.train_ids)
    rng = np.random.default_rng(0)
    n = len(test)
    shap = rng.normal(0, 0.3, size=(n, len(columns)))
    margin = BASE + shap.sum(axis=1)
    ids = test[C.NODE_ID].to_numpy()
    if train_in_test:
        ids = ids.copy()
        ids[0] = train[C.NODE_ID].iloc[0]
    frame = pd.DataFrame(
        {
            C.NODE_ID: ids,
            C.GROUP_ID: np.arange(n),
            C.LABEL: test[C.LABEL].to_numpy(),
            "prob": _sigmoid(margin),
            "margin": margin,
        }
    )
    for j, c in enumerate(columns):
        frame[ex.SHAP_PREFIX + c] = shap[:, j]
    folder = p.feature_set_dir / "explain"
    (folder / "tables").mkdir(parents=True)
    frame.to_parquet(folder / "shap_test.parquet", index=False)
    m = min(40, len(train))
    tshap = rng.normal(0, 0.3, size=(m, len(columns)))
    tframe = pd.DataFrame(
        {
            C.NODE_ID: train[C.NODE_ID].to_numpy()[:m],
            C.LABEL: train[C.LABEL].to_numpy()[:m],
            "prob": _sigmoid(BASE + tshap.sum(axis=1)),
        }
    )
    for j, c in enumerate(columns):
        tframe[ex.SHAP_PREFIX + c] = tshap[:, j]
    tframe.to_parquet(folder / "shap_train_sample.parquet", index=False)
    importance = pd.DataFrame(
        {
            "name": columns,
            "mean_abs_shap": np.abs(shap).mean(axis=0),
            "low": 0.0,
            "high": 1.0,
            "mean_signed_shap": shap.mean(axis=0),
            "rank": np.arange(1, len(columns) + 1),
            "rank_low": 1,
            "rank_high": len(columns),
            "unstable": False,
            "n_rows": n,
        }
    )
    importance.to_csv(folder / "tables" / "importance.csv", index=False)
    meta = {
        "dataset": "fake",
        "repeat": 0,
        "feature_set": p.feature_set_dir.name,
        "model_cls": "LGBMClassifier",
        "selection": "none",
        "threshold": THRESHOLD,
        "df_analyze_run": FAKE_RUN,
        "sections": {
            "shap": {
                "base_value": BASE,
                "feature_perturbation": "interventional",
                "model_output": "raw (log-odds)",
                "n_background": 10,
                "background": "stratified sample of training rows",
            }
        },
    }
    (folder / "explain_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if influence:
        target = int(test[C.NODE_ID].iloc[0])
        pd.DataFrame(
            {
                "target": np.int32(
                    [target, target, target, int(test[C.NODE_ID].iloc[1])]
                ),
                "neighbour": np.int32([7, 8, 9, 7]),
                "relation": ["aaa", "aaa", "bbb", "aaa"],
                "degree": np.int32([2, 2, 1, 1]),
                "delta_margin": np.float32([0.1, -0.5, 0.2, 0.9]),
                "delta_prob": np.float32([0.01, -0.05, 0.02, 0.09]),
                "delta_prob_mean": np.float32([0.01, -0.04, 0.02, 0.08]),
                "out_of_range": [False, True, False, False],
            }
        ).to_parquet(folder / "neighbour_influence.parquet", index=False)
    export = p.df_analyze_output_dir / FAKE_RUN / "train" / "hash" / "results" / "test00"
    export.mkdir(parents=True)
    pd.DataFrame(rng.normal(size=(n, len(columns))), columns=columns).to_csv(
        export / "X_test_00.csv", index=False
    )
    pd.Series(test[C.LABEL].to_numpy(), name="target").to_csv(export / "y_test_00.csv")
    return columns


@pytest.fixture
def env(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=200)
    assert (
        main(
            [
                "--config",
                str(path),
                "--set",
                f"feature_set={FS}",
                "--to",
                "df_analyze_input",
            ]
        )
        == 0
    )
    p = get_paths({**config, "feature_set": FS})
    return config, p


def load(config):
    return ex.load_model(config, "fake", "all", (FS,))


def test_a_finished_explanation_loads_with_every_check_passed(env):
    config, p = env
    columns = write_explain(p)
    model = load(config)
    assert all(c.ok for c in model.checks)
    assert {c.name for c in model.checks} >= {
        "The SHAP rows are exactly the test nodes",
        "No explained test node was trained on",
        "Base value plus SHAP gives each margin (implementation check)",
        "Input values line up with the SHAP rows",
        "Neighbour influence is for test nodes only",
    }
    summary = model.summary()
    json.dumps(srv.clean(summary), allow_nan=False)
    assert summary["n_columns"] == len(columns) and summary["has_influence"]
    assert len(summary["importance"]) == min(ex.N_TOP, len(columns))


def test_every_card_is_a_test_node_that_adds_up(env):
    config, p = env
    write_explain(p)
    model = load(config)
    test = set(pd.read_csv(p.test_ids)[C.NODE_ID])
    for seed in range(20):
        card = model.draw("any", seed)
        assert card["node_id"] in test
        shown = sum(c["shap"] for c in card["contributions"]) + card["rest"]
        assert card["base_value"] + shown == pytest.approx(card["margin"], abs=1e-9)
        assert len(card["contributions"]) <= ex.N_WATERFALL
        assert all(n["label"] in (0, 1) for n in card["nearest_training"])
    json.dumps(srv.clean(card), allow_nan=False)


def test_draws_are_reproducible_and_keep_to_their_kind(env):
    config, p = env
    write_explain(p)
    model = load(config)
    assert model.draw("any", 5)["node_id"] == model.draw("any", 5)["node_id"]
    for seed in range(10):
        assert model.draw("fraud", seed)["label"] == 1
        assert model.draw("benign", seed)["label"] == 0
        fp = model.draw("false_positive", seed)
        assert (
            fp["label"] == 0
            and fp["prob"] >= THRESHOLD
            and fp["outcome"] == "false_positive"
        )
        fn = model.draw("false_negative", seed)
        assert fn["label"] == 1 and fn["prob"] < THRESHOLD


def test_bad_requests_are_refused(env):
    config, p = env
    write_explain(p)
    model = load(config)
    with pytest.raises(ex.ExplainRequestError, match="Unknown draw"):
        model.draw("everything", 1)
    for seed in (-1, 2**40, "1", True, 1.5):
        with pytest.raises(ex.ExplainRequestError, match="seed"):
            model.draw("any", seed)


def test_neighbours_are_this_nodes_own_sorted_by_size(env):
    config, p = env
    write_explain(p)
    model = load(config)
    first = int(model.rows[C.NODE_ID].iloc[0])
    card = model.card(0)
    assert card["node_id"] == first
    aaa = next(r for r in card["neighbours"]["relations"] if r["relation"] == "aaa")
    assert [n["neighbour"] for n in aaa["top"]] == [8, 7]  # |-0.05| before |0.01|
    assert aaa["top"][0]["out_of_range"] is True
    assert aaa["top"][0]["degree_part"] == pytest.approx(-0.01, abs=1e-6)
    # Node 7's 0.09 belongs to the other target, never to this card.
    assert all(
        n["delta_prob"] != pytest.approx(0.09)
        for r in card["neighbours"]["relations"]
        for n in r["top"]
    )


def test_a_training_node_among_the_shap_rows_stops_the_model(env):
    config, p = env
    write_explain(p, train_in_test=True)
    with pytest.raises(ex.ExplainUnavailable, match="test nodes"):
        load(config)


def test_shap_that_does_not_add_up_stops_the_model(env):
    config, p = env
    columns = write_explain(p)
    path = p.feature_set_dir / "explain" / "shap_test.parquet"
    frame = pd.read_parquet(path)
    frame.loc[3, ex.SHAP_PREFIX + columns[0]] += 0.5
    frame.to_parquet(path, index=False)
    with pytest.raises(ex.ExplainUnavailable, match="implementation check"):
        load(config)


def test_rows_that_failed_the_implementation_check_are_never_drawn(env):
    config, p = env
    columns = write_explain(p)
    path = p.feature_set_dir / "explain" / "shap_test.parquet"
    frame = pd.read_parquet(path)
    bad = frame[C.LABEL] == 1
    keep = np.flatnonzero(bad.to_numpy())[0]
    frame.loc[bad & (frame.index != keep), ex.SHAP_PREFIX + columns[0]] = np.nan
    frame.to_parquet(path, index=False)
    model = load(config)
    assert model.notes and "never drawn" in model.notes[0]
    for seed in range(10):
        assert model.draw("fraud", seed)["node_id"] == int(frame.loc[keep, C.NODE_ID])


def test_margins_that_do_not_give_the_probabilities_stop_the_model(env):
    config, p = env
    write_explain(p)
    path = p.feature_set_dir / "explain" / "shap_test.parquet"
    frame = pd.read_parquet(path)
    frame["prob"] = 0.5
    frame.to_parquet(path, index=False)
    with pytest.raises(ex.ExplainUnavailable, match="probabilities"):
        load(config)


def test_a_model_without_neighbour_influence_says_so(env):
    config, p = env
    write_explain(p, influence=False)
    path = p.feature_set_dir / "explain" / "explain_meta.json"
    meta = json.loads(path.read_text("utf-8"))
    meta["sections"]["neighbours"] = {"skipped": True}
    path.write_text(json.dumps(meta), encoding="utf-8")
    model = load(config)
    summary = model.summary()
    assert not summary["has_influence"] and summary["influence_note"] == ""
    assert model.draw("any", 1)["neighbours"] == {"available": False, "relations": []}


def test_a_missing_explanation_says_what_to_run(env):
    config, _ = env
    with pytest.raises(ex.ExplainUnavailable, match="shap_explain.py"):
        load(config)


def test_the_server_offers_loaded_models_and_refuses_unknown_ones(
    env, tmp_path, monkeypatch
):
    config, p = env
    write_explain(p)
    monkeypatch.setattr(
        ex, "TARGETS", (("fake", "all", (FS,)), ("fake", "none", ("pilot_m1_own",)))
    )
    app = srv.App(config, tmp_path / "empty_db", use_model=False)
    app.load_explain()
    models = {m["key"]: m for m in app.api_explain_models()["models"]}
    assert models["fake-all"]["state"] == srv.READY
    assert models["fake-none"]["state"] == srv.UNAVAILABLE
    card = app.api_explain_node({"model": "fake-all", "draw": "any", "seed": 3})
    assert card["seed"] == 3
    for body in ({"model": "nope"}, {"model": "fake-none"}, ["fake-all"]):
        with pytest.raises(srv.BadRequest):
            app.api_explain_node(body)


# ---- The page's files -----------------------------------------------------
HTML = (STATIC_DIR / "explain.html").read_text("utf-8")
JS = (STATIC_DIR / "explain.js").read_text("utf-8")
CSS = (STATIC_DIR / "explain.css").read_text("utf-8")


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
    assert fetched and all(path.startswith("/api/explain/") for path in fetched)
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
    assert "Read this first" in HTML and "not the cause of fraud" in HTML
    assert "faithful by construction" not in HTML + JS
    assert "the important nodes" not in JS


def test_the_main_page_links_to_this_one():
    index = (STATIC_DIR / "index.html").read_text("utf-8")
    assert 'href="/explain"' in index
