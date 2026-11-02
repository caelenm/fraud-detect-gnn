"""The classify panel: only test nodes, and every number checked first.

A fake dataset and faked df-analyze runs in tmp_path (tests/helpers.py), with an
invented model; df-analyze and torch never run. Each failure test breaks one
file and asserts that the panel refuses, because showing a wrong score in front
of an audience is the failure this module exists to prevent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from helpers import (
    fake_dataset,
    fake_live_inputs,
    finish_fake_run,
    table_worker,
)
from sklearn.metrics import average_precision_score

from fraud_detect import columns as C
from fraud_detect.cli import main
from fraud_detect.config import get_paths
from fraud_detect.dashboard import classify as cl
from fraud_detect.dashboard import registry as reg
from fraud_detect.dashboard.worker import WorkerError
from fraud_detect.models import gnn_outputs
from fraud_detect.models import model_export as mx

CV_B = {("dummy", "none"): 0.2, ("lgbm", "none"): 0.6}
CV_E1 = {("dummy", "none"): 0.2, ("lgbm", "none"): 0.5, ("catboost", "none"): 0.7}
E1 = "pilot_m2_own_nbr"
BASE = "pilot_m1_own"


def best_for(feature_set: str = E1, gnn_rung: bool = False) -> dict:
    fs = reg.parse_feature_set(feature_set)
    return {
        "feature_set": feature_set,
        "display_id": fs.display_id,
        "label": fs.setup.label,
        "gnn_rung": gnn_rung,
        "close": [],
    }


@pytest.fixture
def env(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=200)
    finish_fake_run(path, config, BASE, CV_B)
    finish_fake_run(path, config, E1, CV_E1)
    p = get_paths({**config, "feature_set": E1})
    fake_live_inputs(p)
    return config, p


def target_of(config, feature_set: str = E1, headline=None, gnn_rung: bool = False):
    return cl.load_target(
        config, "fake", reg.ALL, best_for(feature_set, gnn_rung), headline
    )


def live_target(config, p):
    target = target_of(config)
    worker = table_worker(target.key, p, target.side.prob)
    check = cl.verify_live(target, worker)
    assert check.ok
    return target, worker


def test_loading_passes_every_check_and_starts_on_saved_predictions(env):
    config, _ = env
    target = target_of(config)
    assert all(c.ok for c in target.checks), [c for c in target.checks if not c.ok]
    assert {c.name for c in target.checks} >= {
        "No test node was trained on",
        "Saved predictions reproduce the recorded PR-AUC",
        "Baseline B has the same test nodes",
        "The exported model is this run's Model A",
        "The prepared test table lines up with the test IDs",
        "The prepared columns are the model's",
    }
    assert target.mode == cl.SAVED and not target.live_verified
    status = target.status()
    assert status["n_test"] == len(target.y) and status["checks_ok"]
    assert status["test_metrics"]["pr_auc"] == pytest.approx(
        target.side.summary["model_a"]["pr_auc"]
    )


def test_whole_test_metrics_are_the_confusion_matrix_at_the_threshold():
    y = np.array([1, 1, 0, 0, 0])
    prob = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
    m = cl.whole_test_metrics(y, prob, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["no_skill"] == pytest.approx(0.4)


# ---- Each fault stops the panel -------------------------------------------
def test_a_training_node_among_the_test_nodes_stops_the_panel(env):
    config, p = env
    train = pd.read_csv(p.train_ids)
    test = pd.read_csv(p.test_ids)
    train.loc[0, C.NODE_ID] = test.loc[0, C.NODE_ID]
    train.to_csv(p.train_ids, index=False)
    with pytest.raises(cl.TargetUnavailable, match="trained on"):
        target_of(config)


def test_a_recorded_score_the_predictions_do_not_give_stops_the_panel(env):
    config, p = env
    path = p.model_reports_dir / "model_a.json"
    summary = json.loads(path.read_text("utf-8"))
    summary["model_a"]["pr_auc"] += 0.01
    path.write_text(json.dumps(summary), "utf-8")
    with pytest.raises(cl.TargetUnavailable, match="reproduce the recorded"):
        target_of(config)


def test_a_database_entry_that_disagrees_with_the_run_stops_the_panel(env):
    config, p = env
    target = target_of(config)
    a = target.side.summary["model_a"]
    good = {
        "test_pr_auc": a["pr_auc"],
        "threshold": a["threshold"],
        "n_test": len(target.y),
    }
    assert all(c.ok for c in target_of(config, headline=good).checks)
    for fault in ({"test_pr_auc": 0.1}, {"threshold": 0.99}, {"n_test": 1}):
        with pytest.raises(cl.TargetUnavailable, match="results-database"):
            target_of(config, headline={**good, **fault})


@pytest.mark.parametrize(
    "change",
    [{"run": "other"}, {"model": "lr"}, {"selection": "pred"}, {"threshold": 0.123}],
    ids=lambda c: next(iter(c)),
)
def test_an_export_of_some_other_model_stops_the_panel(env, change):
    config, p = env
    path = p.model_dir / mx.MODEL_A_META
    meta = json.loads(path.read_text("utf-8"))
    meta.update(change)
    path.write_text(json.dumps(meta), "utf-8")
    with pytest.raises(cl.TargetUnavailable, match="this run's Model A"):
        target_of(config)


def test_a_missing_model_file_stops_the_panel(env):
    config, p = env
    (p.model_dir / "model_a.joblib").unlink()
    with pytest.raises(cl.TargetUnavailable, match="model file"):
        target_of(config)


def export_dir(p):
    return next(p.df_analyze_output_dir.rglob("results/test00"))


def test_a_prepared_table_with_other_rows_stops_the_panel(env):
    config, p = env
    path = export_dir(p) / "X_test_00.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    with pytest.raises(cl.TargetUnavailable, match="lines up"):
        target_of(config)


def test_a_label_export_in_another_order_stops_the_panel(env):
    config, p = env
    path = export_dir(p) / "y_test_00.csv"
    frame = pd.read_csv(path, index_col=0)
    frame.iloc[:, 0] = frame.iloc[:, 0].to_numpy()[::-1]
    assert not np.array_equal(frame.iloc[:, 0], pd.read_csv(p.test_ids)[C.LABEL])
    frame.to_csv(path)
    with pytest.raises(cl.TargetUnavailable, match="lines up"):
        target_of(config)


def test_prepared_columns_the_model_does_not_take_stop_the_panel(env):
    config, p = env
    path = export_dir(p) / "X_test_00.csv"
    pd.read_csv(path).rename(columns={"x": "y"}).to_csv(path, index=False)
    with pytest.raises(cl.TargetUnavailable, match="prepared columns"):
        target_of(config)


def test_no_outputs_says_what_is_needed(tmp_path):
    _, config = fake_dataset(tmp_path, n_nodes=200)
    with pytest.raises(cl.TargetUnavailable, match="docs/DASHBOARD.md"):
        target_of(config)


def test_no_export_falls_back_to_saved_predictions_and_says_so(env):
    config, p = env
    (p.model_dir / mx.MODEL_A_META).unlink()
    target = target_of(config)
    assert target.mode == cl.SAVED and target.prepared is None
    assert "No exported model" in target.note
    result = cl.classify(target, None, "any", 1)
    assert result["mode"] == cl.SAVED and result["verified"] is False


# ---- Verifying the running model ------------------------------------------
def test_verify_live_passes_when_every_row_matches(env):
    config, p = env
    target, worker = live_target(config, p)
    assert target.mode == cl.LIVE and target.live_verified
    assert target.checks[-1].name == (
        "The running model reproduces every saved test probability"
    )
    assert f"{len(target.y):,} rows" in target.checks[-1].detail
    assert worker.calls == 1


def test_verify_live_refuses_a_model_that_differs(env):
    config, p = env
    target = target_of(config)
    worker = table_worker(target.key, p, target.side.prob, offset=1e-3)
    check = cl.verify_live(target, worker)
    assert not check.ok and target.mode == cl.SAVED and not target.live_verified
    assert "saved ones" in target.note


def test_verify_live_notices_one_wrong_row(env):
    config, p = env
    target = target_of(config)
    probs = target.side.prob.copy()
    probs[-1] = np.clip(probs[-1] + 0.2, 0, 1) if probs[-1] < 0.5 else probs[-1] - 0.2
    check = cl.verify_live(target, table_worker(target.key, p, probs))
    assert not check.ok and target.mode == cl.SAVED


def test_the_tolerance_comes_from_the_model_kind(env):
    config, p = env
    target = target_of(config)
    # Model A of this fake run is catboost, which trains on the GPU.
    assert target.side.summary["model_a"]["model"] == "catboost"
    assert target.tolerance == mx.GPU_TOLERANCE


def test_verifying_without_an_export_is_an_error(env):
    config, p = env
    (p.model_dir / mx.MODEL_A_META).unlink()
    target = target_of(config)
    with pytest.raises(cl.ClassifyError):
        cl.verify_live(target, None)


# ---- Draws ----------------------------------------------------------------
def test_draws_only_return_what_was_asked_for():
    y = np.array([0] * 90 + [1] * 10)
    rng = np.random.default_rng(0)
    for _ in range(300):
        assert y[cl.draw_row(y, "fraud", rng)] == 1
        assert y[cl.draw_row(y, "benign", rng)] == 0
    rows = {cl.draw_row(y, "any", rng) for _ in range(2000)}
    assert rows == set(range(100))  # every row can be drawn


def test_a_draw_of_a_class_the_test_set_lacks_is_an_error():
    with pytest.raises(cl.ClassifyError, match="no fraud"):
        cl.draw_row(np.zeros(5, dtype=int), "fraud", np.random.default_rng(0))
    with pytest.raises(cl.ClassifyError, match="Unknown draw"):
        cl.draw_row(np.zeros(5, dtype=int), "nope", np.random.default_rng(0))


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, "7", True, [1]])
def test_a_bad_seed_is_refused(seed):
    with pytest.raises(cl.ClassifyError, match="seed"):
        cl.make_rng("any", seed)


def test_the_same_seed_gives_the_same_node_and_the_buttons_differ(env):
    config, p = env
    target, worker = live_target(config, p)
    a = cl.classify(target, worker, "any", 123)
    b = cl.classify(target, worker, "any", 123)
    assert a["node_id"] == b["node_id"] and a["prob"] == b["prob"]
    picks = {cl.classify(target, worker, "any", s)["node_id"] for s in range(40)}
    assert len(picks) > 5  # different seeds reach different nodes
    assert {cl.classify(target, worker, "any", None)["node_id"] for _ in range(40)} != {
        a["node_id"]
    }


def test_classify_never_returns_a_training_node(env):
    config, p = env
    target, worker = live_target(config, p)
    train = set(pd.read_csv(p.train_ids)[C.NODE_ID])
    test = set(pd.read_csv(p.test_ids)[C.NODE_ID])
    seen = set()
    for draw in cl.DRAWS:
        for seed in range(80):
            result = cl.classify(target, worker, draw, seed)
            assert result["node_id"] not in train
            assert result["node_id"] in test
            seen.add(result["node_id"])
    assert len(seen) > 20


@pytest.mark.parametrize(("draw", "label"), [("fraud", 1), ("benign", 0)])
def test_classify_returns_the_asked_class_and_its_true_label(env, draw, label):
    config, p = env
    target, worker = live_target(config, p)
    labels = pd.read_csv(p.test_ids).set_index(C.NODE_ID)[C.LABEL]
    for seed in range(30):
        result = cl.classify(target, worker, draw, seed)
        assert result["actual"] == label == labels[result["node_id"]]


# ---- One classification ---------------------------------------------------
def test_a_result_is_complete_and_consistent(env):
    config, p = env
    target, worker = live_target(config, p)
    result = cl.classify(target, worker, "any", 5)
    row = int(np.flatnonzero(target.node_ids == result["node_id"])[0])
    assert result["mode"] == cl.LIVE and result["verified"] is True
    assert result["prob"] == pytest.approx(target.side.prob[row], abs=target.tolerance)
    assert result["saved_prob"] == target.side.prob[row]
    assert result["difference"] <= target.tolerance
    assert result["threshold"] == target.side.threshold
    assert result["predicted"] == int(result["prob"] >= result["threshold"])
    assert result["correct"] == (result["predicted"] == result["actual"])
    if result["predicted"]:
        assert result["confidence"] == pytest.approx(result["prob"])
    else:
        assert result["confidence"] == pytest.approx(1 - result["prob"])
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["setup"] == "E1"


@pytest.mark.parametrize(
    ("predicted", "actual", "text"),
    [
        (1, 1, "correct: caught fraud"),
        (0, 0, "correct: not fraud"),
        (1, 0, "false positive"),
        (0, 1, "false negative"),
    ],
)
def test_outcome_labels(predicted, actual, text):
    assert cl.outcome_label(predicted, actual) == text


def test_the_baseline_is_the_same_node_under_b(env):
    config, p = env
    target, worker = live_target(config, p)
    result = cl.classify(target, worker, "any", 9)
    row = int(np.flatnonzero(target.node_ids == result["node_id"])[0])
    base = result["baseline"]
    assert base["setup"] == "B"
    assert base["prob"] == target.baseline.prob[row]
    assert base["threshold"] == target.baseline.threshold
    assert base["predicted"] == int(base["prob"] >= base["threshold"])
    assert base["changed"] == (base["predicted"] != result["predicted"])


def test_a_baseline_on_other_test_nodes_stops_the_panel(env, monkeypatch):
    config, _ = env
    real = cl.compare.load_side

    def other_nodes(cfg, feature_set):
        side = real(cfg, feature_set)
        if feature_set == BASE:
            ids = side.test_ids.copy()
            ids[C.NODE_ID] = ids[C.NODE_ID].to_numpy()[::-1]
            side.test_ids = ids
        return side

    monkeypatch.setattr(cl.compare, "load_side", other_nodes)
    with pytest.raises(cl.TargetUnavailable, match="same test"):
        target_of(config)


def test_top_features_are_the_models_own_and_described_as_percentiles(env):
    config, p = env
    target, worker = live_target(config, p)
    result = cl.classify(target, worker, "any", 3)
    used = {cl.compare.feature_label(c) for c in target.side.used_columns}
    assert 0 < len(result["top_features"]) <= cl.N_TOP_FEATURES
    for feature in result["top_features"]:
        assert feature["name"] in used
        assert 0 <= feature["percentile"] <= 1
    percentiles = [f["percentile"] for f in result["top_features"]]
    assert percentiles == sorted(percentiles, reverse=True)


def test_neighbourhoods_cover_every_relation(env):
    config, p = env
    target, worker = live_target(config, p)
    result = cl.classify(target, worker, "any", 4)
    assert [n["relation"] for n in result["neighbourhoods"]] == ["A-A-A", "B-B-B"]
    for n in result["neighbourhoods"]:
        assert n["svg"].startswith("<svg ") and n["svg"].endswith("</svg>")
        assert n["n_shown"] <= n["degree"]
        assert n["n_train_neighbours"] <= n["degree"]
        if n["train_fraud_share"] is not None:
            assert 0 <= n["train_fraud_share"] <= 1


# ---- A wrong prediction never reaches the page -----------------------------
def test_a_running_model_that_starts_to_disagree_is_stopped(env):
    config, p = env
    target, worker = live_target(config, p)
    worker.models[target.key].estimator.offset = 0.01
    with pytest.raises(cl.ClassifyError, match="Stopped rather than"):
        cl.classify(target, worker, "any", 1)


def test_a_worker_failure_is_a_classify_error(env):
    config, p = env
    target, worker = live_target(config, p)

    def broken(*_):
        raise WorkerError("the worker stopped")

    worker.predict = broken
    with pytest.raises(cl.ClassifyError, match="the worker stopped"):
        cl.classify(target, worker, "any", 1)
    with pytest.raises(cl.ClassifyError, match="not running"):
        cl.classify(target, None, "any", 1)


def test_saved_mode_never_calls_the_worker_and_never_claims_to_be_verified(env):
    config, p = env
    target = target_of(config)
    worker = table_worker(target.key, p, target.side.prob)
    result = cl.classify(target, worker, "any", 2)
    assert worker.calls == 0
    assert result["mode"] == cl.SAVED and result["verified"] is False
    assert result["difference"] is None


# ---- A GNN rung ------------------------------------------------------------
def fake_gnn_rung(path, config, feature_set: str = "pilot_g5_sage") -> None:
    """A GNN rung's saved outputs, with invented probabilities."""
    args = ["--config", str(path), "--set", f"feature_set={feature_set}"]
    assert main([*args, "--to", "df_analyze_input"]) == 0
    p = get_paths({**config, "feature_set": feature_set})
    test_ids = pd.read_csv(p.test_ids)
    rng = np.random.default_rng(1)
    y = test_ids[C.LABEL].to_numpy()
    prob = np.clip(0.5 * y + rng.uniform(0, 0.5, len(y)), 0, 1)
    p.model_reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({C.NODE_ID: test_ids[C.NODE_ID], "prob": prob}).to_parquet(
        p.model_reports_dir / gnn_outputs.GNN_TEST, index=False
    )
    summary = {
        "source": gnn_outputs.GNN_SOURCE,
        "dataset": "fake",
        "feature_set": feature_set,
        "run": "gnn",
        "model_a": {
            "model": "GraphSAGE",
            "selection": "gnn",
            "pr_auc": float(average_precision_score(y, prob)),
            "threshold": 0.5,
        },
        "gnn": {"own_columns": ["own__f00", "own__f01"]},
    }
    (p.model_reports_dir / "model_a.json").write_text(json.dumps(summary), "utf-8")


def test_a_gnn_rung_shows_saved_predictions_and_says_so(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=200)
    finish_fake_run(path, config, BASE, CV_B)
    fake_gnn_rung(path, config)
    target = target_of(config, "pilot_g5_sage", gnn_rung=True)
    assert target.mode == cl.SAVED and target.prepared is None
    assert all(c.ok for c in target.checks)
    assert "saved test prediction, not a live one" in target.note
    result = cl.classify(target, None, "any", 6)
    row = int(np.flatnonzero(target.node_ids == result["node_id"])[0])
    assert result["prob"] == target.side.prob[row]
    assert result["mode"] == cl.SAVED and result["verified"] is False
