"""Model A metrics report: metrics, selection by CV score only, and checks.

df-analyze outputs are faked with invented probabilities (tests/synthetic.py)
and, where files are needed, written to pytest's tmp_path.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from synthetic import fake_df_analyze_results

from fraud_detect.config import apply_overrides
from fraud_detect.models.df_analyze_report import (
    ReportError,
    binary_metrics,
    build_report,
    check_label_encoding,
    choose_model_a,
    load_run,
    metrics_table,
)

Y_TEST = np.array([0, 1] * 10 + [0] * 30)  # 20% positive, like a small test set


def test_binary_metrics_by_hand():
    y = np.array([1, 1, 0, 0, 0])
    prob = np.array([0.9, 0.4, 0.6, 0.2, 0.1])
    pred = (prob > 0.5).astype(int)  # TP 1, FN 1, FP 1, TN 2
    m = binary_metrics(y, prob, pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.6)
    # Ranking: 0.9 (pos), 0.6 (neg), 0.4 (pos), ... -> AP = (1/1 + 2/3) / 2
    assert m["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)
    assert m["auroc"] == pytest.approx(5 / 6)


def test_model_a_is_chosen_by_cv_score_not_test_metrics():
    # catboost has the higher CV score; the fake data gives lgbm (listed later)
    # a stronger test signal.
    cv = {("dummy", "none"): 0.80, ("catboost", "none"): 0.86, ("lgbm", "none"): 0.84}
    entries, tuned = fake_df_analyze_results(Y_TEST, cv)
    table = metrics_table(entries, tuned, Y_TEST)
    assert table["cv_score"].is_monotonic_decreasing
    a = choose_model_a(table)
    assert a["model"] == "catboost"
    lgbm = table[table["model"] == "lgbm"].iloc[0]
    assert lgbm["pr_auc"] > a["pr_auc"]  # better on test, still not chosen


def test_dummy_is_never_model_a_and_ties_break_in_fixed_order():
    cv = {
        ("dummy", "none"): 0.90,
        ("lgbm", "embed"): 0.85,
        ("lgbm", "none"): 0.85,
        ("catboost", "pred"): 0.85,
    }
    entries, tuned = fake_df_analyze_results(Y_TEST, cv)
    table = metrics_table(entries, tuned, Y_TEST)
    a = choose_model_a(table)
    # Tie at 0.85: model name first (catboost < lgbm), then selection order.
    assert (a["model"], a["selection"]) == ("catboost", "pred")
    ordered = list(zip(table["model"], table["selection"], strict=True))
    assert ordered.index(("lgbm", "none")) < ordered.index(("lgbm", "embed"))


def test_dummy_pr_auc_is_close_to_the_positive_rate():
    cv = {("dummy", "none"): 0.8, ("lgbm", "none"): 0.85}
    entries, tuned = fake_df_analyze_results(Y_TEST, cv)
    entries[0]["probs_test"] = [[0.8, 0.2]] * len(Y_TEST)  # constant, like dummy
    table = metrics_table(entries, tuned, Y_TEST)
    dummy = table[table["model"] == "dummy"].iloc[0]
    assert dummy["pr_auc"] == pytest.approx(Y_TEST.mean())
    assert dummy["auroc"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda e, t: e[1].update(score=0.1), "differs"),
        (lambda e, t: e[1].update(probs_test=e[1]["probs_test"][:-1]), "test rows"),
        (lambda e, t: e[1].update(model_cls="SomethingNew"), "Unknown"),
        (lambda e, t: e.pop(), "predictions for"),
        (lambda e, t: t.drop(index=1, inplace=True), "no tuned_models row"),
    ],
)
def test_inconsistent_results_are_rejected(corrupt, message):
    cv = {("dummy", "none"): 0.8, ("lgbm", "none"): 0.85, ("catboost", "none"): 0.84}
    entries, tuned = fake_df_analyze_results(Y_TEST, cv)
    corrupt(entries, tuned)
    with pytest.raises(ReportError, match=message):
        metrics_table(entries, tuned, Y_TEST)


def test_label_encoding_must_keep_fraud_as_class_1():
    check_label_encoding(pd.DataFrame({"0": [0, 1]}))
    with pytest.raises(ReportError):
        check_label_encoding(pd.DataFrame({"0": [1, 0]}))


def test_load_run_and_report_from_files(tmp_path):
    cv = {("dummy", "none"): 0.8, ("lgbm", "none"): 0.85}
    entries, tuned = fake_df_analyze_results(Y_TEST, cv)
    run = tmp_path / "train" / "abc123"
    for sub in ("tuning/test00", "results/test00", "prepared"):
        (run / sub).mkdir(parents=True)
    tuned.to_csv(run / "tuning/test00/tuned_models_00.csv")
    (run / "results/test00/prediction_results_00.json").write_text(
        json.dumps({"predictions": entries})
    )
    pd.DataFrame({"0": [0, 1]}).to_parquet(run / "prepared/labels.parquet")
    (run / "options.json").write_text(json.dumps({"htune_trials": 10, "seed": 555}))

    predictions, tuned_read, options = load_run(tmp_path)
    table = metrics_table(predictions, tuned_read, Y_TEST)
    info = {
        "run": "test-run",
        "htune_trials": options["htune_trials"],
        "seed": options["seed"],
        "n_train": 75,
        "n_test": len(Y_TEST),
        "test_positive_rate": float(Y_TEST.mean()),
    }
    report = build_report(table, choose_model_a(table), info)
    assert "**lgbm** with feature selection `none`" in report
    assert "Dummy baseline PR-AUC" in report
    assert "tuning trials per model: 10" in report


def test_config_overrides():
    config = {"seed": 1, "df_analyze": {"htune_trials": 100, "classifiers": ["lr"]}}
    out = apply_overrides(
        config, ["df_analyze.htune_trials=10", "df_analyze.classifiers=[lgbm, lr]"]
    )
    assert out["df_analyze"] == {"htune_trials": 10, "classifiers": ["lgbm", "lr"]}
    assert config["df_analyze"]["htune_trials"] == 100  # original untouched
    for bad in ["df_analyze.htune_trails=10", "nope.x=1", "seed"]:
        with pytest.raises(ValueError):
            apply_overrides(config, [bad])
