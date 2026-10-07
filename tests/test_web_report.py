"""Web report: confidence cards, escaping, and page content (synthetic data)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from synthetic import fake_test_meta

from fraud_detect import columns as C
from fraud_detect.report import web


def test_confidence_is_probability_of_the_predicted_class():
    prob = np.array([0.9, 0.2, 0.55, 0.45])
    pred = np.array([1, 0, 1, 1])  # last one predicts fraud at 0.45
    actual = np.array([1, 1, 0, 1])
    table = web.confidence_table(prob, pred, actual, fake_test_meta(4))
    assert table["confidence"].tolist() == pytest.approx([0.9, 0.8, 0.55, 0.45])
    assert table["correct"].tolist() == [True, False, False, True]


def test_most_and_least_confident_order_and_ties():
    prob = np.array([0.5, 0.99, 0.01, 0.7, 0.3, 0.99])
    pred = (prob > 0.5).astype(int)
    table = web.confidence_table(prob, pred, pred, fake_test_meta(6))
    most, least = web.most_and_least_confident(table, 3)
    # 0.99 (node 9001), 0.99 (9002, via 1 - 0.01), 0.99 (9005): ties by node ID
    assert most[C.NODE_ID].tolist() == [9001, 9002, 9005]
    assert least[C.NODE_ID].tolist() == [9000, 9003, 9004]


def test_mismatched_lengths_are_rejected():
    with pytest.raises(web.WebReportError, match="Row counts"):
        web.confidence_table(
            np.array([0.5]), np.array([1, 0]), np.array([1]), fake_test_meta(1)
        )


def test_prob_text_keeps_extremes_distinguishable():
    assert web.prob_text(0.5) == "0.500"
    assert web.prob_text(0.00002) == "2.0e-05"
    assert web.prob_text(0.99998) == "1 − 2.0e-05"
    assert web.prob_text(0.99995) != web.prob_text(0.99998)
    assert web.prob_text(1.0) == "1" and web.prob_text(0.0) == "0"


def test_top_features_rank_by_training_percentile():
    train = pd.DataFrame({"own__a": [0.0, 1.0, 2.0, 3.0], "own__b": [10, 20, 30, 40]})
    test = pd.DataFrame({"own__a": [3.0, 0.0], "own__b": [15, 100]})
    top = web.top_features(test, train, n=1)
    # Row 0: own__a at the top of its training range beats own__b at the 25th
    # percentile, although own__b's raw value is larger.
    assert top[0] == [("own__a", 3.0, 1.0)]
    assert top[1] == [("own__b", 100.0, 1.0)]
    with pytest.raises(web.WebReportError):
        web.top_features(test[["own__b", "own__a"]], train, n=1)


def test_helpers():
    assert web.feature_set_key("embed", "linear") == "embed_linear"
    assert web.feature_set_key("none", float("nan")) == "none"
    assert web.parse_params('{"depth": 6}') == {"depth": 6}
    assert web.parse_params("{'depth': 6, 'x': True}") == {"depth": 6, "x": True}


def test_find_entry_matches_model_and_feature_set():
    predictions = [
        {
            "model_cls": "LightGBMClassifier",
            "selection": "embed",
            "embed_select_model": "linear",
        },
        {
            "model_cls": "CatBoostClassifier",
            "selection": "none",
            "embed_select_model": None,
        },
        {
            "model_cls": "CatBoostClassifier",
            "selection": "embed",
            "embed_select_model": "linear",
        },
    ]
    names = {"LightGBMClassifier": "lgbm", "CatBoostClassifier": "catboost"}
    model_a = {"model": "catboost", "selection": "embed", "embed_selector": "linear"}
    assert web.find_entry(predictions, model_a, names) is predictions[2]
    with pytest.raises(web.WebReportError):
        web.find_entry(predictions, {**model_a, "model": "gandalf"}, names)


def report_inputs(n: int = 30, n_samples: int = 5) -> web.ReportInputs:
    rng = np.random.default_rng(0)
    prob = rng.uniform(0, 1, n)
    pred = (prob > 0.5).astype(int)
    actual = rng.integers(0, 2, n)
    meta = fake_test_meta(n)
    unsafe_name = "<script>alert('synthetic')</script> & more"
    meta.at[0, "top_features"] = [(unsafe_name, 0.5, 0.9)]
    model_a = {
        "model": "catboost", "selection": "embed", "embed_selector": "linear",
        "cv_pr_auc": 0.5, "cv_pr_auc_std": 0.01, "tuning_metric": "BalancedAccuracy",
        "tuning_score": 0.6, "pr_auc": 0.5, "auroc": 0.7, "f1": 0.4, "precision": 0.5,
        "recall": 0.3, "accuracy": 0.8, "balanced_accuracy": 0.6,
        "tp": 1, "fp": 2, "fn": 3, "tn": 4,
        "cv_pr_auc_default": 0.45, "cv_tuning_gain": 0.05,
    }  # fmt: skip
    metrics = pd.DataFrame(
        [
            {**model_a},
            {**model_a, "model": "lgbm", "selection": "none", "embed_selector": np.nan,
             "cv_pr_auc_default": 0.52, "cv_tuning_gain": -0.02},
        ]
    )  # fmt: skip
    budget = pd.DataFrame(
        [{"model": "CatBoost Classifier", "selection": "embed_linear",
          "trials_completed": 10, "trials_requested": 10, "elapsed_s": 60.0,
          "time_limit_s": 1800, "stopped_by": "all trials"}]
    )  # fmt: skip
    summary = {
        "run": "SYN-RUN", "seed": 1, "n_train": 60, "n_test": n,
        "test_positive_rate": 0.2, "cv_folds": 5, "model_a": model_a,
    }  # fmt: skip
    return web.ReportInputs(
        summary=summary,
        metrics=metrics,
        params={"depth": 6, "learning_rate": 0.123456789},
        n_features=12,
        samples=web.confidence_table(prob, pred, actual, meta),
        header={
            "build": "SYN-BUILD",
            "generated": "now",
            "git_commit": "abc",
            "run": "SYN-RUN",
        },
        run_config=[("Seed", "1"), ("Nodes", "60 train")],
        budget=budget,
        n_samples=n_samples,
        dataset_label="Synthetic",
    )


def test_render_escapes_values_and_fills_every_section():
    page = web.render(report_inputs(n_samples=5))
    assert "<script>alert" not in page  # every inserted value is escaped
    assert "&lt;script&gt;" in page or "Node 9000" not in page
    assert page.count('class="sample"') == 10  # 5 per card
    assert page.count('class="selected"') == 1  # Model A's row
    assert "CatBoost" in page and "Embedded: linear" in page
    assert "0.123457" in page  # tuned hyperparameter, 6 significant digits
    assert "10 of 10 · all trials run" in page
    assert "CV PR-AUC 0.450 untuned → 0.500 tuned (+0.050)" in page
    assert '<span class="gain good">+0.050</span>' in page
    assert '<span class="gain bad">-0.020</span>' in page
    assert "Contains row-level test data" in page
    assert "complaint" not in page.lower() and "narrative" not in page.lower()
    assert "<link" not in page and "<script" not in page  # self-contained, static
    assert re.search(r'style="width:\d+\.\d%"', page)


def test_report_shows_the_escaped_feature_name_when_selected():
    inputs = report_inputs(n=6, n_samples=6)  # every node appears
    page = web.render(inputs)
    assert "&lt;script&gt;alert(&#x27;synthetic&#x27;)&lt;/script&gt; &amp; more" in page
