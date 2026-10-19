"""Decision threshold, group bootstrap and sanity band (synthetic numbers)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from fraud_detect.models import evaluation as ev


def oof_table(y, prob, folds=None) -> pd.DataFrame:
    folds = folds if folds is not None else np.arange(len(y)) % 5
    return pd.DataFrame({"fold": folds, "prob": prob, "y": y})


# ---- Threshold -------------------------------------------------------------
def test_threshold_maximises_f1_on_out_of_fold_predictions():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.05])
    found = ev.threshold_from_oof(oof_table(y, prob))
    # Cut at 0.6: TP 3, FP 1, FN 0 -> F1 = 6/7, better than any other cut.
    assert found["threshold"] == pytest.approx(0.6)
    assert found["oof_f1"] == pytest.approx(6 / 7)
    assert found["n_folds"] == 5


def test_threshold_cannot_take_test_data():
    params = list(inspect.signature(ev.threshold_from_oof).parameters)
    assert params == ["oof", "rule"]  # no place for test labels or scores
    y = np.array([1, 0, 1, 0])
    with pytest.raises(ev.EvaluationError, match="lacks columns"):
        ev.threshold_from_oof(pd.DataFrame({"prob": [0.1] * 4, "y": y}))  # no folds
    with pytest.raises(ev.EvaluationError, match="no out-of-fold"):
        ev.threshold_from_oof(oof_table(y, [0.2, np.nan, 0.7, 0.1]))
    with pytest.raises(ev.EvaluationError, match="Unknown threshold rule"):
        ev.threshold_from_oof(oof_table(y, [0.2, 0.3, 0.7, 0.1]), rule="youden")


# ---- Group bootstrap ---------------------------------------------------------
def test_bootstrap_resamples_whole_groups(monkeypatch):
    groups = np.repeat(np.arange(10), [5, 1, 3, 1, 1, 2, 4, 1, 1, 1])
    y = (np.arange(len(groups)) % 2).astype(int)
    seen = []
    original = ev.METRICS["pr_auc"]

    def spy(y_true, score, sample_weight=None):
        if sample_weight is not None:
            seen.append(sample_weight.copy())
        return original(y_true, score, sample_weight=sample_weight)

    monkeypatch.setitem(ev.METRICS, "pr_auc", spy)
    ev.group_bootstrap(y, groups, np.linspace(0, 1, len(y)), n_resamples=50, seed=1)
    assert seen
    for weights in seen:
        # Every row of a group gets the group's draw count, and the counts are
        # a draw of exactly as many groups as there are.
        for g in np.unique(groups):
            assert len(np.unique(weights[groups == g])) == 1
        first_rows = np.unique(groups, return_index=True)[1]
        assert weights[first_rows].sum() == 10


def test_grouped_intervals_are_wider_when_labels_cluster_in_groups():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 10)  # 60 users with 10 reviews each
    y = np.repeat(rng.random(60) < 0.3, 10).astype(int)  # labels per user
    score = y * 0.3 + np.repeat(rng.normal(size=60), 10)  # noise shared per user
    grouped = ev.group_bootstrap(y, groups, score, n_resamples=300, seed=0)
    rows = ev.group_bootstrap(y, np.arange(len(y)), score, n_resamples=300, seed=0)
    assert grouped.estimate == pytest.approx(rows.estimate)
    assert (grouped.high - grouped.low) > 1.5 * (rows.high - rows.low)


def test_paired_bootstrap_of_identical_predictions_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = rng.random(200)
    groups = np.arange(200) // 4
    interval = ev.group_bootstrap(y, groups, score, score.copy(), n_resamples=200)
    assert interval.estimate == 0 and interval.low <= 0 <= interval.high
    assert interval.low == pytest.approx(0) and interval.high == pytest.approx(0)
    assert interval.metric == "pr_auc difference"


def test_paired_bootstrap_sees_a_real_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(scale=0.5, size=400)
    bad = y + rng.normal(scale=3.0, size=400)
    interval = ev.group_bootstrap(y, np.arange(400) // 2, good, bad, metric="auroc")
    assert interval.low > 0


def test_bootstrap_rejects_mismatched_inputs():
    with pytest.raises(ev.EvaluationError, match="same length"):
        ev.group_bootstrap(np.array([0, 1]), np.array([0, 1, 2]), np.array([0.1, 0.9]))


# ---- Sanity band -------------------------------------------------------------
def test_sanity_band_warns_only_well_below_the_band():
    band = {"pr_auc": 0.80, "auroc": 0.94}
    assert ev.sanity_check(0.79, band, 0.05) is None
    assert ev.sanity_check(0.95, band, 0.05) is None
    warning = ev.sanity_check(0.70, band, 0.05)
    assert warning and "pipeline bug" in warning
