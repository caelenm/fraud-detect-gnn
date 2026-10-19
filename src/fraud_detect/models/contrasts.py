"""Ladder contrasts across split repeats, and the decision rule for them.

For a contrast B − A (two feature sets' Model A) over several independently
seeded splits, three pieces of evidence are combined, as fixed in
docs/RESEARCH_PLAN.md §11 (2026-10-19) before the runs:

1. the test ΔPR-AUC on every split (from compare_report.py);
2. its mean and SD across splits;
3. the Nadeau-Bengio corrected paired t-test over the CV folds of all splits
   (training data only): per fold, PR-AUC of B's Model A minus A's, from the
   out-of-fold predictions select_model saved. The correction inflates the
   variance for the overlap between training sets of different folds
   (Nadeau & Bengio, Machine Learning 2003; Bouckaert & Frank 2004).

Verdict: "supported" if Δ > 0 on every split, mean Δ > SD of Δ, and p < 0.05;
otherwise "not distinguishable from zero" for a positive mean and "worse"
for a negative one.

Caveat stated with every use: each side's Model A was chosen by these same
CV folds, which makes both sides' CV scores slightly optimistic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from fraud_detect.models.model_export import MODEL_NAMES

SUPPORTED = "supported"
NOT_DISTINGUISHABLE = "not distinguishable from zero"
WORSE = "worse"


def _selector(value: Any) -> str:
    return (
        ""
        if value is None or (isinstance(value, float) and np.isnan(value))
        else str(value)
    )


def model_a_oof(oof: pd.DataFrame, model_a: dict[str, Any]) -> pd.DataFrame:
    """Model A's tuned out-of-fold rows (row, fold, prob) from
    oof_predictions.parquet, in row order."""
    names = oof["model_cls"].map(MODEL_NAMES).fillna(oof["model_cls"])
    selector = oof["embed_selector"].map(_selector)
    want = (
        _selector(model_a.get("embed_selector"))
        if model_a["selection"] == "embed"
        else ""
    )
    mask = (
        (oof["settings"] == "tuned")
        & (names == model_a["model"])
        & (oof["selection"] == model_a["selection"])
        & ((selector == want) | (model_a["selection"] != "embed"))
    )
    rows = oof[mask].sort_values("row")
    if rows.empty or rows["row"].duplicated().any():
        raise ValueError(f"No unique out-of-fold predictions for Model A {model_a}")
    return rows[["row", "fold", "prob"]].reset_index(drop=True)


def fold_pr_auc(rows: pd.DataFrame, y_train: np.ndarray) -> pd.Series:
    """PR-AUC of each held-out fold, indexed by fold."""
    if len(rows) != len(y_train):
        raise ValueError("Out-of-fold predictions do not cover every training row")
    y = y_train[rows["row"].to_numpy()]
    return pd.Series(
        {
            int(k): average_precision_score(y[(rows["fold"] == k).to_numpy()],
                                            rows.loc[rows["fold"] == k, "prob"])
            for k in sorted(rows["fold"].unique())
        }
    )  # fmt: skip


def corrected_t_test(diffs: Sequence[float], test_train_ratio: float) -> dict[str, float]:
    """Nadeau-Bengio corrected resampled t-test of mean(diffs) = 0, for J
    fold differences: t = mean / sqrt((1/J + n_test/n_train) * var), with
    J - 1 degrees of freedom. For k-fold CV, n_test/n_train = 1/(k - 1)."""
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    if j < 2:
        raise ValueError("Need at least two fold differences")
    var = d.var(ddof=1)
    mean = float(d.mean())
    if var == 0:
        return {"mean": mean, "t": float("inf") if mean else 0.0,
                "p": 0.0 if mean else 1.0, "df": j - 1}  # fmt: skip
    t = mean / np.sqrt((1.0 / j + test_train_ratio) * var)
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    return {"mean": mean, "t": float(t), "p": float(p), "df": j - 1}


def verdict(test_deltas: Sequence[float], p: float, alpha: float = 0.05) -> str:
    """The decision rule fixed before the runs (module docstring)."""
    d = np.asarray(test_deltas, dtype=float)
    mean = d.mean()
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    if (d > 0).all() and mean > sd and p < alpha:
        return SUPPORTED
    return NOT_DISTINGUISHABLE if mean > 0 else WORSE
