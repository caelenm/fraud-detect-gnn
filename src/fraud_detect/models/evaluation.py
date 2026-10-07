"""Decision thresholds, group-bootstrap intervals and the sanity-band check.

Decision threshold. F1, precision, recall and the confusion matrix need a
cut-off on the fraud probability. It is chosen on a model's OUT-OF-FOLD
predictions on the training set (from select_model's shared grouped CV),
never on the test set: `threshold_from_oof` only accepts that table. The
default rule maximises F1 there (an open decision, plan §10). Test metrics
are also reported at 0.5.

Bootstrap. Test examples are not independent: a YelpChi user's reviews share
a label, and Amazon's duplicate rows are copies. Intervals therefore
resample whole split groups, with replacement. Drawing groups with
replacement is the same as giving each group a multinomial count and
weighting its rows by that count, which is how it is computed here. The
paired version takes two models' scores on the same test nodes and resamples
their difference, as the later ladder contrasts (5 - 1, 2 - 1) need.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

THRESHOLD_RULES = ("max_f1",)

Metric = Callable[..., float]
METRICS: dict[str, Metric] = {
    "pr_auc": average_precision_score,
    "auroc": roc_auc_score,
}


class EvaluationError(ValueError):
    """Inputs cannot be evaluated as asked; the message says why."""


# --------------------------------------------------------------------------
# Threshold from out-of-fold training predictions
# --------------------------------------------------------------------------
def threshold_from_oof(oof: pd.DataFrame, rule: str = "max_f1") -> dict[str, Any]:
    """Decision threshold from one model's out-of-fold training predictions.

    `oof` is that model's rows of oof_predictions.parquet: one row per
    training example with `fold`, `prob` (the out-of-fold fraud probability)
    and `y` (its label). Test data cannot be passed: there is no argument for
    it, and the table must come from the CV folds.
    """
    if rule not in THRESHOLD_RULES:
        raise EvaluationError(f"Unknown threshold rule {rule!r}; use {THRESHOLD_RULES}")
    missing = {"fold", "prob", "y"} - set(oof.columns)
    if missing:
        raise EvaluationError(f"Out-of-fold table lacks columns {sorted(missing)}")
    if oof["prob"].isna().any() or oof["fold"].isna().any():
        raise EvaluationError("Some training rows have no out-of-fold prediction")
    y = oof["y"].to_numpy(dtype=int)
    prob = oof["prob"].to_numpy(dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, prob)
    # precision/recall have one more entry than thresholds (the last point,
    # recall 0, has no threshold); drop it.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best = int(np.flatnonzero(f1 == f1.max())[-1])  # ties: the highest threshold
    return {
        "rule": rule,
        "threshold": float(thresholds[best]),
        "oof_f1": float(f1[best]),
        "oof_precision": float(p[best]),
        "oof_recall": float(r[best]),
        "n_train_rows": len(oof),
        "n_folds": int(oof["fold"].nunique()),
    }


# --------------------------------------------------------------------------
# Group bootstrap
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Interval:
    metric: str
    estimate: float  # on the full test set (for paired: score_a - score_b)
    low: float
    high: float
    n_resamples: int  # resamples with both classes present (others skipped)
    level: float = 0.95

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "level": self.level,
            "n_resamples": self.n_resamples,
        }


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, int]:
    _, idx = np.unique(np.asarray(groups), return_inverse=True)
    idx = idx.ravel()
    return idx, int(idx.max()) + 1


def group_bootstrap(
    y: np.ndarray,
    groups: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray | None = None,
    metric: str = "pr_auc",
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Percentile interval of `metric(y, score_a)`, or of the paired difference
    `metric(y, score_a) - metric(y, score_b)`, resampling whole groups."""
    y = np.asarray(y, dtype=int)
    a = np.asarray(score_a, dtype=float)
    b = None if score_b is None else np.asarray(score_b, dtype=float)
    if not (len(y) == len(a) == len(groups)) or (b is not None and len(b) != len(y)):
        raise EvaluationError("y, groups and scores must have the same length")
    fn = METRICS[metric]

    def stat(weights: np.ndarray | None) -> float:
        value = fn(y, a, sample_weight=weights)
        return value - fn(y, b, sample_weight=weights) if b is not None else value

    group_idx, n_groups = _group_index(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups))
        weights = counts[group_idx].astype(float)
        present = weights > 0
        if len(np.unique(y[present])) < 2:
            continue  # a metric needs both classes
        values.append(stat(weights))
    if not values:
        raise EvaluationError("No bootstrap resample contained both classes")
    alpha = (1.0 - level) / 2
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return Interval(
        metric=metric if b is None else f"{metric} difference",
        estimate=float(stat(None)),
        low=float(low),
        high=float(high),
        n_resamples=len(values),
        level=level,
    )


# --------------------------------------------------------------------------
# Sanity band (plan §2)
# --------------------------------------------------------------------------
def sanity_check(pr_auc: float, band: dict[str, float], margin: float) -> str | None:
    """A warning if test PR-AUC is more than `margin` below the dataset's
    sanity band (an untuned gradient-boosting model's score), else None."""
    floor = float(band["pr_auc"]) - margin
    if pr_auc >= floor:
        return None
    return (
        f"Model A's test PR-AUC {pr_auc:.3f} is more than {margin} below the sanity "
        f"band ({band['pr_auc']:.2f}, from an untuned gradient-boosting model; "
        "docs/RESEARCH_PLAN.md §2). Assume a pipeline bug (mis-typed features, a "
        "wrong label vector, a misaligned split) and investigate before reporting."
    )
