"""Model A metrics report from a finished df-analyze run.

df-analyze saves, for every (classifier, feature selection) combination it
tuned, the tuning score and the tuned model's predictions and class
probabilities on the test set (`prediction_results_00.json`). This module
turns those into one metrics table on the frozen test set, with the project's
headline metric (PR-AUC) that df-analyze itself does not report.

Model A is chosen by the shared cross-validation of the `select_model` stage
(scripts/dfa/cv_select.py): every combination refit on the same folds of the
training set and scored by PR-AUC from its probabilities (leakage invariant
9). df-analyze's own tuning score is shown for information only: it is not
comparable across models (most are scored with balanced accuracy of hard
predictions over 5 folds, GANDALF on one validation split through its own
code path). Test metrics are reported for every combination but never used
for choosing. df-analyze's `5-fold` table refits models on folds of the test
set and is not used at all.

The test rows are in the order of our saved test IDs: the df_analyze stage
has already verified df-analyze's exported test set against ours row by row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# df-analyze estimator classes -> the classifier names used in the config.
MODEL_NAMES = {
    "CatBoostClassifier": "catboost",
    "DummyClassifier": "dummy",
    "GandalfEstimator": "gandalf",
    "KNNClassifier": "knn",
    "LRClassifier": "lr",
    "LightGBMClassifier": "lgbm",
    "LightGBMRFClassifier": "rf",
    "MLPEstimator": "mlp",
}
# Tie-break order for equal CV scores (fixed in advance, never test-based).
SELECTION_ORDER = ("none", "assoc", "pred", "embed")
# Shared-CV columns (from scripts/dfa/cv_select.py) -> report column names.
SHARED_CV_COLUMNS = {
    "pr_auc_mean": "cv_pr_auc",
    "pr_auc_std": "cv_pr_auc_std",
    "auroc_mean": "cv_auroc",
    "bal_acc_mean": "cv_bal_acc",
}
BASELINE_MODEL = "dummy"

METRIC_COLUMNS = [
    "pr_auc",
    "auroc",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "balanced_accuracy",
]


class ReportError(RuntimeError):
    """Raised when df-analyze's results cannot be read or do not line up."""


@dataclass(frozen=True)
class RunFiles:
    tuned_models: Path
    predictions: Path
    labels: Path
    options: Path


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise ReportError(
            f"Expected exactly one {pattern} under {root}, found "
            f"{[str(m) for m in matches]}"
        )
    return matches[0]


def find_run_files(run_outdir: Path) -> RunFiles:
    return RunFiles(
        tuned_models=_one(run_outdir, "tuning/test00/tuned_models_00.csv"),
        predictions=_one(run_outdir, "results/test00/prediction_results_00.json"),
        labels=_one(run_outdir, "prepared/labels.parquet"),
        options=_one(run_outdir, "options.json"),
    )


def check_label_encoding(labels: pd.DataFrame) -> None:
    """df-analyze re-encodes the target; ours must stay 0 -> 0 and 1 -> 1 so that
    probability column 1 is the fraud class."""
    mapping = {int(k): str(v) for k, v in labels.iloc[:, 0].dropna().items()}
    if mapping != {0: "0", 1: "1"}:
        raise ReportError(f"Unexpected df-analyze target encoding: {mapping}")


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    """Binary metrics with fraud (1) as the positive class. `prob` is the fraud
    probability; `pred` is the model's own 0/1 prediction."""
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, prob)),
        "auroc": float(roc_auc_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _tuned_scores(tuned: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (str(r.model), str(r.selection)): float(r.score)
        for r in tuned.itertuples(index=False)
    }


def _shared_cv_scores(shared_cv: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    """Shared-CV rows keyed like the report: (config model name, selection)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in shared_cv.to_dict("records"):
        cls = row["model_cls"]
        if cls not in MODEL_NAMES:
            raise ReportError(f"Unknown df-analyze model class {cls!r} in the CV table")
        key = (MODEL_NAMES[cls], str(row["selection"]))
        error = row.get("error")
        if isinstance(error, str) and error:
            raise ReportError(
                f"{key} could not be cross-validated ({error}). Every model must be "
                "scored the same way before one can be chosen; see model_selection.log."
            )
        out[key] = {new: float(row[old]) for old, new in SHARED_CV_COLUMNS.items()}
    return out


def metrics_table(
    predictions: list[dict[str, Any]],
    tuned: pd.DataFrame,
    y_test: np.ndarray,
    shared_cv: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (model, selection): shared-CV scores on the training set,
    df-analyze's tuning score, and test metrics. Sorted by shared-CV PR-AUC
    (best first; ties by smaller std, then model name, then selection order)."""
    cv_scores = _tuned_scores(tuned)
    shared = _shared_cv_scores(shared_cv)
    rows = []
    for entry in predictions:
        cls = entry["model_cls"]
        if cls not in MODEL_NAMES:
            raise ReportError(f"Unknown df-analyze model class {cls!r}")
        model, selection = MODEL_NAMES[cls], str(entry["selection"])
        key = (model, selection)
        if key not in cv_scores:
            raise ReportError(f"{key} has predictions but no tuned_models row")
        if not np.isclose(cv_scores[key], float(entry["score"])):
            raise ReportError(
                f"{key}: tuning score {cv_scores[key]} differs from the score "
                f"saved with its predictions ({entry['score']})"
            )
        probs = np.asarray(entry["probs_test"], dtype=float)
        pred = np.asarray(entry["preds_test"], dtype=int)
        if probs.shape != (len(y_test), 2) or pred.shape != (len(y_test),):
            raise ReportError(
                f"{key}: predictions have shape {probs.shape}/{pred.shape}, "
                f"expected {len(y_test)} test rows"
            )
        if key not in shared:
            raise ReportError(
                f"{key} has no shared cross-validation score; rerun the select_model "
                "stage so every model is scored the same way."
            )
        rows.append(
            {
                "model": model,
                "selection": selection,
                "embed_selector": entry.get("embed_select_model") or "",
                **shared[key],
                "tuning_metric": str(entry["metric"]),
                "tuning_score": cv_scores[key],
                **binary_metrics(y_test, probs[:, 1], pred),
            }
        )
    if len(rows) != len(cv_scores):
        raise ReportError(
            f"{len(cv_scores)} tuned models but predictions for {len(rows)}"
        )
    table = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(SELECTION_ORDER)}
    table["_sel"] = table["selection"].map(order).fillna(len(order))
    table = table.sort_values(
        ["cv_pr_auc", "cv_pr_auc_std", "model", "_sel"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    return table.drop(columns="_sel").reset_index(drop=True)


def choose_model_a(table: pd.DataFrame) -> pd.Series:
    """Best non-baseline combination by shared-CV PR-AUC on the training set.
    `table` must be sorted by `metrics_table` (its order encodes the tie-break)."""
    candidates = table[table["model"] != BASELINE_MODEL]
    if candidates.empty:
        raise ReportError("No tuned models besides the dummy baseline")
    return candidates.iloc[0]


def _fmt(value: Any) -> str:
    if isinstance(value, float | np.floating):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "|" + "|".join("---" for _ in frame.columns) + "|",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def build_report(
    table: pd.DataFrame,
    model_a: pd.Series,
    info: dict[str, Any],
    budget: pd.DataFrame | None = None,
) -> str:
    baseline = table[table["model"] == BASELINE_MODEL]
    shown = ["model", "selection", "cv_pr_auc", "cv_pr_auc_std", "tuning_score",
             *METRIC_COLUMNS]  # fmt: skip
    a = model_a
    baseline_line = (
        [f"- Dummy baseline PR-AUC: {baseline['pr_auc'].iloc[0]:.4f}"]
        if not baseline.empty
        else []
    )
    lines = [
        "# Model A (df-analyze) metrics",
        "",
        f"- df-analyze run: `{info['run']}` (tuning trials per model: "
        f"{info['htune_trials']}, seed {info['seed']})",
        f"- Train: {info['n_train']:,} complaints; test: {info['n_test']:,} "
        f"complaints, {info['test_positive_rate']:.2%} fraud",
        f"- No-skill PR-AUC on this test set = its fraud rate: "
        f"{info['test_positive_rate']:.4f}",
        *baseline_line,
        "",
        "## Model A",
        "",
        f"**{a['model']}** with feature selection `{a['selection']}`, chosen by "
        f"its cross-validated PR-AUC on the training set only "
        f"({a['cv_pr_auc']:.4f} ± {a['cv_pr_auc_std']:.4f} over "
        f"{info['cv_folds']} folds shared by every model).",
        "",
        _markdown_table(pd.DataFrame([a[METRIC_COLUMNS]])),
        "",
        f"Confusion matrix on the test set: TP {a['tp']:,}, FP {a['fp']:,}, "
        f"FN {a['fn']:,}, TN {a['tn']:,}.",
        "",
        "## All tuned combinations",
        "",
        "Sorted by cross-validated PR-AUC on the training set (what Model A is "
        "chosen by). The test columns are shown for information only and must "
        "not be used to pick a model.",
        "",
        _markdown_table(table[shown]),
        "",
        *_budget_section(budget),
        "## How to read this",
        "",
        "- **cv_pr_auc** (± **cv_pr_auc_std**): every tuned combination refit "
        f"with its tuned hyperparameters on the same {info['cv_folds']} stratified "
        "folds of the training set and scored from its fraud probability. This "
        "is the only score that is comparable across models.",
        f"- **tuning_score** is df-analyze's own tuning score ({a['tuning_metric']}), "
        "for information only. It is not comparable across models: most are "
        "scored on hard 0/1 predictions over 5 folds, GANDALF on one validation "
        "split through its own code path.",
        "- **PR-AUC** (headline) is average precision of the fraud probability; "
        "**AUROC** uses the same probability. Neither depends on a threshold.",
        "- **F1, precision, recall** are for the fraud class, using each model's "
        "own predictions (df-analyze's default decision rule). df-analyze's own "
        "`f1` is macro-averaged over both classes, so it differs from this one.",
        "- df-analyze's `5-fold` results table refits models on test-set folds "
        "and is deliberately not used.",
        "- These are single-run numbers (one split, one seed), with no variance "
        "estimate.",
    ]
    return "\n".join(lines) + "\n"


def _budget_section(budget: pd.DataFrame | None) -> list[str]:
    if budget is None or budget.empty:
        return []
    return [
        "## Tuning budget",
        "",
        "df-analyze stops each tuning run at a fixed time limit, so models that "
        "are slower per trial may complete fewer trials. `stopped_by` is `all "
        "trials`, `time limit`, or `early stop` (no improvement for 15 trials "
        "after 50, or an exhausted search grid).",
        "",
        _markdown_table(budget),
        "",
    ]


def load_run(run_outdir: Path) -> tuple[list[dict[str, Any]], pd.DataFrame, dict]:
    files = find_run_files(run_outdir)
    check_label_encoding(pd.read_parquet(files.labels))
    tuned = pd.read_csv(files.tuned_models, index_col=0)
    predictions = json.loads(files.predictions.read_text(encoding="utf-8"))
    options = json.loads(files.options.read_text(encoding="utf-8"))
    return predictions["predictions"], tuned, options
