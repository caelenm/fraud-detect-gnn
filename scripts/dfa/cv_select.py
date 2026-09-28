"""Score every tuned df-analyze model with one shared cross-validation.

THIS SCRIPT RUNS INSIDE DF-ANALYZE'S ENVIRONMENT, not this project's:

    uv run --directory <df-analyze> --python '>=3.13.11,<3.14' \
        python <this file> --export-dir <.../results/test00> --out cv.csv

df-analyze's tuning scores cannot be compared across models: most models are
tuned on 5-fold balanced accuracy of hard predictions, while GANDALF is scored
on a single validation split with its own code path. To choose the best
Model A fairly, this script takes each tuned configuration (model, feature
set, tuned hyperparameters, from prediction_results_00.json), refits it with
df-analyze's own `refit_tuned` on the SAME stratified folds of the TRAINING
set, and scores the held-out fold from predicted probabilities. The test set
is never read.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

# Import df-analyze from its source tree (see embed_on_device.py).
DFA_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(DFA_ROOT / "src"))
sys.path.insert(1, str(DFA_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

MODEL_MODULES = (
    "catboost", "dummy", "gandalf", "knn", "lgbm", "linear", "mlp", "svm",
)  # fmt: skip
# Models whose constructor takes the number of classes (see df-analyze's
# hypertune.evaluate_tuned).
NEEDS_NUM_CLASSES = {"GandalfEstimator", "MLPEstimator"}


def find_model_class(name: str) -> type:
    for module in MODEL_MODULES:
        mod = importlib.import_module(f"df_analyze.models.{module}")
        cls = getattr(mod, name, None)
        if isinstance(cls, type):
            return cls
    raise KeyError(f"No df-analyze model class named {name!r}")


def selection_name(entry: dict) -> str:
    """Feature-set name as used in df-analyze's result tables."""
    if entry["selection"] == "embed" and entry.get("embed_select_model"):
        return f"embed_{entry['embed_select_model']}"
    return str(entry["selection"])


def tuned_params(entry: dict) -> dict:
    """Tuned hyperparameters; prediction_results stores them as a string."""
    params = entry.get("params")
    if params is None:
        raise ValueError("df-analyze recorded no tuned parameters (tuning failed)")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = ast.literal_eval(params)  # Python dict repr
    if not isinstance(params, dict):
        raise ValueError(f"Unreadable tuned parameters: {entry.get('params')!r}")
    return params


def fold_scores(y_true: np.ndarray, prob_pos: np.ndarray) -> dict[str, float]:
    """Scores from P(fraud). bal_acc uses the 0.5 cut-off, as df-analyze does."""
    return {
        "pr_auc": float(average_precision_score(y_true, prob_pos)),
        "auroc": float(roc_auc_score(y_true, prob_pos)),
        "bal_acc": float(balanced_accuracy_score(y_true, (prob_pos >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y_true, prob_pos)),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument(
        "--only", nargs="*", default=None, help="model class names (for smoke tests)"
    )
    args = parser.parse_args()

    X = pd.read_csv(args.export_dir / "X_train_00.csv")
    y = pd.read_csv(args.export_dir / "y_train_00.csv", index_col=0).iloc[:, 0]
    y = y.astype(int).reset_index(drop=True).rename("target")
    predictions = json.loads(
        (args.export_dir / "prediction_results_00.json").read_text(encoding="utf-8")
    )["predictions"]
    num_classes = int(y.nunique())
    folds = list(
        StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(
            X, y
        )
    )
    print(f"CV on {len(X)} training rows, {args.folds} folds, {len(predictions)} configs")

    rows = []
    for entry in predictions:
        name = entry["model_cls"]
        if args.only and name not in args.only:
            continue
        sel = selection_name(entry)
        # Keys as in prediction_results, so the pipeline can join the tables.
        row: dict = {
            "model_cls": name,
            "selection": entry["selection"],
            "embed_selector": entry.get("embed_select_model") or "",
        }
        start = time.perf_counter()
        try:
            params = tuned_params(entry)
            cls = find_model_class(name)
            cols = entry["selected_cols"]
            missing = [c for c in cols if c not in X.columns]
            if missing:
                raise ValueError(
                    f"{len(missing)} selected columns missing: {missing[:5]}"
                )
            Xs = X[cols]
            row["n_features"] = len(cols)
            per_fold = []
            for k, (idx_tr, idx_va) in enumerate(folds):
                seed_everything(args.seed + k)
                model = (
                    cls(num_classes=num_classes) if name in NEEDS_NUM_CLASSES else cls()
                )
                model.refit_tuned(X=Xs.iloc[idx_tr], y=y.iloc[idx_tr], tuned_args=params)
                prob = np.asarray(model.predict_proba(Xs.iloc[idx_va]))[:, 1]
                per_fold.append(fold_scores(y.iloc[idx_va].to_numpy(), prob))
            frame = pd.DataFrame(per_fold)
            for metric in frame.columns:
                row[f"{metric}_mean"] = float(frame[metric].mean())
                row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
            row["fold_pr_auc"] = json.dumps([round(v, 6) for v in frame["pr_auc"]])
            row["error"] = ""
        except Exception as e:  # record and continue with the other models
            traceback.print_exc()
            row["error"] = f"{type(e).__name__}: {e}"
        row["seconds"] = round(time.perf_counter() - start, 1)
        print(
            f"{name:<22} {sel:<14} PR-AUC {row.get('pr_auc_mean', float('nan')):.4f} "
            f"({row['seconds']} s){'  ERROR ' + row['error'] if row['error'] else ''}",
            flush=True,
        )
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} cross-validated configurations to {args.out}")
    # Per-model errors are recorded in the table; the pipeline decides what to do.
    return 0 if rows else 5


if __name__ == "__main__":
    sys.exit(main())
