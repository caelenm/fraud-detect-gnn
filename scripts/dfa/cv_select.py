"""Score every tuned df-analyze model with one shared cross-validation.

THIS SCRIPT RUNS INSIDE DF-ANALYZE'S ENVIRONMENT, not this project's:

    uv run --directory <df-analyze> --python '>=3.13.11,<3.14' \
        python <this file> --export-dir <.../results/test00> --out cv.csv \
        --oof-out oof.parquet --folds-file folds.csv

df-analyze's tuning scores cannot be compared across models: most models are
tuned on 5-fold balanced accuracy of hard predictions, while GANDALF is scored
on a single validation split with its own code path, and its internal folds
are random rather than grouped. To choose the best Model A fairly, this
script takes each tuned configuration (model, feature set, tuned
hyperparameters, from prediction_results_00.json), refits it with
df-analyze's own `refit_tuned` on the SAME saved grouped folds of the
TRAINING set (--folds-file: one fold number per training row, in the row
order of df-analyze's X_train export, made by the pipeline from
cv_folds.csv), and scores the held-out fold from predicted probabilities. The
test set is never read.

The out-of-fold probabilities of every configuration are saved too
(--oof-out), so the report stage can choose a decision threshold on training
data only.

Each configuration is also scored with df-analyze's DEFAULT hyperparameters
(no tuning) on the same folds, so the effect of tuning can be reported. Only
the tuned rows are used to choose Model A.
"""

from __future__ import annotations

import argparse
import ast
import faulthandler
import importlib
import json
import os
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


def job_key(entry: dict, settings: str) -> tuple[str, str, str, str]:
    return (
        str(entry["model_cls"]),
        str(entry["selection"]),
        str(entry.get("embed_select_model") or ""),
        settings,
    )


KEY_COLUMNS = ("model_cls", "selection", "embed_selector", "settings")


def read_folds(path: Path, n_rows: int) -> np.ndarray:
    """Each training row's saved fold, checked against the export's size."""
    folds = pd.read_csv(path)["fold"].to_numpy(dtype=np.int64)
    if len(folds) != n_rows:
        raise SystemExit(
            f"{path} has {len(folds)} folds for {n_rows} training rows; the "
            "pipeline writes one per row of df-analyze's X_train export."
        )
    if len(np.unique(folds)) < 2:
        raise SystemExit(f"{path} has fewer than 2 folds")
    return folds


def previous_oof(path: Path, done: set) -> list[pd.DataFrame]:
    """Out-of-fold predictions of configurations kept on --resume."""
    if not path.is_file() or not done:
        return []
    frame = pd.read_parquet(path)
    frame["embed_selector"] = frame["embed_selector"].fillna("")
    keys = frame[list(KEY_COLUMNS)].astype(str).apply(tuple, axis=1)
    return [frame[keys.isin(done)]]


def row_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row["model_cls"]),
        str(row["selection"]),
        str(row.get("embed_selector") or ""),
        str(row["settings"]),
    )


def previous_rows(path: Path) -> list[dict]:
    """Configurations already scored successfully in an earlier, interrupted
    run (failed ones are retried)."""
    if not path.is_file():
        return []
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    frame["embed_selector"] = frame["embed_selector"].fillna("")
    frame["error"] = frame["error"].fillna("")
    return frame[frame["error"] == ""].to_dict("records")


def eta_text(seconds: list[float], remaining: int) -> str:
    if not seconds:
        return ""
    left = sum(seconds) / len(seconds) * remaining
    return f"(about {left / 60:.0f} min left)"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--oof-out", type=Path, required=True, help="out-of-fold probabilities (.parquet)"
    )
    parser.add_argument(
        "--folds-file",
        type=Path,
        required=True,
        help="CSV with a `fold` column: each training row's fold, in X_train order",
    )
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument(
        "--only", nargs="*", default=None, help="model class names (for smoke tests)"
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=["tuned", "default"],
        default=["tuned", "default"],
        help="hyperparameters to score: tuned, and/or df-analyze's defaults",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep configurations already scored in --out and skip them",
    )
    parser.add_argument(
        "--config-timeout",
        type=int,
        default=0,
        help="seconds allowed per configuration (all folds) before the script "
        "dumps stack traces and exits; 0 disables the watchdog",
    )
    args = parser.parse_args()

    X = pd.read_csv(args.export_dir / "X_train_00.csv")
    y = pd.read_csv(args.export_dir / "y_train_00.csv", index_col=0).iloc[:, 0]
    y = y.astype(int).reset_index(drop=True).rename("target")
    predictions = json.loads(
        (args.export_dir / "prediction_results_00.json").read_text(encoding="utf-8")
    )["predictions"]
    num_classes = int(y.nunique())
    fold_of_row = read_folds(args.folds_file, len(X))
    folds = [
        (np.flatnonzero(fold_of_row != k), np.flatnonzero(fold_of_row == k))
        for k in sorted(np.unique(fold_of_row))
    ]
    print(
        f"CV on {len(X)} training rows, {len(folds)} saved grouped folds, "
        f"{len(predictions)} configs"
    )

    # Every tuned fit (which choose Model A) runs before any default-settings
    # fit, and the table is rewritten after each configuration, so a crash,
    # hang or shutdown never loses finished scores. --resume keeps them.
    jobs = [
        (entry, settings)
        for settings in args.settings
        for entry in predictions
        if not args.only or entry["model_cls"] in args.only
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = previous_rows(args.out) if args.resume else []
    done = {row_key(r) for r in rows}
    oof = previous_oof(args.oof_out, done) if args.resume else []
    if done:
        print(f"Resuming: {len(done)} of {len(jobs)} configurations already scored")
    session_seconds: list[float] = []
    for i, (entry, settings) in enumerate(jobs, start=1):
        name = entry["model_cls"]
        sel = selection_name(entry)
        if job_key(entry, settings) in done:
            continue
        print(f"[{i}/{len(jobs)}] {name} · {sel} · {settings} settings "
              f"{eta_text(session_seconds, len(jobs) - i + 1)}", flush=True)  # fmt: skip
        # Keys as in prediction_results, so the pipeline can join the tables.
        row: dict = {
            "model_cls": name,
            "selection": entry["selection"],
            "embed_selector": entry.get("embed_select_model") or "",
            "settings": settings,
        }
        start = time.perf_counter()
        if args.config_timeout > 0:
            # Watchdog: if this configuration hangs, print every thread's stack
            # (to diagnose it) and exit, instead of blocking the pipeline forever.
            faulthandler.dump_traceback_later(args.config_timeout, exit=True)
        try:
            # "default": df-analyze's own default hyperparameters, i.e. the same
            # model and feature set with no tuning, as the before-tuning baseline.
            params = tuned_params(entry) if settings == "tuned" else {}
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
            prob_oof = np.full(len(X), np.nan)
            for k, (idx_tr, idx_va) in enumerate(folds):
                seed_everything(args.seed + k)
                model = (
                    cls(num_classes=num_classes) if name in NEEDS_NUM_CLASSES else cls()
                )
                model.refit_tuned(X=Xs.iloc[idx_tr], y=y.iloc[idx_tr], tuned_args=params)
                prob = np.asarray(model.predict_proba(Xs.iloc[idx_va]))[:, 1]
                per_fold.append(fold_scores(y.iloc[idx_va].to_numpy(), prob))
                prob_oof[idx_va] = prob
            frame = pd.DataFrame(per_fold)
            for metric in frame.columns:
                row[f"{metric}_mean"] = float(frame[metric].mean())
                row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
            row["fold_pr_auc"] = json.dumps([round(v, 6) for v in frame["pr_auc"]])
            row["error"] = ""
            oof.append(
                pd.DataFrame(
                    {
                        **{k: row[k] for k in KEY_COLUMNS},
                        "row": np.arange(len(X)),
                        "fold": fold_of_row,
                        "prob": prob_oof,
                    }
                )
            )
        except Exception as e:  # record and continue with the other models
            traceback.print_exc()
            row["error"] = f"{type(e).__name__}: {e}"
        faulthandler.cancel_dump_traceback_later()
        row["seconds"] = round(time.perf_counter() - start, 1)
        print(
            f"{name:<22} {sel:<14} {settings:<8} "
            f"PR-AUC {row.get('pr_auc_mean', float('nan')):.4f} "
            f"({row['seconds']} s){'  ERROR ' + row['error'] if row['error'] else ''}",
            flush=True,
        )
        rows.append(row)
        session_seconds.append(row["seconds"])
        # Out-of-fold probabilities first, then the table that marks the
        # configuration as done; both atomic, so a crash never leaves a
        # half-written file or a scored configuration without its predictions.
        if oof:
            tmp = args.oof_out.with_name(args.oof_out.stem + ".tmp.parquet")
            pd.concat(oof, ignore_index=True).to_parquet(tmp, index=False)
            os.replace(tmp, args.oof_out)
        tmp = args.out.with_name(args.out.name + ".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, args.out)

    print(f"Saved {len(rows)} cross-validated configurations to {args.out}")
    # Per-model errors are recorded in the table; the pipeline decides what to do.
    return 0 if rows else 5


if __name__ == "__main__":
    sys.exit(main())
