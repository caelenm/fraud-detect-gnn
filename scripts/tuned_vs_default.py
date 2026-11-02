"""Tuned against default settings in the shared CV, per setup and split.

    uv run scripts/tuned_vs_default.py

The select_model stage scores every classifier and selection on the saved
grouped folds with both tuned and default settings (model_selection_cv.csv).
This takes the best PR-AUC of each kind, to tell whether a setup's shortfall
comes from its columns or from its tuning. Training folds only: it reads no
test data, and nothing is chosen here (Model A stays the best tuned row).
Writes outputs/<dataset>/compare/gnn_<dataset>_tuned_vs_default.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from gnn_report import DATASETS, SETUPS, split_config

from fraud_detect.config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    ConfigError,
    get_paths,
    load_config,
)
from fraud_detect.runstate import atomic_write_text


def best_rows(config: dict, dataset: str, n: int) -> list[dict]:
    rows = []
    for name, (sid, _) in SETUPS.items():
        fs = f"pilot_{name}"
        for k in range(n):
            cfg = split_config(config, dataset, k, n)
            path = (
                get_paths({**cfg, "feature_set": fs}).model_reports_dir
                / "model_selection_cv.csv"
            )
            if not path.is_file():  # GNN rungs have no df-analyze table
                continue
            table = pd.read_csv(path).dropna(subset=["pr_auc_mean"])
            for settings, group in table.groupby("settings"):
                best = group.loc[group["pr_auc_mean"].idxmax()]
                rows.append(
                    {
                        "setup": sid,
                        "feature_set": fs,
                        "split_repeat": k,
                        "settings": settings,
                        "model": best["model_cls"],
                        "selection": best["selection"],
                        "n_features": int(best["n_features"]),
                        "cv_pr_auc": float(best["pr_auc_mean"]),
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for dataset, n in DATASETS.items():
        frame = pd.DataFrame(best_rows(config, dataset, n))
        if frame.empty:
            continue
        out = REPO_ROOT / "outputs" / dataset / "compare"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"gnn_{dataset}_tuned_vs_default.csv"
        atomic_write_text(path, frame.to_csv(index=False))
        table = frame.pivot_table(
            index="setup", columns="settings", values="cv_pr_auc", sort=False
        )
        print(f"{dataset} ({n} splits), best CV PR-AUC, mean over splits:")
        print(table.round(3).to_string())
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
