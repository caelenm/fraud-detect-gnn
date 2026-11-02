"""YelpChi's df-analyze setups at up to 100 trials against the pilot's 10.

    uv run scripts/trials_check.py

Pre-registered in RESEARCH_PLAN §11 (2026-11-02). Reads what the pipeline saved
at both budgets, per setup and split: Model A's test and shared-CV PR-AUC, the
best default row's CV PR-AUC (its settings do not depend on the budget, so any
difference is run-to-run noise), and how each search stopped. Nothing is chosen
here. Writes outputs/yelpchi/compare/gnn_yelpchi_trials{.csv,.md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from gnn_report import split_config

from fraud_detect.config import DEFAULT_CONFIG, ConfigError, get_paths, load_config
from fraud_detect.models import contrasts
from fraud_detect.runstate import atomic_write_csv, atomic_write_text

DATASET, N_SPLITS = "yelpchi", 3
SETUPS = {
    "B": "m1_own",
    "E1": "m2_own_nbr",
    "E2": "m5_own_gnnssl",
    "E7": "m7_own_nbr_gnnssl",
    "E8": "m8_own_nbr_gnnsup",
}
BUDGETS = {"t10": "pilot_", "t100": "t100_"}


def budget_columns(config: dict, k: int, fs: str, budget: str) -> dict:
    """One setup on one split at one budget."""
    cfg = split_config(config, DATASET, k, N_SPLITS)
    reports = get_paths({**cfg, "feature_set": fs}).model_reports_dir
    summary = json.loads((reports / "model_a.json").read_text("utf-8"))
    cv = pd.read_csv(reports / "model_selection_cv.csv")
    searches = pd.read_csv(reports / "tuning_budget.csv")
    searches = searches[~searches["model"].str.startswith("Dummy")]
    return {
        f"model_a_{budget}": summary["model_a"]["model"],
        f"selection_{budget}": summary["model_a"]["selection"],
        f"test_pr_auc_{budget}": float(summary["model_a"]["pr_auc"]),
        f"cv_pr_auc_{budget}": float(summary["model_a"]["cv_pr_auc"]),
        f"cv_pr_auc_default_{budget}": float(
            contrasts.best_default_row(cv)["pr_auc_mean"]
        ),
        f"trials_median_{budget}": float(searches["trials_completed"].median()),
        f"searches_all_trials_{budget}": int(
            (searches["stopped_by"] == "all trials").sum()
        ),
        f"searches_early_stop_{budget}": int(
            (searches["stopped_by"] == "early stop").sum()
        ),
        f"searches_time_limit_{budget}": int(
            (searches["stopped_by"] == "time limit").sum()
        ),
        f"tuning_minutes_{budget}": float(searches["elapsed_s"].sum() / 60),
    }


def report(frame: pd.DataFrame) -> str:
    means = frame.groupby("setup", sort=False).mean(numeric_only=True)
    lines = [
        "# Up to 100 trials against the pilot's 10 (YelpChi, 3 splits)",
        "",
        "Pre-registered in docs/RESEARCH_PLAN.md §11 (2026-11-02). Not a verdict on "
        "the ladder. Means over the splits. Default CV: the same settings at both "
        "budgets, so a gap between its two columns is run-to-run noise.",
        "",
        "| Setup | Test, 10 | Test, 100 | Test Δ | CV, 10 | CV, 100 | Default CV, 10 | "
        "Default CV, 100 | Median trials, 100 | Time-limit stops, 100 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for setup, r in means.iterrows():
        lines.append(
            f"| {setup} | {r['test_pr_auc_t10']:.3f} | {r['test_pr_auc_t100']:.3f} | "
            f"{r['test_pr_auc_t100'] - r['test_pr_auc_t10']:+.3f} | "
            f"{r['cv_pr_auc_t10']:.3f} | {r['cv_pr_auc_t100']:.3f} | "
            f"{r['cv_pr_auc_default_t10']:.3f} | {r['cv_pr_auc_default_t100']:.3f} | "
            f"{r['trials_median_t100']:.0f} | "
            f"{r['searches_time_limit_t100'] * N_SPLITS:.0f} |"
        )
    lines += ["", "| Setup | Split | Model A, 10 | Model A, 100 |", "|---|---|---|---|"]
    for r in frame.to_dict("records"):
        lines.append(
            f"| {r['setup']} | {r['split_repeat']} | "
            f"{r['model_a_t10']} ({r['selection_t10']}) | "
            f"{r['model_a_t100']} ({r['selection_t100']}) |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    rows = []
    for setup, name in SETUPS.items():
        for k in range(N_SPLITS):
            row = {"setup": setup, "split_repeat": k}
            for budget, prefix in BUDGETS.items():
                row |= budget_columns(config, k, f"{prefix}{name}", budget)
            rows.append(row)
    frame = pd.DataFrame(rows)
    cfg0 = split_config(config, DATASET, 0, N_SPLITS)
    out = get_paths(cfg0).outputs_dir / "compare"
    name = f"gnn_{DATASET}_trials"
    atomic_write_csv(out / f"{name}.csv", frame)
    text = report(frame)
    atomic_write_text(out / f"{name}.md", text)
    print(text)
    print(f"Saved {out / name}.csv and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
