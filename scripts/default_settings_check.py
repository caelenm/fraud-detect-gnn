"""Default-settings sensitivity check on YelpChi (RESEARCH_PLAN §11, 2026-11-02).

    uv run scripts/default_settings_check.py

Pre-registered before it ran. For each df-analyze setup and split it takes the
default-settings configuration with the highest shared-CV PR-AUC (training
folds only), fits it on the whole training export in df-analyze's environment
(scripts/dfa/default_test.py) and scores the test set. It then applies the
verdict rule of contrasts.py, with the CV p from the same configurations' saved
out-of-fold predictions. Model A stays the best tuned row, and the test set
chooses nothing. As a code check, each tuned Model A is refit the same way and
compared with the pipeline's test PR-AUC. Writes
outputs/yelpchi/compare/gnn_yelpchi_default_settings{,_scores}.csv and .md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gnn_report import split_config
from sklearn.metrics import average_precision_score

from fraud_detect import columns as C
from fraud_detect.config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    ConfigError,
    df_analyze_dir,
    get_paths,
    load_config,
)
from fraud_detect.external import run_df_analyze_script
from fraud_detect.models import contrasts, df_analyze
from fraud_detect.models.model_export import MODEL_NAMES
from fraud_detect.runstate import atomic_write_csv, atomic_write_text

DATASET, N_SPLITS, SEED = "yelpchi", 3, 555
SETUPS = {
    "B": "pilot_m1_own",
    "E1": "pilot_m2_own_nbr",
    "E2": "pilot_m5_own_gnnssl",
    "E7": "pilot_m7_own_nbr_gnnssl",
    "E8": "pilot_m8_own_nbr_gnnsup",
}
# Read as first − second.
CONTRASTS = (("E1", "B"), ("E2", "B"), ("E7", "E1"), ("E8", "E1"),
             ("E2", "E1"), ("E7", "E2"), ("E8", "E7"))  # fmt: skip
CODE_CHECK_TOLERANCE = 0.005
RUNNER = REPO_ROOT / "scripts" / "dfa" / "default_test.py"


def model_a_row(cv: pd.DataFrame, model_a: dict) -> pd.Series:
    """Model A's tuned row of model_selection_cv.csv."""
    names = cv["model_cls"].map(MODEL_NAMES).fillna(cv["model_cls"])
    mask = (
        (cv["settings"] == "tuned")
        & (names == model_a["model"])
        & (cv["selection"] == model_a["selection"])
    )
    if model_a["selection"] == "embed":
        mask &= cv["embed_selector"].fillna("") == (model_a.get("embed_selector") or "")
    if mask.sum() != 1:
        raise ValueError(f"No unique tuned row for Model A {model_a['model']}")
    return cv[mask].iloc[0]


def job_of(row: pd.Series) -> dict:
    selector = row["embed_selector"]
    return {
        "settings": row["settings"],
        "model_cls": row["model_cls"],
        "selection": row["selection"],
        "embed_selector": "" if pd.isna(selector) else str(selector),
    }


def score_split(config: dict, k: int, setup: str, fs: str) -> tuple[dict, pd.Series]:
    """One setup on one split: its scores, and the default row's per-fold CV PR-AUC."""
    cfg = split_config(config, DATASET, k, N_SPLITS)
    p = get_paths({**cfg, "feature_set": fs})
    summary = json.loads((p.model_reports_dir / "model_a.json").read_text("utf-8"))
    run = p.df_analyze_output_dir / summary["run"]
    cv = pd.read_csv(p.model_reports_dir / "model_selection_cv.csv")
    default = contrasts.best_default_row(cv)  # training folds only
    tuned = model_a_row(cv, summary["model_a"])
    out = p.feature_set_dir / "default_settings"
    probs_path = out / "test_probs.parquet"
    if not probs_path.is_file():
        jobs = out / "jobs.json"
        atomic_write_text(jobs, json.dumps([job_of(default), job_of(tuned)], indent=2))
        dfa = cfg["df_analyze"]
        run_df_analyze_script(
            df_analyze_dir(cfg),
            str(RUNNER),
            [
                "--export-dir",
                str(df_analyze.find_export_dir(run).resolve()),
                "--jobs",
                str(jobs.resolve()),
                "--out",
                str(probs_path.resolve()),
                "--seed",
                str(SEED),
            ],  # fmt: skip
            out / "default_test.log",
            dfa.get("commit"),
            python=dfa.get("python"),
        )
    probs = pd.read_parquet(probs_path)
    y_test = pd.read_parquet(p.df_analyze_input_dir / "test.parquet")[C.TARGET].to_numpy()
    y_train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")[
        C.TARGET
    ].to_numpy()

    def test_pr_auc(settings: str) -> float:
        rows = probs[probs["settings"] == settings].sort_values("row")
        if len(rows) != len(y_test):
            raise ValueError(f"{fs} split {k}: {len(rows)} test rows, not {len(y_test)}")
        return float(average_precision_score(y_test, rows["prob"]))

    oof = pd.read_parquet(run / "oof_predictions.parquet")
    folds = contrasts.fold_pr_auc(contrasts.configuration_oof(oof, default), y_train)
    refit = test_pr_auc("tuned")
    pipeline = float(summary["model_a"]["pr_auc"])
    return {
        "setup": setup,
        "feature_set": fs,
        "split_repeat": k,
        "default_model": default["model_cls"],
        "default_selection": default["selection"],
        "n_features": int(default["n_features"]),
        "cv_pr_auc_default": float(default["pr_auc_mean"]),
        "test_pr_auc_default": test_pr_auc("default"),
        "test_pr_auc_model_a": pipeline,
        "test_pr_auc_model_a_refit": refit,
        "code_check_ok": abs(refit - pipeline) <= CODE_CHECK_TOLERANCE,
    }, folds


def contrast_rows(scores: pd.DataFrame, folds: dict, n_folds: int) -> pd.DataFrame:
    test = scores.set_index(["setup", "split_repeat"])["test_pr_auc_default"]
    rows = []
    for b, a in CONTRASTS:
        deltas = [test[b, k] - test[a, k] for k in range(N_SPLITS)]
        fold_diffs = []
        for k in range(N_SPLITS):
            fold_diffs += list((folds[b, k] - folds[a, k]).to_numpy())
        t = contrasts.corrected_t_test(fold_diffs, 1.0 / (n_folds - 1))
        d = np.asarray(deltas)
        rows.append(
            {
                "contrast": f"{b} - {a}",
                **{f"test_delta_split{k}": v for k, v in enumerate(deltas)},
                "test_delta_mean": float(d.mean()),
                "test_delta_sd": float(d.std(ddof=1)),
                "cv_folds": len(fold_diffs),
                "cv_delta_mean": t["mean"],
                "corrected_t": t["t"],
                "p": t["p"],
                "verdict": contrasts.verdict(deltas, t["p"]),
            }
        )
    return pd.DataFrame(rows)


def report(scores: pd.DataFrame, table: pd.DataFrame) -> str:
    means = scores.groupby("setup", sort=False)[
        ["test_pr_auc_model_a", "test_pr_auc_default", "cv_pr_auc_default"]
    ].mean()
    lines = [
        "# Default-settings sensitivity check (YelpChi, 3 splits, pilot runs)",
        "",
        "Pre-registered in docs/RESEARCH_PLAN.md §11 (2026-11-02). Not a verdict on "
        "the ladder: Model A stays the best tuned row.",
        "",
        "| Setup | Test PR-AUC, tuned Model A | Test PR-AUC, default | "
        "CV PR-AUC, default |",
        "|---|---|---|---|",
    ]
    for setup, r in means.iterrows():
        lines.append(f"| {setup} | {r['test_pr_auc_model_a']:.3f} | "
                     f"{r['test_pr_auc_default']:.3f} | "
                     f"{r['cv_pr_auc_default']:.3f} |")  # fmt: skip
    lines += [
        "",
        "| Contrast | Δ split 0 | Δ split 1 | Δ split 2 | Mean ± SD | CV Δ | p | "
        "Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in table.to_dict("records"):
        deltas = " | ".join(f"{r[f'test_delta_split{k}']:+.4f}" for k in range(N_SPLITS))
        lines.append(f"| {r['contrast']} | {deltas} | {r['test_delta_mean']:+.4f} ± "
                     f"{r['test_delta_sd']:.4f} | {r['cv_delta_mean']:+.4f} | "
                     f"{r['p']:.3g} | {r['verdict']} |")  # fmt: skip
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
    rows, folds = [], {}
    for setup, fs in SETUPS.items():
        for k in range(N_SPLITS):
            row, folds[setup, k] = score_split(config, k, setup, fs)
            rows.append(row)
    scores = pd.DataFrame(rows)
    table = contrast_rows(scores, folds, int(config["split"]["cv_folds"]))
    cfg0 = split_config(config, DATASET, 0, N_SPLITS)
    out = get_paths(cfg0).outputs_dir / "compare"
    name = f"gnn_{DATASET}_default_settings"
    atomic_write_csv(out / f"{name}.csv", table)
    atomic_write_csv(out / f"{name}_scores.csv", scores)
    text = report(scores, table)
    atomic_write_text(out / f"{name}.md", text)
    print(text)
    worst = (
        (scores["test_pr_auc_model_a_refit"] - scores["test_pr_auc_model_a"]).abs().max()
    )
    print(f"Code check: largest difference from the pipeline's Model A {worst:.5f}")
    if not scores["code_check_ok"].all():
        print(f"CODE CHECK FAILED (tolerance {CODE_CHECK_TOLERANCE})", file=sys.stderr)
        return 3
    print(f"Saved {out / name}.csv, _scores.csv and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
