"""Ladder contrasts across split repeats, with the pre-registered verdict.

    uv run scripts/repeat_summary.py --dataset amazon --repeats 3 --prefix pilot_
    uv run scripts/repeat_summary.py --dataset amazon --name gnnrand_summary \\
        --pairs m5r_own_gnnrand:m5_own_gnnssl m1_own:m5r_own_gnnrand

For each contrast (by default m2 − m1, m5 − m1, m5 − m2, with all features
and _no36; or each --pairs A:B, read as B − A, feature-set names without the
prefix) it reads the test ΔPR-AUC of every split from compare_report.py's
compare_metrics.json (build those first, with --repeat k), and runs the
Nadeau-Bengio corrected paired t-test over all splits' CV folds from the
saved out-of-fold predictions (training data only). It writes
outputs/<dataset>/compare/<name>.csv and .md (default name repeat_summary).
See src/fraud_detect/models/contrasts.py for the decision rule.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detect import columns as C
from fraud_detect.config import (
    DEFAULT_CONFIG,
    ConfigError,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.models import contrasts
from fraud_detect.runstate import atomic_write_csv, atomic_write_text

RUNGS = {"m1": "m1_own", "m2": "m2_own_nbr", "m5": "m5_own_gnnssl"}
CONTRASTS = (("m1", "m2"), ("m1", "m5"), ("m2", "m5"))
VARIANTS = ("", "_no36")


def repeat_config(config: dict, repeat: int, n_repeats: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["split"]["n_repeats"] = n_repeats
    cfg["split"]["repeat"] = repeat
    return cfg


def side(cfg: dict, feature_set: str) -> tuple[dict, pd.Series]:
    """Model A's summary and its per-fold CV PR-AUC for one feature set."""
    p = get_paths({**cfg, "feature_set": feature_set})
    summary = json.loads((p.model_reports_dir / "model_a.json").read_text("utf-8"))
    run = p.df_analyze_output_dir / summary["run"]
    oof = pd.read_parquet(run / "oof_predictions.parquet")
    y = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")[C.TARGET].to_numpy()
    rows = contrasts.model_a_oof(oof, summary["model_a"])
    return summary, contrasts.fold_pr_auc(rows, y)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", help="yelpchi or amazon")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefix", default="pilot_", help="feature-set name prefix")
    parser.add_argument(
        "--pairs",
        nargs="+",
        metavar="A:B",
        help="contrasts B − A, feature-set names without the prefix (default: the "
        "ladder's m2 − m1, m5 − m1 and m5 − m2, with all features and _no36)",
    )
    parser.add_argument("--name", default="repeat_summary", help="output file stem")
    args = parser.parse_args(argv)
    try:
        config = with_dataset(load_config(args.config), args.dataset)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    n_folds = int(config["split"]["cv_folds"])
    if args.pairs:
        if not all(pair.count(":") == 1 for pair in args.pairs):
            print("error: --pairs takes A:B items", file=sys.stderr)
            return 2
        contrast_list = [
            ("-", f"{b} - {a}", a, b) for a, b in (pair.split(":") for pair in args.pairs)
        ]
    else:
        contrast_list = [
            ("all features" if not variant else "without #36", f"{b_key} - {a_key}",
             f"{RUNGS[a_key]}{variant}", f"{RUNGS[b_key]}{variant}")
            for variant in VARIANTS for a_key, b_key in CONTRASTS
        ]  # fmt: skip
    rows = []
    for variant_label, contrast_label, a_name, b_name in contrast_list:
        a, b = f"{args.prefix}{a_name}", f"{args.prefix}{b_name}"
        test_deltas, fold_diffs, pr_a, pr_b = [], [], [], []
        for k in range(args.repeats):
            cfg = repeat_config(config, k, args.repeats)
            path = get_paths(cfg).compare_dir(a, b) / "compare_metrics.json"
            if not path.is_file():
                print(f"error: missing {path}; run compare_report.py --repeat {k}",
                      file=sys.stderr)  # fmt: skip
                return 2
            m = json.loads(path.read_text("utf-8"))
            test_deltas.append(m["delta_pr_auc"]["estimate"])
            pr_a.append(m["a"]["pr_auc"]["estimate"])
            pr_b.append(m["b"]["pr_auc"]["estimate"])
            (_, folds_a), (_, folds_b) = side(cfg, a), side(cfg, b)
            fold_diffs += list((folds_b - folds_a).to_numpy())
        test = contrasts.corrected_t_test(fold_diffs, 1.0 / (n_folds - 1))
        d = np.asarray(test_deltas)
        rows.append(
            {
                "variant": variant_label,
                "contrast": contrast_label,
                "a": a,
                "b": b,
                **{f"test_delta_split{k}": v for k, v in enumerate(test_deltas)},
                "test_delta_mean": float(d.mean()),
                "test_delta_sd": float(d.std(ddof=1)),
                "test_pr_auc_a_mean": float(np.mean(pr_a)),
                "test_pr_auc_a_sd": float(np.std(pr_a, ddof=1)),
                "test_pr_auc_b_mean": float(np.mean(pr_b)),
                "test_pr_auc_b_sd": float(np.std(pr_b, ddof=1)),
                "cv_folds": len(fold_diffs),
                "cv_delta_mean": test["mean"],
                "corrected_t": test["t"],
                "p": test["p"],
                "verdict": contrasts.verdict(test_deltas, test["p"]),
            }
        )
    table = pd.DataFrame(rows)
    out = get_paths(repeat_config(config, 0, args.repeats)).outputs_dir / "compare"
    atomic_write_csv(out / f"{args.name}.csv", table)
    lines = [
        f"# Ladder contrasts over {args.repeats} splits ({config['dataset']}, "
        f"{args.prefix or 'full'} runs)",
        "",
        "Test ΔPR-AUC (B − A) per split; mean ± SD across splits; Nadeau-Bengio "
        f"corrected paired t-test over {args.repeats} × {n_folds} grouped CV folds "
        "(training data only). Verdict rule fixed before the runs "
        "(docs/RESEARCH_PLAN.md §11). Each side's Model A was chosen by the same "
        "folds, which makes both sides' CV scores slightly optimistic.",
        "",
        "| Variant | Contrast | "
        + " | ".join(f"Δ split {k}" for k in range(args.repeats))
        + " | Mean ± SD | CV Δ | p | Verdict |",
        "|" + "---|" * (args.repeats + 6),
    ]
    for r in table.to_dict("records"):
        deltas = " | ".join(
            f"{r[f'test_delta_split{k}']:+.4f}" for k in range(args.repeats)
        )
        lines.append(
            f"| {r['variant']} | {r['contrast']} | {deltas} | "
            f"{r['test_delta_mean']:+.4f} ± {r['test_delta_sd']:.4f} | "
            f"{r['cv_delta_mean']:+.4f} | {r['p']:.3g} | "
            f"{r['verdict']} |"
        )
    text = "\n".join(lines) + "\n"
    atomic_write_text(out / f"{args.name}.md", text)
    print(text)
    print(f"Saved {out / f'{args.name}.csv'} and {args.name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
