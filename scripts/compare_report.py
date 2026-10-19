"""Comparison page: two feature sets' Model A side by side, test node by node.

    uv run scripts/compare_report.py --dataset amazon \\
        --a pilot_m1_own --b pilot_m5_own_gnnssl

Writes outputs/<dataset>/compare/<a>__vs__<b>/compare.html and
compare_metrics.json. Both feature sets must have finished the pipeline
through df_analyze_report. Not a pipeline stage: it spans two feature sets
(see src/fraud_detect/report/compare.py). The page contains row-level test
data: share it within the group only, never commit or upload it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fraud_detect.config import (
    DEFAULT_CONFIG,
    ConfigError,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.report import compare
from fraud_detect.runstate import atomic_write_json, atomic_write_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", help="yelpchi or amazon")
    parser.add_argument("--a", required=True, help="feature set A (the baseline)")
    parser.add_argument("--b", required=True, help="feature set B")
    parser.add_argument(
        "--resamples",
        type=int,
        default=compare.N_RESAMPLES,
        help="group-bootstrap resamples (default %(default)s)",
    )
    parser.add_argument(
        "--random-cards",
        type=int,
        default=compare.N_RANDOM_CARDS,
        help="seeded random test nodes with a precomputed card, besides the "
        "galleries (default %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.a == args.b:
        print("error: --a and --b must be different feature sets", file=sys.stderr)
        return 2
    start = time.monotonic()
    try:
        config = with_dataset(load_config(args.config), args.dataset)
        result = compare.build(
            config, args.a, args.b, n_resamples=args.resamples,
            n_random_cards=args.random_cards,
        )  # fmt: skip
    except (ConfigError, compare.CompareError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = get_paths(config).compare_dir(args.a, args.b)
    atomic_write_json(out / "compare_metrics.json", result.metrics)
    atomic_write_text(out / "compare.html", result.html)
    m = result.metrics
    d = m["delta_pr_auc"]
    for side in ("a", "b"):
        s = m[side]
        print(f"{side.upper()} {s['feature_set']:<22} test PR-AUC "
              f"{s['pr_auc']['estimate']:.4f} [{s['pr_auc']['low']:.4f}, "
              f"{s['pr_auc']['high']:.4f}]  AUROC {s['auroc']['estimate']:.4f}  "
              f"Recall@K {s['recall_at_k']:.3f}")  # fmt: skip
    print(
        f"ΔPR-AUC (B − A) {d['estimate']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}], "
        f"B ahead in {d['share_above_zero']:.1%} of {d['n_resamples']:,} resamples"
    )
    size = (out / "compare.html").stat().st_size / 1e6
    print(
        f"Saved {out / 'compare.html'} ({size:.1f} MB) and compare_metrics.json in "
        f"{time.monotonic() - start:.0f} s. Row-level test data: group only."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
