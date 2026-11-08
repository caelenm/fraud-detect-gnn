"""Summarise scripts/dfa/shap_explain.py's outputs: aggregate tables and one page.

    uv run scripts/explain_report.py --dataset amazon --feature-set m2_own_nbr_no19 \\
        --audit pilot_m2_own_nbr --split-set pilot_m2_own_nbr_no19

Writes outputs/<dataset>[_repeat<k>]/<feature set>/explain/explain.html (row-level:
share within the group only, never commit or upload) and tables/*.csv, including
tables/db_summary.csv for the results database:

    uv run scripts/results_db.py add --kind explain --csv <that file> ...

Not a pipeline stage: see src/fraud_detect/report/explain.py. --audit adds another
feature set's top columns as a leakage-audit box. --split-set compares that feature
set's importance across split repeats 0, 1 and 2; each needs its own shap section.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from fraud_detect.config import (
    DEFAULT_CONFIG,
    ConfigError,
    Paths,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.models.explain import OUTPUT_DIR
from fraud_detect.report import explain as report
from fraud_detect.runstate import atomic_write_text


def paths_for(args: argparse.Namespace, feature_set: str, repeat: int) -> Paths:
    config = load_config(args.config)
    config["split"]["n_repeats"] = max(int(config["split"]["n_repeats"]), repeat + 1)
    config["split"]["repeat"] = repeat
    config = with_dataset(config, args.dataset)
    config["feature_set"] = feature_set
    return get_paths(config)


def edges_of(p: Paths, folder: Path) -> dict[str, np.ndarray] | None:
    """Edge lists of the relations the influence table covers, if it exists."""
    influence = folder / "neighbour_influence.parquet"
    if not influence.is_file():
        return None
    import pandas as pd  # noqa: PLC0415  (only when there is an influence table)

    relations = (
        pd.read_parquet(influence, columns=["relation"])["relation"].astype(str).unique()
    )
    out = {}
    for relation in relations:
        with np.load(p.relation_file(relation)) as f:
            out[relation] = f["edges"]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--audit", help="feature set shown as a leakage audit")
    parser.add_argument("--split-set", help="feature set compared across splits 0, 1, 2")
    parser.add_argument("--seed", type=int, default=555)
    args = parser.parse_args(argv)
    try:
        config: dict[str, Any] = with_dataset(load_config(args.config), args.dataset)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    correlated = (
        list((config.get("feature_variants") or {}).get("no36", []))
        if args.dataset == "amazon"
        else []
    )
    p = paths_for(args, args.feature_set, args.repeat)
    folder = p.feature_set_dir / OUTPUT_DIR
    try:
        summary = report.summarise(
            folder,
            p.df_analyze_input_dir / "feature_set.json",
            correlated,
            edges_of(p, folder),
            args.seed,
        )
        if args.audit:
            audit = paths_for(args, args.audit, args.repeat)
            audit_folder = audit.feature_set_dir / OUTPUT_DIR
            summary.audit, _, _, _ = report.importance_summary(
                audit_folder,
                audit.df_analyze_input_dir / "feature_set.json",
                correlated,
                args.seed,
            )
            summary.audit_meta = report.read_meta(audit_folder)
        if args.split_set:
            frames = {}
            for repeat in (0, 1, 2):
                other = paths_for(args, args.split_set, repeat)
                table, _, _, _ = report.importance_summary(
                    other.feature_set_dir / OUTPUT_DIR,
                    other.df_analyze_input_dir / "feature_set.json",
                    correlated,
                    args.seed,
                )
                frames[f"{args.split_set} split {repeat}"] = table.set_index("name")[
                    "mean_abs_shap"
                ]
            summary.split_agreement = report.split_agreement(frames)
    except report.ExplainReportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    table = report.write_tables(summary, folder)
    page = folder / "explain.html"
    atomic_write_text(page, report.render_page(summary))
    print(f"Wrote {page}\nWrote {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
