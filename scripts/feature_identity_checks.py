"""Issue #36: local checks of what f13, f15, f18 and f19 are (training rows only).

    uv run scripts/feature_identity_checks.py --dataset amazon

The checks listed in docs/RESEARCH_PLAN.md §12, step 0 (a) and (b), written
down before they were run:

(a) Is f19 a count of unhelpful votes? If some other column c is the
    helpful-vote count, c / (c + f19) reproduces the label rule (< 0.2 =
    fraud, > 0.8 = benign) for almost every training user.
(b) Do f15, f18 and f13 look like rating statistics? f15 on the half-star
    grid {1, 1.5, ..., 5} (a median rating), f18 within [1, 5] (a mean
    rating), f13 within [0, 1] (a share), and f15 and f18 strongly
    correlated (Spearman).

Only training users and aggregates are used and written:
outputs/<dataset>/probe/vote_ratio_check.csv and feature_identity.md.
Nothing is chosen from these numbers by the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fraud_detect import columns as C
from fraud_detect.config import (
    DEFAULT_CONFIG,
    ConfigError,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.models import probe
from fraud_detect.runstate import atomic_write_csv, atomic_write_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", help="amazon")
    args = parser.parse_args(argv)
    try:
        config = with_dataset(load_config(args.config), args.dataset)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    p = get_paths(config)
    own = pd.read_parquet(p.block_file(C.OWN_BLOCK))
    train_ids = pd.read_csv(p.train_ids)
    train = own.set_index(C.NODE_ID).loc[train_ids[C.NODE_ID].to_numpy()]

    # (a) the vote-ratio test for f19
    ratio = probe.vote_ratio_check(own, train_ids, "own__f19")
    atomic_write_csv(p.probe_dir / "vote_ratio_check.csv", ratio)

    # (b) rating-statistic identities
    f13, f15, f18 = (train[f"own__f{k}"].to_numpy(float) for k in (13, 15, 18))
    on_grid = np.isclose(f15 * 2, np.round(f15 * 2)) & (f15 >= 1) & (f15 <= 5)
    rho = stats.spearmanr(f15, f18).statistic
    lines = [
        f"# Issue #36 feature-identity checks ({config['dataset']}, "
        f"{len(train):,} training users)",
        "",
        "Pre-registered in docs/RESEARCH_PLAN.md §12, step 0 (a) and (b). "
        "Training rows and aggregates only.",
        "",
        "## (a) c / (c + f19) against the label rule",
        "",
        "Best five columns c by AUROC of the ratio for 'benign' (1.0 = the "
        "ratio reproduces the label exactly):",
        "",
        "| column | AUROC | fraud with ratio < 0.2 | benign with ratio > 0.8 | "
        "no votes | min(c + f19) |",
        "|---|---|---|---|---|---|",
    ]
    for r in ratio.head(5).itertuples():
        lines.append(
            f"| {r.column} | {r.ratio_auroc_benign:.3f} | {r.fraud_below_low:.1%} | "
            f"{r.benign_above_high:.1%} | {r.share_with_no_votes:.1%} | "
            f"{r.min_total:g} |"
        )
    lines += [
        "",
        "## (b) Rating-statistic identities",
        "",
        f"- f15 on the half-star grid 1..5: {on_grid.mean():.1%} of users; "
        f"distinct values {sorted(set(np.round(f15, 4)))[:12]}",
        f"- f18 range: [{f18.min():.4g}, {f18.max():.4g}] (a mean rating lies in [1, 5])",
        f"- f13 range: [{f13.min():.4g}, {f13.max():.4g}] (a share lies in [0, 1])",
        f"- Spearman(f15, f18) = {rho:.3f}",
        "",
    ]
    text = "\n".join(lines)
    atomic_write_text(p.probe_dir / "feature_identity.md", text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
