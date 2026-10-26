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
(c) Added 2026-10-26: does the column map read from CARE-GNN's
    amazon_preprocess.py at a64ff75 (it keeps GraphRfi columns 0-18 and
    30-35, so f19 = GraphRfi column 30, a vote count) hold on our data?
    Identities that hold if the map is right (RESEARCH_PLAN §11, 2026-10-26):
    f02 + ... + f06 = f00 (star counts sum to the number of ratings),
    f07 + ... + f11 = 1 (star shares), f12 = f07 + f08 and f13 = f10 + f11
    (negative and positive shares), f14 = entropy of f07..f11,
    f16 >= f15, f18 >= f17 (max, median, mean, min rating),
    f22 = 1 exactly when f20 = 0 (one review date), f24 in {-1, 0, 1}.

Only training users and aggregates are used and written:
outputs/<dataset>/probe/vote_ratio_check.csv, column_map_check.csv and
feature_identity.md. Nothing is chosen from these numbers by the pipeline.
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


def column_map_identities(train: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Check (c): one row per identity, with the share of users it holds for."""

    def f(k: int) -> np.ndarray:
        return train[f"own__f{k:02d}"].to_numpy(float)

    shares = np.column_stack([f(k) for k in range(7, 12)])
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(shares > 0, shares * np.log(shares), 0.0)
    entropy_e = -plogp.sum(axis=1)
    # f14 looks stored to 4 decimals, so its identities get a 1e-4 tolerance.
    checks: list[tuple[str, np.ndarray, np.ndarray, float]] = [
        ("f02 + ... + f06 = f00", sum(f(k) for k in range(2, 7)), f(0), tol),
        ("f07 + ... + f11 = 1", shares.sum(axis=1), np.ones(len(train)), tol),
        ("f12 = f07 + f08", f(7) + f(8), f(12), tol),
        ("f13 = f10 + f11", f(10) + f(11), f(13), tol),
        ("f14 = entropy(f07..f11), natural log, within 1e-4", entropy_e, f(14), 1e-4),
        (
            "f14 = entropy(f07..f11), log2, within 1e-4",
            entropy_e / np.log(2),
            f(14),
            1e-4,
        ),
        ("f07 = f02 / f00", f(2) / np.where(f(0) == 0, np.nan, f(0)), f(7), tol),
    ]
    rows = []
    for name, lhs, rhs, eps in checks:
        err = np.abs(lhs - rhs)
        ok = np.isfinite(err)
        rows.append(
            {
                "identity": name,
                "n_users": int(ok.sum()),
                "share_holding": float((err[ok] <= eps).mean()) if ok.any() else np.nan,
                "max_abs_error": float(err[ok].max()) if ok.any() else np.nan,
            }
        )
    inequalities = [
        (
            "f16 >= f15 >= f17 (max >= median >= min)",
            (f(16) >= f(15) - tol) & (f(15) >= f(17) - tol),
        ),
        (
            "f16 >= f18 >= f17 (max >= mean >= min)",
            (f(16) >= f(18) - tol) & (f(18) >= f(17) - tol),
        ),
        ("f22 = 1 exactly when f20 = 0", (f(22) == 1) == (f(20) == 0)),
        ("f22 in {0, 1}", np.isin(f(22), [0, 1])),
        ("f24 in {-1, 0, 1}", np.isin(f(24), [-1, 0, 1])),
        (
            "f19 is a non-negative integer",
            (f(19) >= 0) & np.isclose(f(19), np.round(f(19))),
        ),
        (
            "f00, f02..f06 are non-negative integers",
            np.all(
                [
                    (f(k) >= 0) & np.isclose(f(k), np.round(f(k)))
                    for k in (0, 2, 3, 4, 5, 6)
                ],
                axis=0,
            ),
        ),
    ]
    for name, holds in inequalities:
        rows.append(
            {
                "identity": name,
                "n_users": len(train),
                "share_holding": float(np.mean(holds)),
                "max_abs_error": np.nan,
            }
        )
    return pd.DataFrame(rows)


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

    # (c) the column map from CARE-GNN's amazon_preprocess.py
    if config["dataset"] == "amazon":
        cmap = column_map_identities(train)
        atomic_write_csv(p.probe_dir / "column_map_check.csv", cmap)
        lines += [
            "## (c) Column-map identities (CARE-GNN amazon_preprocess.py at a64ff75)",
            "",
            "| identity | users | share holding | max abs error |",
            "|---|---|---|---|",
        ]
        for r in cmap.itertuples():
            err = "" if pd.isna(r.max_abs_error) else f"{r.max_abs_error:.3g}"
            lines.append(
                f"| {r.identity} | {r.n_users:,} | {r.share_holding:.2%} | {err} |"
            )
        lines.append("")
    text = "\n".join(lines)
    atomic_write_text(p.probe_dir / "feature_identity.md", text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
