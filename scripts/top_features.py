"""Training-only ranking of the own features (phase 3, Task 5.1).

    uv run scripts/top_features.py --dataset amazon

For each saved grouped CV fold of the training set, the probe's fixed
HistGradientBoosting (src/fraud_detect/models/probe.py) is fit on the other
folds, and each feature's permutation importance is the drop in the held-out
fold's PR-AUC when it is shuffled; importances are averaged over the folds.
Next to it, each feature's univariate AUROC on the training nodes. Writes
outputs/<dataset>/probe/top_features.csv and says whether the top 4 are the
config's `no36` features (issue #36). Never reads a test row; a check, not a
result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from fraud_detect import columns as C
from fraud_detect.config import (
    DEFAULT_CONFIG,
    ConfigError,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.models import probe
from fraud_detect.runstate import atomic_write_csv

TOP_K = 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", help="yelpchi or amazon")
    args = parser.parse_args(argv)
    try:
        config = with_dataset(load_config(args.config), args.dataset)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    p = get_paths(config)
    start = time.monotonic()
    own = pd.read_parquet(p.block_file(C.OWN_BLOCK))
    table = probe.feature_ranking(
        own, pd.read_csv(p.train_ids), pd.read_csv(p.cv_folds), int(config["seed"])
    )
    flagged = set((config.get("feature_variants") or {}).get("no36") or [])
    table["in_no36"] = table["column"].isin(flagged)
    out = p.probe_dir / "top_features.csv"
    atomic_write_csv(out, table)
    shown = ["column", "perm_importance_mean", "perm_importance_sd", "univariate_auroc",
             "rank_permutation", "rank_univariate_auroc", "in_no36"]  # fmt: skip
    print(table[shown].to_string(index=False))
    for rank in ("rank_permutation", "rank_univariate_auroc"):
        top = set(table.loc[table[rank] <= TOP_K, "column"])
        same = "the same as" if top == flagged else "NOT the same as"
        print(f"Top {TOP_K} by {rank}: {', '.join(sorted(top))} ({same} no36)")
    print(f"Saved {out} ({time.monotonic() - start:.0f} s). Training folds only; "
          "a check, not a result.")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
