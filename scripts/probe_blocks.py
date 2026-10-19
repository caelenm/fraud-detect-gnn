"""Quick probe of the feature blocks on the training set's grouped CV folds.

    uv run scripts/probe_blocks.py --dataset amazon

Fits one fixed, untuned HistGradientBoostingClassifier per saved CV fold on
own, own+nbr and own+gnnssl (training nodes only) and writes the mean ± SD
held-out PR-AUC to outputs/<dataset>/probe/probe_cv.csv. Build the blocks
first (the neighbour_agg and gnn_embed stages). Never reads test rows; see
src/fraud_detect/models/probe.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from fraud_detect.config import (
    DEFAULT_CONFIG,
    ConfigError,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.models import probe
from fraud_detect.runstate import atomic_write_csv


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
    for path in (p.train_ids, p.cv_folds, p.block_file("own")):
        if not path.is_file():
            print(f"error: missing {path}; run the pipeline's data stages first",
                  file=sys.stderr)  # fmt: skip
            return 2
    start = time.monotonic()
    table = probe.probe(
        p.features_dir, pd.read_csv(p.train_ids), pd.read_csv(p.cv_folds),
        int(config["seed"]),
    )  # fmt: skip
    out = p.probe_dir / "probe_cv.csv"
    atomic_write_csv(out, table)
    print(table.to_string(index=False))
    print(f"Saved {out} ({time.monotonic() - start:.0f} s). Training folds only; "
          "a probe, not a result.")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
