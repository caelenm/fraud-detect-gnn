"""RQ2 null-graph probe: how much of the nbr gain survives an exact
degree-preserving rewiring (2026-10-26; RESEARCH_PLAN §11).

    uv run scripts/null_graph_probe.py --dataset amazon --variant no19

On split repeat 0's training nodes and saved grouped CV folds, the fixed probe
of src/fraud_detect/models/probe.py scores own features (B) alone, + deg (N1,
log1p degree per relation), + nbr on the real graph (E1) and + nbr on a
rewired graph in which every node keeps exactly its degree per relation (N2:
features/graph.py swap_rewire, about 10 double-edge swaps per edge), for 5
rewiring seeds. The gain splits into degree (N1 - B), neighbour columns on a
random graph (N2 - N1) and real structure (E1 - N2). Blocks are built in
memory from the features and edges only, never the labels. Writes
outputs/<dataset>/probe/null_graph_probe_<variant>.csv; a training-CV check,
not a result: no test row is read.
"""

from __future__ import annotations

import argparse
import sys
import time
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
from fraud_detect.features import blocks, graph
from fraud_detect.models import probe
from fraud_detect.pipeline import load_edges
from fraud_detect.runstate import atomic_write_csv

REWIRE_SEEDS = 5


def probe_row(own_input, extra, train_ids, folds, seed, **labels) -> dict:
    """Probe CV scores for own_input joined with an extra block (or alone)."""
    features = own_input if extra is None else own_input.merge(extra, on=C.NODE_ID)
    x, y, fold, cols = probe.training_rows(features, train_ids, folds)
    return {**labels, "n_features": len(cols), **probe.cv_scores(x, y, fold, seed)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", help="yelpchi or amazon")
    parser.add_argument("--variant", help="feature variant, e.g. no19 (default: all)")
    args = parser.parse_args(argv)
    try:
        config = with_dataset(load_config(args.config), args.dataset)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    start = time.monotonic()
    p = get_paths(config)
    seed = int(config["seed"])
    variant = args.variant or "all"
    own = pd.read_parquet(p.block_file(C.OWN_BLOCK))
    if args.variant:
        own = blocks.own_view(own, f"{C.OWN_BLOCK}_{args.variant}",
                              config.get("feature_variants") or {})  # fmt: skip
    n = len(own)
    train_ids, folds = pd.read_csv(p.train_ids), pd.read_csv(p.cv_folds)
    relations = list(config["datasets"][config["dataset"]]["relations"])
    edges = {r: load_edges(p.relation_file(r)) for r in relations}
    common = (train_ids, folds, seed)

    rows = [
        probe_row(own, None, *common, block="own (B)"),
        probe_row(own, graph.build_deg_block(own, edges), *common, block="+ deg (N1)"),
        probe_row(own, graph.build_nbr_block(own, edges), *common, block="+ nbr (E1)"),
    ]
    print(f"B, N1, E1 done ({time.monotonic() - start:.0f} s)", flush=True)
    for i in range(REWIRE_SEEDS):
        rewired, stats = {}, {}
        for j, (r, e) in enumerate(edges.items()):
            t0 = time.monotonic()
            rewired[r], swaps = graph.swap_rewire(e, n, seed + 100 * i + j)
            new = rewired[r].astype(np.int64)
            kept = np.isin(new[0] * n + new[1], e[0].astype(np.int64) * n + e[1]).mean()
            degree = np.bincount(e.ravel(), minlength=n)
            same_degrees = np.array_equal(degree, np.bincount(new.ravel(), minlength=n))
            if not same_degrees or new.shape != e.shape:
                print(f"error: rewiring changed degrees or edges on {r}", file=sys.stderr)
                return 1
            stats |= {f"swaps_per_edge_{r}": swaps / e.shape[1],
                      f"real_edges_kept_{r}": kept,
                      f"rewire_seconds_{r}": time.monotonic() - t0}  # fmt: skip
        nbr = graph.build_nbr_block(own, rewired)
        rows.append(probe_row(own, nbr, *common, block="+ nbr, rewired (N2)",
                              rewire_seed=seed + 100 * i, **stats))  # fmt: skip
        shown = ", ".join(f"{k} {v:.3f}" for k, v in stats.items())
        print(f"N2 seed {i} done ({time.monotonic() - start:.0f} s): {shown}", flush=True)
    table = pd.DataFrame(rows)
    table.insert(0, "variant", variant)
    out = p.probe_dir / f"null_graph_probe_{variant}.csv"
    atomic_write_csv(out, table)

    score = table.groupby("block", sort=False)["cv_pr_auc_mean"]
    b, n1, e1 = (score.mean()[k] for k in ("own (B)", "+ deg (N1)", "+ nbr (E1)"))
    n2 = score.get_group("+ nbr, rewired (N2)")
    print(table[["block", "rewire_seed", "n_features", "cv_pr_auc_mean",
                 "cv_pr_auc_std"]].to_string(index=False))  # fmt: skip
    print(
        f"{variant}: B {b:.4f}; N1 {n1:.4f}; N2 {n2.mean():.4f} ± {n2.std(ddof=1):.4f} "
        f"over {len(n2)} seeds; E1 {e1:.4f}"
    )
    print(
        f"Split of E1 - B = {e1 - b:+.4f}: degree (N1 - B) {n1 - b:+.4f}, "
        f"columns on a random graph (N2 - N1) {n2.mean() - n1:+.4f}, "
        f"real structure (E1 - N2) {e1 - n2.mean():+.4f}"
    )
    print(f"Saved {out} ({time.monotonic() - start:.0f} s). Training folds only; "
          "a check, not a result.")  # fmt: skip
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
