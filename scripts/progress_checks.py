"""Training-only checks for the progress report (decided 2026-10-19).

    uv run scripts/progress_checks.py --dataset amazon

Everything here uses the training nodes of split repeat 0 and their saved
grouped CV folds; no test row is read and nothing is chosen from the results.
It writes three tables to outputs/<dataset>/probe/:

- shortcut_check.csv: how well each feature of the config's `no36` variant
  (issue #36) alone separates the training classes.
- random_graph_control.csv: the probe (fixed HistGradientBoosting,
  models/probe.py) for own + nbr and own + gnnssl built on the REAL graph
  and on a randomly rewired graph with the same degrees, for both variants.
  If the graph gain comes from real structure, it should mostly vanish on
  the random graph.
- embedding_sensitivity.csv: the probe for own + gnnssl over a grid of
  embedding settings (size 16/32/64, at most 100/300 epochs, PCA 4/8/16),
  for both variants. A sensitivity check only: the runs keep the defaults
  (size 64, 300 epochs, PCA 8), fixed before any result.

Needs the gnn extra. GNN embeddings run on the CPU (bit-identical).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
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
from fraud_detect.pipeline import load_edges, split_masks
from fraud_detect.runstate import atomic_write_csv

VARIANT = "no36"
DIMS, EPOCHS, PCA_KS = (16, 32, 64), (100, 300), (4, 8, 16)


def probe_row(own_input, extra, train_ids, folds, seed, **labels) -> dict:
    """Probe CV scores for own_input joined with an extra block (or alone)."""
    features = own_input if extra is None else own_input.merge(extra, on=C.NODE_ID)
    x, y, fold, cols = probe.training_rows(features, train_ids, folds)
    return {**labels, "n_features": len(cols), **probe.cv_scores(x, y, fold, seed)}


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
    from fraud_detect.graph import ssl  # noqa: PLC0415  (needs the gnn extra)

    start = time.monotonic()
    p = get_paths(config)
    seed = int(config["seed"])
    variants = config.get("feature_variants") or {}
    train_ids, folds = pd.read_csv(p.train_ids), pd.read_csv(p.cv_folds)
    own = pd.read_parquet(p.block_file(C.OWN_BLOCK))
    nodes = pd.read_parquet(p.nodes)
    fit_mask, loss_mask = split_masks(nodes)
    n = len(own)
    relations = list(config["datasets"][config["dataset"]]["relations"])
    edges = {r: load_edges(p.relation_file(r)) for r in relations}
    cfg = ssl.SSLConfig.from_config(config["gnn_embed"], seed)
    if cfg.device != "cpu":
        cfg = replace(cfg, device="cpu")  # bit-identical, as for the runs

    # 1. Single-feature check of the variant's columns (issue #36).
    shortcut = probe.single_feature_check(own, train_ids, variants[VARIANT])
    atomic_write_csv(p.probe_dir / "shortcut_check.csv", shortcut)
    print(shortcut.to_string(index=False), flush=True)

    inputs = {"all": own, VARIANT: blocks.own_view(own, f"own_{VARIANT}", variants)}

    # 2. Random-graph control: same degrees, random partners.
    rewired = {
        r: graph.rewire_edges(e, n, seed + i) for i, (r, e) in enumerate(edges.items())
    }
    kept = {r: rewired[r].shape[1] / edges[r].shape[1] for r in relations}
    print("rewired edges kept:", {r: f"{v:.3f}" for r, v in kept.items()}, flush=True)
    rows = []
    for name, own_input in inputs.items():
        common = (train_ids, folds, seed)
        rows.append(probe_row(own_input, None, *common, variant=name, graph="none",
                              block="own only"))  # fmt: skip
        for g, graph_edges in (("real", edges), ("random", rewired)):
            nbr = graph.build_nbr_block(own_input, graph_edges)
            rows.append(probe_row(own_input, nbr, *common, variant=name, graph=g,
                                  block="+ nbr"))  # fmt: skip
            emb = ssl.gnnssl_block(own_input, graph_edges, fit_mask, loss_mask, cfg)
            rows.append(probe_row(own_input, emb.block, *common, variant=name,
                                  graph=g, block="+ gnnssl"))  # fmt: skip
            elapsed = time.monotonic() - start
            print(f"control {name} {g} done ({elapsed:.0f} s)", flush=True)
    control = pd.DataFrame(rows)
    control["edges_kept_after_rewiring"] = " ".join(
        f"{r}:{kept[r]:.3f}" for r in relations
    )
    atomic_write_csv(p.probe_dir / "random_graph_control.csv", control)
    shown = ["variant", "graph", "block", "n_features", "cv_pr_auc_mean", "cv_pr_auc_std"]
    print(control[shown].to_string(index=False), flush=True)

    # 3. Embedding sensitivity (the runs keep the defaults).
    rows = []
    for name, own_input in inputs.items():
        x = ssl.standardized_features(own_input, fit_mask)
        for dim in DIMS:
            for epochs in EPOCHS:
                c = replace(cfg, hidden=dim, out_dim=dim, max_epochs=epochs)
                emb = {r: ssl.embed_relation(x, e, loss_mask, c, r).embedding
                       for r, e in edges.items()}  # fmt: skip
                for k in PCA_KS:
                    block, _ = ssl.pca_block(emb, fit_mask, k, seed)
                    default = (dim, epochs, k) == (64, 300, 8)
                    rows.append(probe_row(own_input, block, train_ids, folds, seed,
                                          variant=name, dim=dim, max_epochs=epochs,
                                          pca_k=k, default=default))  # fmt: skip
                print(f"sensitivity {name} dim {dim} epochs {epochs} done "
                      f"({time.monotonic() - start:.0f} s)", flush=True)  # fmt: skip
    grid = pd.DataFrame(rows)
    atomic_write_csv(p.probe_dir / "embedding_sensitivity.csv", grid)
    for name, frame in grid.groupby("variant"):
        s = frame["cv_pr_auc_mean"]
        d = frame.loc[frame["default"], "cv_pr_auc_mean"].iloc[0]
        rank = int((s > d).sum()) + 1
        print(f"{name}: default {d:.4f}; grid min {s.min():.4f}, median "
              f"{s.median():.4f}, max {s.max():.4f}; "
              f"default ranks {rank} of {len(s)}")  # fmt: skip
    print(f"Saved to {p.probe_dir} in {time.monotonic() - start:.0f} s. Training folds "
          "only; checks, not results.")  # fmt: skip
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
