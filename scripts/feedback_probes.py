"""Training-CV probes for the professor's feedback (docs/PHASE3_FEEDBACK_PLAN.md).

    uv run scripts/feedback_probes.py --dataset amazon                 # all three
    uv run scripts/feedback_probes.py --dataset amazon --task gnnrand

Every probe is the fixed, untuned HistGradientBoosting of
src/fraud_detect/models/probe.py on the saved grouped CV folds of split
repeat 0's training nodes. Blocks are built in memory (nothing under data/ is
written), from the own features and the edges only, never the labels; GNN
embeddings run on the CPU. Tables go to outputs/<dataset>/probe/:

- `only36` (Task 5.3) -> only36_probe.csv: own, own + nbr and own + gnnssl for
  all features, without the #36 features (no36), with only them (only36) and
  with only the training-ranked top 4 (top4, scripts/top_features.py).
- `gnnrand` (Task 3a) -> gnnrand_probe.csv: own + gnnrand (the gnnssl encoder,
  untrained) for 5 initialisation seeds, next to own, own + nbr and
  own + gnnssl, for all features and no36.
- `sizes` (Task 1.2) -> embedding_size_sweep.csv: own + gnnssl for DGI sizes
  16 / 32 / 64 and k = 1, 2, 4, 8, 16, 32 PCA components per relation, plus the
  raw embedding without PCA, for all features and no36. The pre-registered
  choice (plan Task 1.3) is marked at DGI size 64: the smallest k whose mean CV
  PR-AUC is within one standard error (SD / sqrt(folds)) of the best k's.

Checks on training folds only, not results: no test row is read.
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

TASKS = ("only36", "gnnrand", "sizes")
INIT_SEEDS = 5  # gnnrand: seeds seed, seed + 1, ...
DIMS, PCA_KS = (16, 32, 64), (1, 2, 4, 8, 16, 32)
DEFAULT_DIM = 64
RAW = "raw"


def probe_row(own_input, extra, train_ids, folds, seed, **labels) -> dict:
    """Probe CV scores for own_input joined with an extra block (or alone)."""
    features = own_input if extra is None else own_input.merge(extra, on=C.NODE_ID)
    x, y, fold, cols = probe.training_rows(features, train_ids, folds)
    return {**labels, "n_features": len(cols), **probe.cv_scores(x, y, fold, seed)}


def raw_block(
    embeddings: dict[str, np.ndarray], name: str = C.GNNSSL_BLOCK
) -> pd.DataFrame:
    """The raw embedding of every relation as columns (no PCA)."""
    n = len(next(iter(embeddings.values())))
    out: dict[str, np.ndarray] = {C.NODE_ID: np.arange(n, dtype=np.int64)}
    for relation, z in embeddings.items():
        for j in range(z.shape[1]):
            out[f"{name}{C.BLOCK_SEPARATOR}{relation}_z{j + 1:02d}"] = z[:, j]
    return pd.DataFrame(out)


def choose_k(frame: pd.DataFrame) -> int | str:
    """The smallest k whose mean CV PR-AUC is within one standard error of
    the best k's (raw counts as k = the embedding size)."""
    size = frame["k"].map(lambda k: DEFAULT_DIM if k == RAW else int(k))
    best = frame.loc[frame["cv_pr_auc_mean"].idxmax()]
    se = best["cv_pr_auc_std"] / np.sqrt(best["n_folds"])
    ok = frame[frame["cv_pr_auc_mean"] >= best["cv_pr_auc_mean"] - se]
    return ok.loc[size[ok.index].idxmin(), "k"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", help="yelpchi or amazon")
    parser.add_argument("--task", nargs="+", choices=TASKS, default=list(TASKS))
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
    fit_mask, loss_mask = split_masks(pd.read_parquet(p.nodes))
    relations = list(config["datasets"][config["dataset"]]["relations"])
    edges = {r: load_edges(p.relation_file(r)) for r in relations}
    cfg = replace(ssl.SSLConfig.from_config(config["gnn_embed"], seed), device="cpu")
    inputs = {"all": own}
    for v in ("no36", "only36", "top4"):
        if v in variants:
            inputs[v] = blocks.own_view(own, f"own_{v}", variants)
    common = (train_ids, folds, seed)

    cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def dgi(variant: str, dim: int) -> dict[str, np.ndarray]:
        """DGI embeddings per relation, trained once per (variant, size)."""
        if (variant, dim) not in cache:
            x = ssl.standardized_features(inputs[variant], fit_mask)
            c = replace(cfg, hidden=dim, out_dim=dim)
            cache[variant, dim] = {
                r: ssl.embed_relation(x, e, loss_mask, c, r).embedding
                for r, e in edges.items()
            }
            took = time.monotonic() - start
            print(f"  DGI {variant} size {dim} trained ({took:.0f} s)", flush=True)
        return cache[variant, dim]

    def baseline_rows(variant: str) -> list[dict]:
        own_input = inputs[variant]
        gnnssl, _ = ssl.pca_block(dgi(variant, DEFAULT_DIM), fit_mask,
                                  cfg.pca_components, seed)  # fmt: skip
        return [
            probe_row(own_input, None, *common, variant=variant, block="own"),
            probe_row(own_input, graph.build_nbr_block(own_input, edges), *common,
                      variant=variant, block="+ nbr"),
            probe_row(own_input, gnnssl, *common, variant=variant, block="+ gnnssl"),
        ]  # fmt: skip

    shown = ["variant", "block", "n_features", "cv_pr_auc_mean", "cv_pr_auc_std"]
    if "only36" in args.task:
        rows = [r for v in inputs for r in baseline_rows(v)]
        table = pd.DataFrame(rows)
        atomic_write_csv(p.probe_dir / "only36_probe.csv", table)
        print(table[shown].to_string(index=False), flush=True)

    if "gnnrand" in args.task:
        rows = []
        for variant in ("all", "no36"):
            rows += baseline_rows(variant)
            for i in range(INIT_SEEDS):
                rand = ssl.gnnrand_block(inputs[variant], edges, fit_mask,
                                         replace(cfg, seed=seed + i))  # fmt: skip
                rows.append(probe_row(inputs[variant], rand.block, *common,
                                      variant=variant, block="+ gnnrand",
                                      init_seed=seed + i))  # fmt: skip
        table = pd.DataFrame(rows)
        atomic_write_csv(p.probe_dir / "gnnrand_probe.csv", table)
        print(table[[*shown, "init_seed"]].to_string(index=False))
        rand = table[table["block"] == "+ gnnrand"].groupby("variant")["cv_pr_auc_mean"]
        for variant, s in rand:
            print(f"{variant}: own + gnnrand over {len(s)} init seeds: "
                  f"{s.mean():.4f} ± {s.std(ddof=1):.4f} (min {s.min():.4f}, "
                  f"max {s.max():.4f})", flush=True)  # fmt: skip

    if "sizes" in args.task:
        rows = []
        for variant in ("all", "no36"):
            for dim in DIMS:
                emb = dgi(variant, dim)
                for k in [k for k in PCA_KS if k <= dim]:
                    block, _ = ssl.pca_block(emb, fit_mask, k, seed)
                    labels = {"variant": variant, "dgi_size": dim, "k": k}
                    rows.append(probe_row(inputs[variant], block, *common, **labels))
                rows.append(probe_row(inputs[variant], raw_block(emb), *common,
                                      variant=variant, dgi_size=dim, k=RAW))  # fmt: skip
        table = pd.DataFrame(rows)
        table["default"] = (table["dgi_size"] == DEFAULT_DIM) & (
            table["k"] == cfg.pca_components
        )
        table["chosen"] = False
        for variant in ("all", "no36"):
            at64 = table[
                (table["variant"] == variant) & (table["dgi_size"] == DEFAULT_DIM)
            ]
            k = choose_k(at64)
            table.loc[at64.index[at64["k"] == k], "chosen"] = True
            print(f"{variant}: pre-registered choice at DGI size {DEFAULT_DIM}: k = {k}")
        atomic_write_csv(p.probe_dir / "embedding_size_sweep.csv", table)
        print(table[["variant", "dgi_size", "k", "n_features", "cv_pr_auc_mean",
                     "cv_pr_auc_std", "chosen"]].to_string(index=False))  # fmt: skip

    print(f"Saved to {p.probe_dir} in {time.monotonic() - start:.0f} s. Training folds "
          "only; checks, not results.")  # fmt: skip
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
