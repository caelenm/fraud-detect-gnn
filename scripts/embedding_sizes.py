"""How big the GNN embeddings are (phase 3, Task 1.1).

    uv run scripts/embedding_sizes.py --dataset amazon

Reads every finished gnn_embed folder of split repeat 0
(outputs/<dataset>/gnn_embed*/meta.json and raw_<relation>.npy) and every
feature set with a finished Model A, and writes to outputs/<dataset>/gnn_embed/:

- sizes.csv, one row per GNN block and relation: the raw embedding's shape,
  epochs, how many of its dimensions are constant on the training nodes, its
  numerical and effective rank, and the cumulative explained variance of a
  PCA fit on the training nodes at k = 1, 2, 4, ..., 64, with the k needed for
  90 / 95 / 99%;
- sizes_feature_sets.csv, one row per feature set and block: the columns
  df-analyze received and how many of them Model A kept after df-analyze's
  feature selection;
- sizes.md, both tables.

Unsupervised statistics and column counts only; no label is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from fraud_detect import columns as C
from fraud_detect.config import (
    DEFAULT_CONFIG,
    ConfigError,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.pipeline import split_masks
from fraud_detect.report import compare
from fraud_detect.runstate import atomic_write_csv, atomic_write_text

KS = (1, 2, 4, 8, 16, 32, 64)
TARGETS = (0.90, 0.95, 0.99)


def markdown_table(frame: pd.DataFrame) -> str:
    """A plain markdown table (floats to 3 decimals)."""

    def cell(v: object) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "|" + "---|" * len(frame.columns)]  # fmt: skip
    lines += ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in frame.itertuples(index=False)]  # fmt: skip
    return "\n".join(lines)


def embedding_row(z: np.ndarray, fit_mask: np.ndarray) -> dict:
    """Size statistics of one raw embedding, from its training rows."""
    train = z[fit_mask].astype(np.float64)
    centred = train - train.mean(axis=0)
    s = np.linalg.svd(centred, compute_uv=False)
    p = s / s.sum()
    pca = PCA(n_components=min(z.shape[1], len(train)), svd_solver="full").fit(train)
    cum = np.cumsum(pca.explained_variance_ratio_)
    row = {
        "n_nodes": z.shape[0],
        "dim": z.shape[1],
        "constant_dims_on_train": int((train.std(axis=0) < 1e-8).sum()),
        "numerical_rank": int(np.linalg.matrix_rank(centred)),
        "effective_rank": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
    }
    for k in KS:
        if k <= len(cum):
            row[f"var_k{k}"] = float(cum[k - 1])
    for t in TARGETS:
        row[f"k_for_{round(t * 100)}pct"] = int(np.searchsorted(cum, t) + 1)
    return row


def feature_set_rows(config: dict, feature_set: str) -> list[dict]:
    """Columns per block that df-analyze received, and how many Model A kept."""
    side = compare.load_side(config, feature_set)
    p = get_paths({**config, "feature_set": feature_set})
    info = json.loads((p.df_analyze_input_dir / "feature_set.json").read_text("utf-8"))
    given = [c["block_column"] for c in info["columns"].values()]
    used = set(side.used_columns)
    rows = []
    for block in info["blocks"]:
        stored = C.OWN_BLOCK if block.startswith(C.OWN_BLOCK) else block
        cols = [c for c in given if c.split(C.BLOCK_SEPARATOR)[0] == stored]
        rows.append(
            {
                "feature_set": feature_set,
                "block": block,
                "columns_given": len(cols),
                "columns_kept_by_model_a": len(used & set(cols)),
                "model_a": f"{side.summary['model_a']['model']} / "
                f"{side.summary['model_a']['selection']}",
            }
        )
    return rows


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
    fit_mask, _ = split_masks(pd.read_parquet(p.nodes))

    rows = []
    for meta_path in sorted(p.outputs_dir.glob("gnn_embed*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for relation, log in meta["relations"].items():
            z = np.load(meta_path.parent / meta["raw_embeddings"][relation])
            rows.append(
                {
                    "block": meta["block"],
                    "relation": relation,
                    "epochs_run": log.get("epochs_run", 0),
                    "pca_columns_in_block": len(log["explained_variance_ratio"]),
                    **embedding_row(z, fit_mask),
                }
            )
    sizes = pd.DataFrame(rows)

    fs_rows = []
    for feature_set in config["feature_sets"]:
        if (get_paths({**config, "feature_set": feature_set}).model_reports_dir
                / "model_a.json").is_file():  # fmt: skip
            try:
                fs_rows += feature_set_rows(config, feature_set)
            except compare.CompareError as e:
                print(f"skipped {feature_set}: {e}")
    sets = pd.DataFrame(fs_rows)

    out = p.gnn_embed_dir
    atomic_write_csv(out / "sizes.csv", sizes)
    atomic_write_csv(out / "sizes_feature_sets.csv", sets)
    text = "\n".join(
        [
            f"# GNN embedding sizes ({config['dataset']}, split repeat 0)",
            "",
            "Raw embeddings, from their training rows (PCA fit on training nodes; "
            "`var_k<k>` is the variance the first k components keep). Effective "
            "rank: exp of the entropy of the normalised singular values.",
            "",
            markdown_table(sizes),
            "",
            "Columns df-analyze received per block, and how many Model A kept after "
            "df-analyze's feature selection.",
            "",
            markdown_table(sets),
            "",
        ]
    )
    atomic_write_text(out / "sizes.md", text)
    print(text)
    print(f"Saved {out / 'sizes.md'}, sizes.csv and sizes_feature_sets.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
