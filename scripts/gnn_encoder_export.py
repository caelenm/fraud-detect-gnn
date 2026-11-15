"""Rebuild one feature set's DGI encoders and save their weights for explanations.

    uv run --extra gnn scripts/gnn_encoder_export.py --dataset yelpchi \\
        --feature-set pilot_m5_own_gnnssl

The gnn_embed stage saves embeddings, not weights. Its CPU run is seeded and
bit-identical, so this retrains each relation's encoder with the saved settings and
stops unless the result reproduces the saved embeddings and PCA columns. It writes
outputs/<dataset>[_repeat<k>]/<feature set>/explain/encoder_weights.npz (with the
standardised own features, so it is row-level and never committed) for the
neighbours_encoder section of scripts/dfa/shap_explain.py, which runs in df-analyze's
environment without torch. Not a pipeline stage, so no finished run or bundle reruns.
No label is read: the masks come from the split column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detect import columns as C
from fraud_detect.config import (
    DEFAULT_CONFIG,
    dataset_config,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.graph import ssl
from fraud_detect.models.explain import OUTPUT_DIR
from fraud_detect.pipeline import read_nodes, split_masks
from fraud_detect.runstate import atomic_write_json

# Both sides are float32 from the same seeded CPU run, so this allows only for a
# different machine's rounding.
TOLERANCE = 1e-4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["split"]["n_repeats"] = max(int(config["split"]["n_repeats"]), args.repeat + 1)
    config["split"]["repeat"] = args.repeat
    config = with_dataset(config, args.dataset)
    config["feature_set"] = args.feature_set
    p = get_paths(config)
    block = C.GNNSSL_BLOCK
    relations = list(dataset_config(config)["relations"])
    meta = json.loads(
        (p.raw_embedding(relations[0], block).parent / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    if meta["own_columns_left_out"]:
        print("error: only the plain gnnssl block is supported", file=sys.stderr)
        return 2
    cfg = ssl.SSLConfig(**{**meta["config"], "device": "cpu"})
    if cfg.layers != 2 or cfg.pca_components == "raw":
        print("error: only a 2-layer encoder with a PCA is supported", file=sys.stderr)
        return 2
    fit_mask, loss_mask = split_masks(read_nodes(p))
    own = pd.read_parquet(p.block_file(C.OWN_BLOCK))
    x = ssl.standardized_features(own, fit_mask)
    saved_block = pd.read_parquet(p.block_file(block))
    arrays: dict[str, np.ndarray] = {"own_scaled": x.astype(np.float32)}
    report: dict[str, dict[str, float]] = {}
    for relation in relations:
        with np.load(p.relation_file(relation)) as f:
            edges = f["edges"]
        result = ssl.embed_relation(x, edges, loss_mask, cfg, relation)
        saved = np.load(p.raw_embedding(relation, block))
        z_gap = float(np.abs(result.embedding - saved).max())
        pca = ssl.fit_pca(saved, fit_mask, int(cfg.pca_components), cfg.seed)
        width = max(2, len(str(cfg.pca_components)))
        names = [
            f"{block}{C.BLOCK_SEPARATOR}{relation}_pc{j + 1:0{width}d}"
            for j in range(int(cfg.pca_components))
        ]
        pca_gap = float(
            np.abs(pca.transform(saved) - saved_block[names].to_numpy(dtype=float)).max()
        )
        print(f"{relation}: embedding gap {z_gap:.3g}, PCA gap {pca_gap:.3g}", flush=True)
        if z_gap > TOLERANCE or pca_gap > TOLERANCE:
            print(
                f"error: {relation} does not reproduce the saved block, so these are "
                "not the encoder behind the model. Stop and ask.",
                file=sys.stderr,
            )
            return 3
        state = result.encoder_state
        arrays |= {
            f"{relation}/w1": state["weights.0.weight"],
            f"{relation}/b1": state["biases.0"],
            f"{relation}/s1": state["activations.0.weight"],
            f"{relation}/w2": state["weights.1.weight"],
            f"{relation}/b2": state["biases.1"],
            f"{relation}/s2": state["activations.1.weight"],
            f"{relation}/pca_components": pca.components_,
            f"{relation}/pca_mean": pca.mean_,
        }
        report[relation] = {"embedding_gap": z_gap, "pca_gap": pca_gap}
    out = p.feature_set_dir / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "encoder_weights.tmp.npz"
    np.savez(tmp, **arrays)
    tmp.replace(out / "encoder_weights.npz")
    atomic_write_json(
        out / "encoder_weights.json", {"relations": report, "seed": cfg.seed}
    )
    print(f"Wrote {out / 'encoder_weights.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
