"""Where a trained GAT (E10) puts its attention: aggregate statistics.

Refits one of E10's models (the fold model of the first seed, with the
rung's chosen setting and fit seed) and reads its attention weights per
layer, relation and head. GAT fits are not bit-identical between processes
(the backward pass of the per-edge score gather uses a parallel scatter-add,
and CUDA runs differ anyway), so the refit is an equivalent model, not the
same one: its held-out fold's PR-AUC must be within REFIT_TOLERANCE of that
fold's score in gnn_grid.csv, and both are reported.

    uv run scripts/gat_attention.py --dataset amazon --split 0

Writes outputs/<dataset>[_repeat<k>]/<feature set>/attention/:
- attention_summary.csv: aggregate rows only (one per layer, relation and
  head); can go into the results database (--kind probe).
- attention_meta.json: the setting, the check, and what each column means.
- attention_nodes.parquet: per-node statistics (row-level: local only,
  never committed or shared outside the group).
- attention_edges.parquet and attention_targets.parquet (row-level, local
  only; for the dashboard's /attention page): for every test node, per layer
  and relation, its --top-k incoming neighbours by head-averaged weight (with
  each head's weight), and the node's degree, the weight those neighbours
  carry, its attention focus, how much the layer leans on the relation, and
  the refit model's P(fraud). No label is written to either file.

Statistics, per (layer, relation, head), over nodes with >= 2 neighbours in
that relation:
- entropy_ratio: the attention's entropy over log(degree); 1 is uniform
  attention, which is what E5's mean computes, and lower is more focused.
- max_alpha / uniform_max: the largest weight against 1 / degree.
Per (layer, relation), head-averaged:
- relation_to_root: ||the relation's term|| / ||the root term||, averaged
  over nodes with a neighbour: how much the layer leans on that relation.
- fraud_share_attention / fraud_share_uniform, by target group: among
  neighbours whose training label is known, the share of attention on fraud
  neighbours against their share of those neighbours (what uniform weights
  give), over targets with both a fraud and a benign labelled neighbour.
  Labels here are training labels only, used to describe the model, never
  as an input; test labels are not read.

Attention weights describe where the model puts weight, not why it decides
(Jain & Wallace, 2019); report them that way.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from fraud_detect import columns as C
from fraud_detect.config import (
    DEFAULT_CONFIG,
    apply_overrides,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.models import gnn_outputs
from fraud_detect.pipeline import (
    Context,
    _import_supervised,
    _train_config,
    gnn_data,
    gnn_edges,
    gnn_rung_summary,
)
from fraud_detect.runstate import (
    atomic_write_csv,
    atomic_write_parquet,
    atomic_write_text,
)

# How far the refit's held-out fold PR-AUC may be from the rung's saved one.
REFIT_TOLERANCE = 0.005
# Neighbours kept per test node, layer and relation for the dashboard's map.
TOP_K = 12


def top_edges(
    targets: np.ndarray, indptr: np.ndarray, src: np.ndarray, alpha: np.ndarray, k: int
) -> tuple[list[np.ndarray], list[int]]:
    """Per target: the positions (into the dst-sorted edge list) of its k
    incoming edges with the largest head-averaged weight, largest first, and
    its in-degree. alpha is (E, heads)."""
    mean = alpha.mean(axis=1)
    picks, degrees = [], []
    for t in targets:
        lo, hi = int(indptr[t]), int(indptr[t + 1])
        order = np.argsort(-mean[lo:hi], kind="stable")[:k]
        picks.append(lo + order)
        degrees.append(hi - lo)
    return picks, degrees


def per_target(values: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Sum of edge values (E, ...) per target node."""
    out = values.new_zeros((n, *values.shape[1:]))
    return out.index_add_(0, dst, values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", type=int, default=0, help="split repeat (0-2)")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--feature-set", default="pilot_g10_gat")
    parser.add_argument("--fold", type=int, default=0, help="which fold model to refit")
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help="neighbours kept per test node, layer and relation",
    )  # fmt: skip
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="config overrides, as given to run.py")  # fmt: skip
    args = parser.parse_args(argv)

    config = apply_overrides(
        with_dataset(load_config(args.config), args.dataset),
        [f"split.n_repeats={args.n_repeats}", f"split.repeat={args.split}",
         f"feature_set={args.feature_set}", *args.set],
    )  # fmt: skip
    paths = get_paths(config)
    out_dir = paths.model_reports_dir.parent / "attention"
    ctx = Context(config=config, paths=paths, run_dir=out_dir)
    sup = _import_supervised()
    spec = config["gnn_models"][args.feature_set]
    cfg = _train_config(ctx, sup, spec)
    if cfg.model != "gat":
        print(f"error: {args.feature_set} is not a GAT rung", file=sys.stderr)
        return 2
    summary = gnn_rung_summary(ctx, args.feature_set)  # the finished rung
    setting = sup.Setting(**summary["gnn"]["setting"])
    torch.set_num_threads(torch.get_num_threads())

    start = time.monotonic()
    data = gnn_data(ctx, "own")
    g = sup.graph_input(data.x, gnn_edges(ctx, spec["graph"], len(data.x)),
                        spec.get("norm", "mean"), cfg.device, cfg.model)  # fmt: skip
    held = data.folds == args.fold
    seed = sup.fit_seed(ctx.seed, args.fold)
    fit_nodes = data.train_nodes[~held]
    res = sup._fit_rows(g, fit_nodes, data.y_train[~held], data.groups[~held],
                        setting, cfg, seed, keep_model=True)  # fmt: skip
    layers, logits = sup.gat_attention(res.model, g)
    if not np.allclose(logits.numpy(), res.logits, atol=1e-5):
        print(
            "error: the attention read-out does not reproduce the model", file=sys.stderr
        )
        return 1

    # The refit is the rung's own fold model: same held-out PR-AUC as the grid.
    held_prob = sup.sigmoid(res.logits)[data.train_nodes[held]]
    fold_ap = float(average_precision_score(data.y_train[held], held_prob))
    grid_path = paths.model_reports_dir / gnn_outputs.GNN_GRID
    check: dict = {"refit_fold_pr_auc": fold_ap}
    if grid_path.is_file():
        grid = pd.read_csv(grid_path).set_index("setting")
        saved = float(grid.loc[setting.key, f"fold{args.fold}_pr_auc"])
        check.update(saved_fold_pr_auc=saved, identical=bool(abs(saved - fold_ap) < 1e-9),
                     matches=bool(abs(saved - fold_ap) <= REFIT_TOLERANCE))  # fmt: skip
    else:
        check.update(saved_fold_pr_auc=None, matches=None,
                     note="no gnn_grid.csv (setting_from rung); not checked")  # fmt: skip
    if check["matches"] is False:
        saved = check["saved_fold_pr_auc"]
        msg = f"refit fold PR-AUC {fold_ap:.6f} is more than {REFIT_TOLERANCE} from"
        print(f"error: {msg} the saved {saved:.6f}", file=sys.stderr)
        return 1

    # Training labels only (describing the model, never an input).
    n = len(data.x)
    label = torch.full((n,), -1, dtype=torch.long)
    label[torch.from_numpy(np.array(data.train_nodes))] = torch.from_numpy(
        np.array(data.y_train, dtype=np.int64)
    )
    group = np.full(n, "other (test or unlabelled)", dtype=object)
    group[data.train_nodes[data.y_train == 1]] = "train fraud"
    group[data.train_nodes[data.y_train == 0]] = "train benign"

    rows, node_rows = [], {C.NODE_ID: np.arange(n)}
    test_nodes = np.asarray(data.test_nodes, dtype=np.int64)
    target_rows: list[dict] = []
    edge_parts: list[pd.DataFrame] = []
    for li, layer in enumerate(layers):
        for r, rel in enumerate(g.relations):
            edges, alpha = g.edge_index[r], layer.alpha[r]
            deg = torch.diff(edges.indptr)
            many = deg >= 2
            log_deg = torch.log(deg.clamp(min=2).double())
            heads = alpha.shape[1]
            for h in [*range(heads), "mean"]:
                a = (alpha.mean(1) if h == "mean" else alpha[:, h]).double()
                ent = per_target(-(a * torch.log(a.clamp(min=1e-30))), edges.dst, n)
                amax = torch.zeros(n, dtype=torch.float64).scatter_reduce_(
                    0, edges.dst, a, reduce="amax", include_self=True
                )
                ratio = (ent / log_deg)[many]
                if not len(ratio):
                    continue  # no node with two neighbours in this relation
                rows.append({
                    "layer": li + 1, "relation": rel, "head": str(h),
                    "n_edges_directed": int(len(edges.src)),
                    "n_nodes_deg2plus": int(many.sum()),
                    "mean_degree_deg2plus": float(deg[many].double().mean()),
                    "entropy_ratio": float(ratio.mean()),
                    "entropy_ratio_p10": float(ratio.quantile(0.1)),
                    "max_alpha": float(amax[many].mean()),
                    "uniform_max": float((1.0 / deg[many].double()).mean()),
                })  # fmt: skip
                if h == "mean":
                    node_rows[f"l{li + 1}_{rel}_entropy_ratio"] = np.where(
                        many.numpy(), (ent / log_deg).numpy(), np.nan
                    )
                    node_rows[f"l{li + 1}_{rel}_max_alpha"] = np.where(
                        deg.numpy() > 0, amax.numpy(), np.nan
                    )
            # Head-averaged: how much the relation counts, and fraud attention.
            a = alpha.mean(1).double()
            has = deg >= 1
            rel_root = (layer.relation_norm[r] / layer.root_norm.clamp(min=1e-12))[has]
            lab = label[edges.src]
            known = (lab >= 0).double()
            fraud = (lab == 1).double()
            att_known = per_target(a * known, edges.dst, n)
            att_fraud = per_target(a * fraud, edges.dst, n)
            n_known = per_target(known, edges.dst, n)
            n_fraud = per_target(fraud, edges.dst, n)
            mixed = ((n_fraud >= 1) & (n_known - n_fraud >= 1)).numpy()
            share_att = (att_fraud / att_known.clamp(min=1e-30)).numpy()
            share_uni = (n_fraud / n_known.clamp(min=1)).numpy()
            for grp in ("train fraud", "train benign", "other (test or unlabelled)"):
                sel = mixed & (group == grp)
                rows.append({
                    "layer": li + 1, "relation": rel, "head": "mean",
                    "target_group": grp, "n_targets_mixed": int(sel.sum()),
                    "fraud_share_attention": float(share_att[sel].mean())
                    if sel.any() else np.nan,
                    "fraud_share_uniform": float(share_uni[sel].mean())
                    if sel.any() else np.nan,
                    "relation_to_root": float(rel_root.mean()),
                })  # fmt: skip
            # The dashboard's map: each test node's strongest incoming edges.
            alpha_np = alpha.double().numpy()
            focus = (per_target(-(a * torch.log(a.clamp(min=1e-30))), edges.dst, n)
                     / log_deg).numpy()  # fmt: skip
            node_rel_root = (
                layer.relation_norm[r] / layer.root_norm.clamp(min=1e-12)
            ).numpy()
            picks, degrees = top_edges(
                test_nodes, edges.indptr.numpy(), edges.src.numpy(), alpha_np, args.top_k
            )
            for t, pick, d in zip(test_nodes, picks, degrees, strict=True):
                shown = alpha_np[pick].mean(axis=1)
                target_rows.append({
                    "target": int(t), "layer": li + 1, "relation": rel, "degree": d,
                    "n_shown": len(pick), "shown_weight": float(shown.sum()),
                    "focus": float(focus[t]) if d >= 2 else np.nan,
                    "relation_to_root": float(node_rel_root[t]) if d else np.nan,
                })  # fmt: skip
                if not len(pick):
                    continue
                edge_part = {
                    "target": np.full(len(pick), t), "layer": li + 1, "relation": rel,
                    "rank": np.arange(1, len(pick) + 1),
                    "neighbour": edges.src.numpy()[pick], "alpha": shown,
                }  # fmt: skip
                for head in range(heads):
                    edge_part[f"alpha_h{head}"] = alpha_np[pick, head]
                edge_parts.append(pd.DataFrame(edge_part))
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(out_dir / "attention_summary.csv", table)
    atomic_write_parquet(out_dir / "attention_nodes.parquet", pd.DataFrame(node_rows))
    targets = pd.DataFrame(target_rows)
    refit_prob = sup.sigmoid(res.logits)
    targets["refit_prob"] = refit_prob[targets["target"].to_numpy()]
    edges_table = (pd.concat(edge_parts, ignore_index=True) if edge_parts
                   else pd.DataFrame(columns=["target", "layer", "relation", "rank",
                                              "neighbour", "alpha"]))  # fmt: skip
    atomic_write_parquet(out_dir / "attention_edges.parquet", edges_table)
    atomic_write_parquet(out_dir / "attention_targets.parquet", targets)
    meta = {
        "dataset": args.dataset, "split_repeat": args.split,
        "feature_set": args.feature_set, "setting": summary["gnn"]["setting"],
        "fold_model": args.fold, "fit_seed": seed, "heads": cfg.heads,
        "layers": cfg.layers, "best_epoch": res.log["best_epoch"], "check": check,
        "top_k": args.top_k, "n_test_targets": int(len(test_nodes)),
        "relations": list(g.relations),
        "seconds": round(time.monotonic() - start, 1),
        "labels_used": "training labels only, to describe the model; no test label",
    }  # fmt: skip
    atomic_write_text(out_dir / "attention_meta.json", json.dumps(meta, indent=2) + "\n")

    print(f"{args.feature_set} on {args.dataset} split {args.split}: fold {args.fold} "
          f"model, {setting.key}, best epoch {res.log['best_epoch']}; refit check "
          f"{check['matches']} (fold PR-AUC {fold_ap:.4f})")  # fmt: skip
    head_mean = table[(table["head"] == "mean") & table["target_group"].isna()]
    print("\nAttention focus (head-averaged; entropy ratio 1 = uniform, like E5):")
    print(head_mean[["layer", "relation", "mean_degree_deg2plus", "entropy_ratio",
                     "max_alpha", "uniform_max"]].to_string(index=False))  # fmt: skip
    fr = table[table["target_group"].notna()]
    print("\nRelation weight and attention on fraud neighbours (head-averaged):")
    columns = [
        "layer", "relation", "relation_to_root", "target_group", "n_targets_mixed",
        "fraud_share_attention", "fraud_share_uniform",
    ]  # fmt: skip
    print(fr[columns].to_string(index=False))
    print(f"\nWrote {out_dir} (attention_nodes, _edges and _targets are row-level: "
          "local only)")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
