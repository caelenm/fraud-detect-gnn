"""Pipeline stages, in order. `run.py` and `scripts/NN_*.py` both call these.

Each stage reads its inputs from data/ or outputs/ (under the active
dataset's subfolder) and writes its outputs there, so any stage can be rerun
on its own. New stages (graph features, GNNs, evaluation) are added to STAGES
as they are implemented.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detect import columns as C
from fraud_detect.config import (
    REPO_ROOT,
    Paths,
    dataset_config,
    df_analyze_dir,
    get_paths,
)
from fraud_detect.data import audit, care_gnn, download, split
from fraud_detect.external import ExternalToolError, run_df_analyze_script
from fraud_detect.features import blocks, graph, own
from fraud_detect.models import (
    df_analyze,
    df_analyze_report,
    evaluation,
    gnn_outputs,
    model_export,
    selection,
)
from fraud_detect.report import web
from fraud_detect.runlog import git_commit, package_versions
from fraud_detect.runstate import (
    CHECKPOINT_DIR,
    UnitStore,
    atomic_write,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    config_fingerprint,
)


class StageError(RuntimeError):
    """A stage cannot run; the message says what to do."""


@dataclass(frozen=True)
class Context:
    config: dict[str, Any]
    paths: Paths
    run_dir: Path
    # True when this stage was interrupted earlier and is being continued with
    # `run.py --resume`: keep the units it already finished.
    resume: bool = False
    # run.py --force: the download stage may then replace a raw file whose
    # checksum differs from the expected one.
    force: bool = False

    @property
    def seed(self) -> int:
        return int(self.config["seed"])

    @property
    def dataset(self) -> str:
        return str(self.config["dataset"])

    @property
    def split_seed(self) -> int:
        """Seed of the grouped split: independent for each split repeat."""
        return self.seed + 1000 * int(self.config["split"].get("repeat", 0))

    @property
    def data(self) -> dict[str, Any]:
        """The active dataset's config section."""
        return dataset_config(self.config)

    def unit_store(self, stage: str) -> UnitStore:
        """Per-unit checkpoints for a long stage (see fraud_detect.runstate).
        Emptied on a fresh run, kept on --resume."""
        store = UnitStore(self.paths.outputs_dir / CHECKPOINT_DIR / stage)
        if not self.resume:
            store.reset()
        return store


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    run: Callable[[Context], None]
    outputs: Callable[[Paths], list[Path]]
    # Optional: other files whose presence also means the stage has nothing
    # left to do (see the download stage).
    skip_when: Callable[[Paths], list[Path]] | None = None

    def is_done(self, paths: Paths) -> bool:
        """True when every output exists (or every `skip_when` file does)."""
        if all(o.exists() for o in self.outputs(paths)):
            return True
        return self.skip_when is not None and all(
            p.exists() for p in self.skip_when(paths)
        )


def _require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            raise StageError(f"Missing input {p}. Run the earlier stages first.")


def _require_mat(ctx: Context) -> None:
    """The raw .mat; a clone that only unpacked a bundle has not downloaded it."""
    if not ctx.paths.mat.exists():
        raise StageError(
            f"Missing {ctx.paths.mat}. Download it with `uv run run.py --dataset "
            f"{ctx.dataset} --only download --force`."
        )


# --------------------------------------------------------------------------
# Stage 1: download (checksum-verified)
# --------------------------------------------------------------------------
def run_download(ctx: Context) -> None:
    try:
        download.download_dataset(
            ctx.paths.raw_dir, ctx.config["download"]["base_url"], ctx.data, ctx.force
        )
    except download.DownloadError as e:
        raise StageError(str(e)) from e


# --------------------------------------------------------------------------
# Stage 2: load the .mat: node table, relation edge lists, count checks
# --------------------------------------------------------------------------
def run_load(ctx: Context) -> None:
    p, data = ctx.paths, ctx.data
    _require_mat(ctx)
    print(f"Reading {p.mat}", flush=True)
    try:
        graph = care_gnn.read_mat(p.mat, data["relations"])
        edges = {
            name: care_gnn.edge_list(graph.adjacency[name])
            for name in care_gnn.relation_names(data["relations"])
        }
        care_gnn.check_edges_in_range(edges, graph.n_nodes)
        groups = care_gnn.group_ids(data["groups"], graph.features, edges)
        nodes = care_gnn.node_table(graph.labels, int(data["unlabelled_prefix"]), groups)
        counts = care_gnn.observed_counts(graph, nodes, edges)
        care_gnn.check_counts(counts, data["expected"], p.mat.name)
    except care_gnn.DatasetError as e:
        raise StageError(str(e)) from e

    mat_sha256 = download.sha256_file(p.mat)
    relations = []
    for name, e in edges.items():
        atomic_write(p.relation_file(name),
                     lambda tmp, e=e: np.savez_compressed(tmp, edges=e))  # fmt: skip
        relations.append(
            {
                "name": name,
                "source_key": data["relations"].get(name, care_gnn.HOMO),
                "file": p.relation_file(name).name,
                "n_edges": int(e.shape[1]),
                "n_nodes": graph.n_nodes,
                "mat_sha256": mat_sha256,
            }
        )
    atomic_write_parquet(p.nodes, nodes)
    labelled = nodes[C.IS_LABELLED]
    summary = {
        "dataset": ctx.dataset,
        "source": p.mat.name,
        "mat_sha256": mat_sha256,
        **counts,
        "n_unlabelled": int((~labelled).sum()),
        "positive_rate_labelled": counts["n_positive"] / max(counts["n_labelled"], 1),
        "grouping": data["groups"],
        "n_groups_labelled": int(nodes.loc[labelled, C.GROUP_ID].nunique()),
    }
    atomic_write_json(p.reports_dir / "load_summary.json", summary)
    manifest = {
        "dataset": ctx.dataset,
        "format": "npz key 'edges': int32 array of shape (2, E); each undirected "
        "edge once with src < dst; no self-loops",
        "source": p.mat.name,
        "mat_sha256": mat_sha256,
        "n_nodes": graph.n_nodes,
        "relations": relations,
    }
    # Written last: the stage counts as finished only when everything is saved.
    atomic_write_json(p.graph_manifest, manifest)
    edge_text = ", ".join(f"{r['name']} {r['n_edges']:,}" for r in relations)
    print(
        f"{graph.n_nodes:,} nodes ({counts['n_labelled']:,} labelled, "
        f"{counts['n_positive']:,} positive), {counts['n_features']} features, "
        f"{summary['n_groups_labelled']:,} split groups among labelled nodes; "
        f"undirected edges: {edge_text}. All counts match the config."
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_edges(path: Path) -> np.ndarray:
    """A relation's (2, E) edge list written by the load stage."""
    with np.load(path) as f:
        return f["edges"]


def read_nodes(p: Paths) -> pd.DataFrame:
    """The node table, checked against the saved split files when they exist."""
    _require(p.nodes)
    nodes = pd.read_parquet(p.nodes)
    if p.train_ids.exists() and p.test_ids.exists():
        try:
            split.check_against_saved_ids(
                nodes, pd.read_csv(p.train_ids), pd.read_csv(p.test_ids)
            )
        except split.SplitError as e:
            raise StageError(str(e)) from e
    return nodes


# --------------------------------------------------------------------------
# Stage 3: grouped, stratified train/test split and shared CV folds
# --------------------------------------------------------------------------
def run_split(ctx: Context) -> None:
    p = ctx.paths
    _require(p.nodes)
    nodes = pd.read_parquet(p.nodes)
    cfg = ctx.config["split"]
    try:
        nodes = split.split_nodes(nodes, cfg, ctx.split_seed)
        summary = split.summary(nodes, cfg, ctx.split_seed)
    except split.SplitError as e:
        raise StageError(str(e)) from e
    atomic_write_parquet(p.nodes, nodes)
    for path, table in (
        (p.train_ids, split.ids_table(nodes, C.TRAIN)),
        (p.test_ids, split.ids_table(nodes, C.TEST)),
        (p.cv_folds, split.folds_table(nodes)),
    ):
        atomic_write_csv(path, table)
    atomic_write_json(p.reports_dir / "split_summary.json", summary)
    atomic_write_json(p.split_summary, summary)  # last: marks the stage done
    for part in ("train", "test"):
        s = summary[part]
        print(f"{part:>5}: {s['n_nodes']:>6,} nodes, {s['n_positive']:>5,} positive "
              f"({s['positive_rate']:.4f}), {s['n_groups']:>6,} groups")  # fmt: skip
    print(f"CV folds on train: {cfg['cv_folds']}; no group crosses train/test or folds.")


# --------------------------------------------------------------------------
# Stage 4: own-feature block and column types (training nodes only)
# --------------------------------------------------------------------------
def run_features(ctx: Context) -> None:
    p = ctx.paths
    _require_mat(ctx)
    _require(p.train_ids)
    nodes = read_nodes(p)
    features = care_gnn.read_features(p.mat)
    if len(features) != len(nodes):
        raise StageError(
            f"{p.mat.name} has {len(features)} rows, nodes.parquet {len(nodes)}"
        )
    block = own.own_block(features)
    train_ids = pd.read_csv(p.train_ids)[C.NODE_ID].to_numpy()
    spec = own.column_spec(block, train_ids)
    if spec[own.CONSTANT]:
        raise StageError(
            f"Columns constant on the training nodes: {spec[own.CONSTANT]}. They "
            "carry no information and df-analyze would drop them; ask the group "
            "before removing them."
        )
    atomic_write_parquet(p.block_file(C.OWN_BLOCK), block)
    atomic_write_json(p.column_spec, spec)  # last: marks the stage done
    print(
        f"own block: {block.shape[1] - 1} features for {len(block):,} nodes; types "
        f"from training nodes: {len(spec[own.BINARY])} binary, "
        f"{len(spec[own.ORDINAL])} ordinal, {len(spec[own.CONTINUOUS])} continuous"
    )


# --------------------------------------------------------------------------
# Stage 5: audit (counts and statistics only)
# --------------------------------------------------------------------------
AUDIT_REPORT = "audit.md"
AUDIT_TABLES = ("audit_features.csv", "audit_relations.csv", "audit_groups.csv")


def run_audit(ctx: Context) -> None:
    p, data = ctx.paths, ctx.data
    _require(p.graph_manifest, p.block_file(C.OWN_BLOCK), p.column_spec, p.split_summary)
    nodes = read_nodes(p)
    block = pd.read_parquet(p.block_file(C.OWN_BLOCK))
    spec = _read_json(p.column_spec)
    manifest = _read_json(p.graph_manifest)
    threshold = float(ctx.config["audit"]["shortcut_auroc"])

    labelled = nodes[C.IS_LABELLED]
    found = {
        "n_nodes": len(nodes),
        "n_labelled": int(labelled.sum()),
        "n_positive": int(nodes.loc[labelled, C.LABEL].sum()),
        "n_features": block.shape[1] - 1,
        "n_edges": {r["name"]: r["n_edges"] for r in manifest["relations"]},
    }
    try:
        care_gnn.check_counts(found, data["expected"], ctx.dataset)
    except care_gnn.DatasetError as e:
        raise StageError(str(e)) from e

    features = audit.feature_table(block, nodes, spec, threshold)
    duplicates = audit.duplicate_counts(block, nodes)
    groups = audit.group_table(nodes)
    purity = audit.group_purity(nodes)
    relations = pd.DataFrame(
        [
            audit.relation_row(r["name"], load_edges(p.graph_dir / r["file"]), nodes)
            for r in manifest["relations"]
        ]
    )
    report = audit.report(
        ctx.dataset,
        audit.count_table(found, data["expected"]),
        _read_json(p.split_summary),
        features,
        duplicates,
        groups,
        purity,
        relations,
        threshold,
    )
    for name, table in zip(AUDIT_TABLES, (features, relations, groups), strict=True):
        atomic_write_csv(p.reports_dir / name, table)
    atomic_write_json(p.reports_dir / "audit_summary.json",
                      {"duplicates": duplicates, "group_purity": purity})  # fmt: skip
    if duplicates["test_rows_identical_to_a_train_row"]:
        raise StageError(
            f"{duplicates['test_rows_identical_to_a_train_row']} test rows are identical "
            "to a training row, so identical nodes sit on both sides of the split. "
            f"See {p.reports_dir / 'audit_features.csv'} and fix the grouping."
        )
    atomic_write(p.reports_dir / AUDIT_REPORT,
                 lambda tmp: tmp.write_text(report, encoding="utf-8"))  # fmt: skip
    flagged = features[features["possible_shortcut"]]
    print(report.split("## Exact duplicate")[0])
    print(
        f"{len(flagged)} feature(s) flagged as possible shortcuts "
        f"(AUROC >= {threshold}). Full audit: {p.reports_dir / AUDIT_REPORT}"
    )


# --------------------------------------------------------------------------
# Stages 6-7: graph feature blocks (nbr, gnnssl)
# --------------------------------------------------------------------------
# A block stage builds its block only for feature sets that use it, and its
# outputs are only the blocks the active feature set uses: none for m1_own,
# so a finished m1_own run (or bundle) has nothing left to run here.
GNN_EMBED_META = "meta.json"


def block_outputs(kind: str) -> Callable[[Paths], list[Path]]:
    """A block stage's outputs: each block of this kind that the active feature
    set uses (e.g. nbr, nbr_no36) and its column types (written last); nothing
    when the feature set uses none."""

    def outputs(p: Paths) -> list[Path]:
        names = blocks.blocks_of_kind(p.blocks, kind)
        return [f for b in names for f in (p.block_file(b), p.block_spec(b))]

    return outputs


def gnn_block_names(names: list[str]) -> list[str]:
    """The blocks the gnn_embed stage builds (gnnssl and gnnrand kinds)."""
    return [n for n in names if blocks.parse_block(n)[0] in C.GNN_BLOCK_KINDS]


def gnn_embed_outputs(p: Paths) -> list[Path]:
    files = [f for kind in C.GNN_BLOCK_KINDS for f in block_outputs(kind)(p)]
    meta = [p.embed_dir(b) / GNN_EMBED_META for b in gnn_block_names(p.blocks)]
    return [*files, *meta]


def _blocks_to_build(ctx: Context, *kinds: str) -> list[str]:
    names = [n for n in ctx.paths.blocks if blocks.parse_block(n)[0] in kinds]
    if not names:
        what = " or ".join(repr(k) for k in kinds)
        print(
            f"Feature set {ctx.config['feature_set']} uses no {what} block, so there "
            "is nothing to build (run with --set feature_set=<a set using it>)."
        )
    return names


def _variants(ctx: Context) -> dict[str, Any]:
    return dict(ctx.config.get("feature_variants") or {})


def _own_input(ctx: Context, block: str) -> pd.DataFrame:
    """The own features a graph block is built from: all of them, or without
    the columns its variant leaves out (e.g. issue #36 for *_no36)."""
    p = ctx.paths
    _require(p.block_file(C.OWN_BLOCK), p.train_ids)
    try:
        return blocks.own_view(pd.read_parquet(p.block_file(C.OWN_BLOCK)), block,
                               _variants(ctx))  # fmt: skip
    except blocks.BlockError as e:
        raise StageError(str(e)) from e


def _relation_edges(ctx: Context) -> dict[str, np.ndarray]:
    """The dataset's relations (not homo), in config order, as edge lists."""
    p = ctx.paths
    _require(p.graph_manifest)
    return {r: load_edges(p.relation_file(r)) for r in ctx.data["relations"]}


def _write_block(p: Paths, name: str, block: pd.DataFrame) -> dict[str, Any]:
    """Validate a graph block, type its columns on the training nodes (as the
    features stage does for own), and save both; the spec file is written
    last and marks the block as finished."""
    try:
        blocks.validate_block(name, block)
    except blocks.BlockError as e:
        raise StageError(str(e)) from e
    train_ids = pd.read_csv(p.train_ids)[C.NODE_ID].to_numpy()
    spec = own.column_spec(block, train_ids)
    if spec[own.CONSTANT]:
        raise StageError(
            f"Columns of block {name!r} constant on the training nodes: "
            f"{spec[own.CONSTANT]}. Ask the group before dropping them."
        )
    atomic_write_parquet(p.block_file(name), block)
    atomic_write_json(p.block_spec(name), spec)
    return spec


def _type_counts(spec: dict[str, Any]) -> str:
    return ", ".join(
        f"{len(spec[kind])} {kind}" for kind in (own.BINARY, own.ORDINAL, own.CONTINUOUS)
    )


def rewired_edges(
    ctx: Context, edges: dict[str, np.ndarray], n_nodes: int
) -> dict[str, np.ndarray]:
    """N2's null graph: each relation rewired by double-edge swaps so every
    node keeps exactly its degree (features/graph.py swap_rewire). The seed is
    tied to the split repeat (the split seed, plus the relation's index), and
    the graph is saved under graph_rewired/seed<split seed>/ and reused while
    its manifest matches, so a later GNN control (N3) sees the same graph.
    Built from the edge lists only, never a label or a feature."""
    p, seed = ctx.paths, ctx.split_seed
    swaps = float(ctx.config["neighbour_agg"]["rewire_swaps_per_edge"])
    folder = p.rewired_dir(seed)
    manifest = folder / "manifest.json"
    source = json.loads(p.graph_manifest.read_text(encoding="utf-8"))
    settings = {
        "split_seed": seed,
        "swaps_per_edge": swaps,
        "n_nodes": n_nodes,
        "mat_sha256": source.get("mat_sha256"),
        "relations": {
            r: {"seed": seed + j, "n_edges": int(e.shape[1])}
            for j, (r, e) in enumerate(edges.items())
        },  # fmt: skip
    }
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("settings") == settings:
            print(f"Reusing the rewired graph in {folder}")
            return {r: load_edges(folder / f"{r}.npz") for r in edges}
    out: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, float]] = {}
    for r, e in edges.items():
        start = time.monotonic()
        new, made = graph.swap_rewire(e, n_nodes, settings["relations"][r]["seed"], swaps)
        before = np.bincount(e.ravel(), minlength=n_nodes)
        if new.shape != e.shape or not np.array_equal(
            before, np.bincount(new.ravel(), minlength=n_nodes)
        ):
            raise StageError(f"Rewiring {r} changed the edge count or a degree")
        key = n_nodes  # (src, dst) -> one integer, to count surviving real edges
        kept = np.isin(new[0].astype(np.int64) * key + new[1],
                       e[0].astype(np.int64) * key + e[1]).mean()  # fmt: skip
        atomic_write(folder / f"{r}.npz",
                     lambda tmp, x=new: np.savez_compressed(tmp, edges=x))  # fmt: skip
        stats[r] = {"swaps_made": made, "swaps_per_edge_made": made / max(e.shape[1], 1),
                    "real_edges_kept": float(kept),
                    "seconds": round(time.monotonic() - start, 1)}  # fmt: skip
        out[r] = new
        print(f"Rewired {r}: {made / max(e.shape[1], 1):.1f} swaps per edge, "
              f"{kept:.1%} of real edges kept, every degree unchanged")  # fmt: skip
    atomic_write_json(manifest, {"settings": settings, "stats": stats})
    return out


def run_neighbour_agg(ctx: Context) -> None:
    p, cfg = ctx.paths, ctx.config["neighbour_agg"]
    if list(cfg["statistics"]) != ["mean"]:
        raise StageError(
            f"neighbour_agg.statistics is {cfg['statistics']}; only [mean] is "
            "implemented (1-hop max and 2-hop means are later ablations)."
        )
    names = _blocks_to_build(ctx, *C.NBR_BLOCK_KINDS)
    if not names:
        return
    edges = _relation_edges(ctx)
    rewired: dict[str, np.ndarray] | None = None
    for name in names:
        own_input = _own_input(ctx, name)
        kind = blocks.parse_block(name)[0]
        try:
            if kind == C.DEG_BLOCK:
                block = graph.build_deg_block(own_input, edges, name=name)
            else:
                if kind == C.NBRRW_BLOCK and rewired is None:
                    rewired = rewired_edges(ctx, edges, len(own_input))
                block = graph.build_nbr_block(
                    own_input,
                    rewired if kind == C.NBRRW_BLOCK else edges,
                    log1p_degree=bool(cfg["log1p_degree"]),
                    name=name,
                )
        except ValueError as e:
            raise StageError(str(e)) from e
        spec = _write_block(p, name, block)
        print(
            f"{name} block: {block.shape[1] - 1} features from {own_input.shape[1] - 1} "
            f"own features for {len(block):,} nodes over relations {', '.join(edges)}; "
            f"types from training nodes: {_type_counts(spec)}"
        )


def split_masks(nodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """In node order: `fit` = training nodes (scaler and PCA fits); `loss` = the
    non-test nodes, train and unlabelled (the DGI loss). From the split
    column only, never the labels."""
    ordered = nodes.sort_values(C.NODE_ID)
    if not np.array_equal(ordered[C.NODE_ID].to_numpy(), np.arange(len(ordered))):
        raise StageError("nodes.parquet must have one row per node 0..n-1")
    split = ordered[C.SPLIT].astype("string").fillna("").to_numpy(dtype=str)
    unknown = sorted(set(split) - {C.TRAIN, C.TEST, C.UNLABELLED})
    if unknown:
        raise StageError(f"Nodes without a train/test/unlabelled split: {unknown}")
    return split == C.TRAIN, split != C.TEST


def gnn_embed_fingerprint(ctx: Context, block: str = C.GNNSSL_BLOCK) -> str:
    """What a gnnssl or gnnrand block depends on besides the data: its kind,
    settings, the seed, the relations and the own columns its variant leaves
    out or keeps. Recorded in meta.json and checked before use."""
    try:
        kind = blocks.parse_block(block)[0]
        spec = blocks.variant_spec(block, _variants(ctx))
    except blocks.BlockError as e:
        raise StageError(str(e)) from e
    fields: dict[str, Any] = {
        "gnn_embed": ctx.config["gnn_embed"],
        "seed": ctx.seed,
        "relations": list(ctx.data["relations"]),
    }
    # Each field is added only when it applies, so the fingerprints of the
    # blocks built before it existed (gnnssl, gnnssl_no36) stay unchanged.
    if isinstance(spec, dict):
        fields["kept_own_columns"] = spec[blocks.KEEP]
    elif spec:
        fields["excluded_own_columns"] = spec
    if kind != C.GNNSSL_BLOCK:
        fields["kind"] = kind
    if blocks.parse_block(block)[1] in (ctx.config.get("gnn_embed_sizes") or {}):
        fields["size"] = gnn_embed_size(ctx, block)
    return config_fingerprint(fields)


def gnn_embed_size(ctx: Context, block: str) -> int | str:
    """PCA components per relation of a gnnssl or gnnrand block: the block
    variant's entry in gnn_embed_sizes (a number, or "raw" for the embedding
    itself; Z1's size rungs, 2026-10-26), else gnn_embed.pca_components."""
    sizes = ctx.config.get("gnn_embed_sizes") or {}
    variant = blocks.parse_block(block)[1]
    return sizes.get(variant, int(ctx.config["gnn_embed"]["pca_components"]))


def run_gnn_embed(ctx: Context) -> None:
    """GNN embeddings for each gnnssl or gnnrand block the feature set uses
    (graph/ssl.py): DGI-trained for gnnssl, untrained for gnnrand; one unit
    per (block, relation), then PCA per relation. The raw embeddings stay in
    outputs/<dataset>/gnn_embed[_<block>]/ for the comparison page."""
    names = _blocks_to_build(ctx, *C.GNN_BLOCK_KINDS)
    if not names:
        return
    try:
        from fraud_detect.graph import ssl  # noqa: PLC0415  (needs the gnn extra)
    except ImportError as e:
        raise StageError(
            "The gnn_embed stage needs torch and torch_geometric. Install them with "
            f"`uv sync --extra gnn` ({e})."
        ) from e
    p = ctx.paths
    try:
        cfg = ssl.SSLConfig.from_config(ctx.config["gnn_embed"], ctx.seed)
        device = ssl.resolve_device(cfg.device)
    except (TypeError, ValueError) as e:
        raise StageError(f"gnn_embed settings: {e}") from e
    fit_mask, loss_mask = split_masks(read_nodes(p))
    edges = _relation_edges(ctx)
    store = ctx.unit_store("gnn_embed")
    for name in names:
        _embed_block(ctx, ssl, cfg, device, name, store, fit_mask, loss_mask, edges)


def _embed_block(
    ctx: Context,
    ssl: Any,
    cfg: Any,
    device: str,
    name: str,
    store: UnitStore,
    fit_mask: np.ndarray,
    loss_mask: np.ndarray,
    edges: dict[str, np.ndarray],
) -> None:
    """One gnnssl block (DGI per relation) or gnnrand block (the same encoder,
    untrained): a saved unit per relation, PCA, meta.json, then the block and
    its spec."""
    p = ctx.paths
    trained = blocks.parse_block(name)[0] == C.GNNSSL_BLOCK
    own_input = _own_input(ctx, name)
    try:
        x = ssl.standardized_features(own_input, fit_mask)
        ssl.check_masks(len(x), fit_mask, loss_mask)
    except ValueError as e:
        raise StageError(str(e)) from e
    how = (
        f"DGI per relation on {device}" if trained
        else f"untrained GCN encoder (seeded initial weights) per relation on {device}"
    )  # fmt: skip
    loss = f"loss over {int(loss_mask.sum()):,} non-test nodes; " if trained else ""
    print(
        f"{name}: {how} from {x.shape[1]} own features; {len(x):,} nodes; {loss}"
        f"scaler and PCA fit on {int(fit_mask.sum()):,} training nodes",
        flush=True,
    )
    embeddings: dict[str, np.ndarray] = {}
    logs: dict[str, dict[str, Any]] = {}
    for relation, rel_edges in edges.items():
        unit, path = f"{name}/{relation}", p.raw_embedding(relation, name)
        if store.done(unit) and path.is_file():
            logs[relation] = store.load(unit)["log"]
            embeddings[relation] = np.load(path)
            print(f"  {relation}: finished in the interrupted run; kept")
            continue
        if trained:
            result = ssl.embed_relation(x, rel_edges, loss_mask, cfg, relation)
        else:
            result = ssl.untrained_relation(x, rel_edges, cfg, relation)
        atomic_write(path, lambda tmp, z=result.embedding: np.save(tmp, z))
        store.save(unit, {"log": result.log})
        embeddings[relation], logs[relation] = result.embedding, result.log
        log = result.log
        epochs = (
            f"{log['epochs_run']} epochs (best {log['best_epoch']}, loss "
            f"{log['best_loss']:.4f})" if trained else "untrained"
        )  # fmt: skip
        print(
            f"  {relation}: {log['n_edges_used']:,} edges, {epochs} on "
            f"{log['device']} in {log['seconds']:.1f} s",
            flush=True,
        )

    size = gnn_embed_size(ctx, name)
    try:
        block, pca_info = ssl.pca_block(embeddings, fit_mask, size, cfg.seed, name=name)
    except ValueError as e:
        raise StageError(str(e)) from e
    versions = package_versions()
    own_columns = list(pd.read_parquet(p.block_file(C.OWN_BLOCK)).columns)
    if trained:
        method = {
            "method": "Deep Graph Infomax (torch_geometric.nn.DeepGraphInfomax), one "
            "model per relation, no labels",
            "corruption": "row shuffle of the node features",
            "loss_nodes": "non-test nodes (train and unlabelled); test nodes take "
            "part in message passing only",
        }
    else:
        method = {
            "method": "untrained control: the gnnssl encoder with its seeded "
            "initial weights (DGI's starting point) and no training step, one per "
            "relation, no labels",
            "loss_nodes": "none (nothing is trained)",
        }
    meta = {
        "block": name,
        "own_columns_used": [c for c in own_input.columns if c != C.NODE_ID],
        "own_columns_left_out": blocks.excluded_columns(
            name, _variants(ctx), own_columns
        ),  # fmt: skip
        **method,
        "encoder": f"{cfg.layers} rounds of GCN-normalised propagation over A + I, "
        f"hidden {cfg.hidden}, output {cfg.out_dim}, PReLU",
        "fit_nodes": "the feature scaler and each PCA are fit on training nodes only",
        "pca_components_used": size,
        "n_nodes": len(x),
        "n_fit_nodes": int(fit_mask.sum()),
        "n_loss_nodes": int(loss_mask.sum()),
        "config": asdict(cfg),
        "config_sha256": gnn_embed_fingerprint(ctx, name),
        "versions": {
            k: versions[k] for k in ("torch", "torch-geometric") if k in versions
        },
        "cuda_device": ssl.device_name(device),
        "relations": {r: {**logs[r], **pca_info[r]} for r in edges},
        "raw_embeddings": {r: p.raw_embedding(r, name).name for r in edges},
        "columns": [c for c in block.columns if c != C.NODE_ID],
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    atomic_write_json(p.embed_dir(name) / GNN_EMBED_META, meta)
    spec = _write_block(p, name, block)
    explained = ", ".join(
        f"{r} {pca_info[r]['explained_variance_total']:.0%}" for r in edges
    )
    print(
        f"{name} block: {block.shape[1] - 1} features for {len(block):,} nodes "
        f"(size {size} per relation); PCA "
        f"variance kept per relation: {explained}; types from training nodes: "
        f"{_type_counts(spec)}"
    )


def check_gnn_embed_settings(ctx: Context, block: str) -> None:
    """Warn when a gnnssl or gnnrand block was built with other gnn_embed
    settings, seed, relations or left-out columns than the current config."""
    meta_path = ctx.paths.embed_dir(block) / GNN_EMBED_META
    _require(meta_path)
    if _read_json(meta_path).get("config_sha256") != gnn_embed_fingerprint(ctx, block):
        print(
            f"WARNING: the {block} block was built with different gnn_embed settings, "
            "seed, relations or left-out columns than the current config. Rebuild it "
            f"with `uv run run.py --dataset {ctx.dataset} --only gnn_embed --force "
            f"--set feature_set={ctx.config['feature_set']}` if that was not intended."
        )


# --------------------------------------------------------------------------
# Supervised GNNs (2026-10-31, docs/GNN_MODELS_PLAN.md): shared helpers for
# the gnn_oof block stage (Model 6) and the gnn_classify stage (Models 3-4)
# --------------------------------------------------------------------------
GNN_MODEL_LABELS = {
    ("real", "mean"): "GraphSAGE",
    ("real", "sym"): "R-GCN-style GNN",
    ("none", "mean"): "MLP (GNN without edges)",
    ("none", "sym"): "MLP (GNN without edges)",
    ("rewired", "mean"): "GraphSAGE (rewired graph)",
    ("rewired", "sym"): "R-GCN-style GNN (rewired graph)",
}


GAT_LABELS = {
    "real": "GAT",
    "rewired": "GAT (rewired graph)",
    "none": "MLP (GAT without edges)",
}
SIGN_LABELS = {
    "real": "SIGN",
    "rewired": "SIGN (rewired graph)",
    "none": "MLP (SIGN without edges)",
}


def gnn_model_label(spec: dict[str, Any]) -> str:
    """The model's name in reports. A rung with its own `train:` block
    (docs/BIGGER_GNN_PLAN.md) is marked "larger", or named SIGN."""
    train = spec.get("train") or {}
    if train.get("model", "sage") == "sign":
        return SIGN_LABELS[spec["graph"]]
    if train.get("model") == "gat":
        return GAT_LABELS[spec["graph"]]
    label = GNN_MODEL_LABELS[(spec["graph"], spec.get("norm", "mean"))]
    return f"{label}, larger" if train else label


def _import_supervised() -> Any:
    try:
        from fraud_detect.graph import supervised  # noqa: PLC0415  (needs the gnn extra)
    except ImportError as e:
        raise StageError(
            "The supervised GNN stages need torch. Install it with "
            f"`uv sync --extra gnn` ({e})."
        ) from e
    return supervised


def _train_config(ctx: Context, sup: Any, spec: dict[str, Any]) -> Any:
    """gnn_classify's settings, with the rung's own `train:` block on top."""
    try:
        return sup.rung_config(ctx.config["gnn_classify"], spec)
    except (TypeError, ValueError) as e:
        raise StageError(f"gnn_classify settings: {e}") from e


@dataclass
class GNNData:
    """What a supervised GNN stage needs, with labels for training rows only."""

    x: np.ndarray  # standardised own features, node order (scaler: training nodes)
    train_nodes: np.ndarray  # training node IDs, in train_ids.csv order
    y_train: np.ndarray  # their labels
    folds: np.ndarray  # their saved CV folds
    groups: np.ndarray  # their split groups
    test_nodes: np.ndarray  # test node IDs, in test_ids.csv order (no labels)
    own_columns: list[str]


def gnn_data(ctx: Context, own_block: str) -> GNNData:
    """The own features (all nodes) and the training rows' labels, folds and
    groups. Test labels are not read here."""
    p = ctx.paths
    _require(p.train_ids, p.test_ids, p.cv_folds)
    from fraud_detect.graph import ssl  # noqa: PLC0415  (needs the gnn extra)

    nodes = read_nodes(p)
    fit_mask, _ = split_masks(nodes)
    own_input = _own_input(ctx, own_block)
    try:
        x = ssl.standardized_features(own_input, fit_mask)
    except ValueError as e:
        raise StageError(str(e)) from e
    train_ids = pd.read_csv(p.train_ids)
    folds = folds_in_train_order(p)
    groups = nodes.set_index(C.NODE_ID)[C.GROUP_ID]
    return GNNData(
        x=x,
        train_nodes=train_ids[C.NODE_ID].to_numpy(dtype=np.int64),
        y_train=train_ids[C.LABEL].to_numpy(dtype=int),
        folds=folds["fold"].to_numpy(dtype=int),
        groups=groups.loc[train_ids[C.NODE_ID]].to_numpy(),
        test_nodes=pd.read_csv(p.test_ids, usecols=[C.NODE_ID])[C.NODE_ID].to_numpy(
            dtype=np.int64
        ),
        own_columns=[c for c in own_input.columns if c != C.NODE_ID],
    )


def gnn_edges(ctx: Context, graph_kind: str, n_nodes: int) -> dict[str, np.ndarray]:
    """Every relation (real), N2's rewired graph (rewired, built or reused),
    or no edges (none)."""
    if graph_kind == "none":
        return {}
    edges = _relation_edges(ctx)
    return rewired_edges(ctx, edges, n_nodes) if graph_kind == "rewired" else edges


def _fs_paths(ctx: Context, feature_set: str) -> Paths:
    """Paths of another feature set on the same dataset and split repeat."""
    return get_paths({**ctx.config, "feature_set": feature_set}, ctx.paths.root)


def gnn_rung_summary(ctx: Context, feature_set: str) -> dict[str, Any]:
    """A finished GNN classifier rung's model_a.json on this split repeat."""
    path = _fs_paths(ctx, feature_set).model_reports_dir / MODEL_A_SUMMARY
    if not path.is_file():
        raise StageError(
            f"{feature_set} has no result on this split ({path}). Run it first: "
            f"`uv run run.py --dataset {ctx.dataset} --set feature_set={feature_set}`."
        )
    summary = _read_json(path)
    if not gnn_outputs.is_gnn_summary(summary):
        raise StageError(f"{path} is not a GNN classifier rung's result")
    return summary


# --------------------------------------------------------------------------
# Stage 8: gnnsup block (Model 6): the best GNN's out-of-fold score
# --------------------------------------------------------------------------
def gnn_oof_outputs(p: Paths) -> list[Path]:
    names = blocks.blocks_of_kind(p.blocks, C.GNNSUP_BLOCK)
    return [
        f for b in names for f in (p.nested_scores(b), p.block_file(b), p.block_spec(b))
    ]


def best_gnn(ctx: Context) -> dict[str, Any]:
    """The rule fixed before the runs (RESEARCH_PLAN §11, 2026-10-31): on this
    split's own training folds, a candidate other than the first (E5) is
    chosen only if its mean per-fold out-of-fold PR-AUC gain over the first
    exceeds one standard error (SD / sqrt(folds)); the largest such gain wins.
    Otherwise the first. Test results never enter."""
    candidates = list(ctx.config["gnn_oof"]["candidates"])
    summaries = {fs: gnn_rung_summary(ctx, fs) for fs in candidates}
    base = np.asarray(summaries[candidates[0]]["gnn"]["fold_pr_auc"])
    rows, chosen, best_gain = [], candidates[0], 0.0
    for fs in candidates[1:]:
        d = np.asarray(summaries[fs]["gnn"]["fold_pr_auc"]) - base
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        beats = bool(d.mean() > se)
        rows.append({"candidate": fs, "mean_gain": float(d.mean()), "se": se,
                     "beats_first": beats})  # fmt: skip
        if beats and d.mean() > best_gain:
            chosen, best_gain = fs, float(d.mean())
    return {
        "feature_set": chosen,
        "spec": dict(ctx.config["gnn_models"][chosen]),
        "setting": summaries[chosen]["gnn"]["setting"],
        "own_block": feature_set_blocks_of(ctx, chosen)[0],
        "rule": "first candidate unless another's mean per-fold out-of-fold PR-AUC "
        "gain over it exceeds one SE on this split's training folds",
        "comparisons": rows,
        "candidates_cv_pr_auc": {
            fs: summaries[fs]["model_a"]["cv_pr_auc"] for fs in candidates
        },
    }


def feature_set_blocks_of(ctx: Context, feature_set: str) -> list[str]:
    return list(ctx.config["feature_sets"][feature_set])


def run_gnn_oof(ctx: Context) -> None:
    """Model 6's gnnsup block, the one documented exception to invariant 1:
    the best GNN (best_gnn) with its Part A setting, nested-cross-fit on the
    saved folds for each seed (one unit per seed). gnnsup__score is the logit
    of the seed-averaged probability: out-of-fold for training rows, the mean
    of the fold models for every other node. The nested scores (for the
    shared CV) go to <block>.nested.parquet. No test label is read."""
    names = _blocks_to_build(ctx, C.GNNSUP_BLOCK)
    if not names:
        return
    if names != [C.GNNSUP_BLOCK]:
        raise StageError(f"Only the plain {C.GNNSUP_BLOCK} block is defined, not {names}")
    sup = _import_supervised()
    choice = best_gnn(ctx)
    spec = choice["spec"]
    p, cfg = ctx.paths, _train_config(ctx, sup, spec)
    setting = sup.Setting(**choice["setting"])
    data = gnn_data(ctx, choice["own_block"])
    g = sup.graph_input(data.x, gnn_edges(ctx, spec["graph"], len(data.x)),
                        spec.get("norm", "mean"), cfg.device, cfg.model)  # fmt: skip
    print(
        f"gnnsup: {gnn_model_label(spec)} ({choice['feature_set']}, chosen by the "
        f"pre-registered rule) with {setting.key}; nested cross-fitting on "
        f"{len(np.unique(data.folds))} folds x {cfg.seeds} seeds; {len(data.x):,} nodes",
        flush=True,
    )
    store = ctx.unit_store("gnn_oof")
    seeds = [ctx.seed + s for s in range(cfg.seeds)]
    results = []
    for seed in seeds:
        unit = f"gnnsup/{choice['feature_set']}/{setting.key}/seed{seed}"
        if store.done(unit):
            results.append(store.load(unit))
            print(f"  seed {seed}: finished in the interrupted run; kept")
            continue
        start = time.monotonic()
        outer, nested = sup.nested_cross_fit(
            g, data.train_nodes, data.y_train, data.folds, data.groups, setting, cfg, seed
        )
        result = {
            "seed": seed,
            "oof": outer.oof.tolist(),
            "all_nodes": outer.all_nodes.tolist(),
            "nested": {str(k): v.tolist() for k, v in nested.items()},
            "fold_pr_auc": sup.fold_pr_auc(data.y_train, outer.oof, data.folds),
            "logs": outer.logs,
        }
        store.save(unit, result)
        results.append(result)
        print(f"  seed {seed}: out-of-fold PR-AUC by fold "
              f"{[round(v, 3) for v in result['fold_pr_auc']]} "
              f"({time.monotonic() - start:.0f} s)", flush=True)  # fmt: skip
    oof = np.mean([r["oof"] for r in results], axis=0)
    all_nodes = np.mean([r["all_nodes"] for r in results], axis=0)
    column = f"{C.GNNSUP_BLOCK}{C.BLOCK_SEPARATOR}score"
    score = sup.logit(all_nodes)
    score[data.train_nodes] = sup.logit(oof)
    block = pd.DataFrame(
        {C.NODE_ID: np.arange(len(score), dtype=np.int64), column: score}
    )
    nested_rows = []
    for k in sorted(np.unique(data.folds)):
        values = np.mean([r["nested"][str(k)] for r in results], axis=0)
        outside = data.folds != k
        if np.isnan(values[outside]).any() or not np.isnan(values[~outside]).all():
            raise StageError(
                f"Nested scores for outer fold {k} do not cover the other folds"
            )
        nested_rows.append(pd.DataFrame({
            C.NODE_ID: data.train_nodes[outside],
            "outer_fold": int(k),
            column: sup.logit(values[outside]),
        }))  # fmt: skip
    versions = package_versions()
    meta = {
        "block": C.GNNSUP_BLOCK,
        "method": "logit of a supervised GNN's out-of-fold P(fraud), averaged over "
        "seeds (Model 6; invariant 1's documented exception)",
        "training_rows": "out-of-fold: each from the model fit without the row's CV fold",
        "other_nodes": "mean of the fold models (test labels are never read)",
        "nested": "for outer fold k, the other folds' scores from models fit without "
        "fold k (select_model fits on these when fold k is held out)",
        "choice": choice,
        "setting": asdict(setting),
        "seeds": seeds,
        "config": asdict(cfg),
        "per_seed_fold_pr_auc": {str(r["seed"]): r["fold_pr_auc"] for r in results},
        "fold_pr_auc": sup.fold_pr_auc(data.y_train, oof, data.folds),
        "versions": {k: versions[k] for k in ("torch",) if k in versions},
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    atomic_write_json(p.gnn_oof_dir / "meta.json", meta)
    atomic_write_parquet(p.nested_scores(C.GNNSUP_BLOCK),
                         pd.concat(nested_rows, ignore_index=True))  # fmt: skip
    spec_out = _write_block(p, C.GNNSUP_BLOCK, block)
    print(
        f"gnnsup block: {column} for {len(block):,} nodes from {choice['feature_set']}; "
        f"out-of-fold PR-AUC by fold {[round(v, 3) for v in meta['fold_pr_auc']]}; "
        f"types from training nodes: {_type_counts(spec_out)}"
    )


# --------------------------------------------------------------------------
# Stage 9: df-analyze input tables for the active feature set
# --------------------------------------------------------------------------
FEATURE_SET_INFO = "feature_set.json"


def feature_set_blocks(ctx: Context) -> list[str]:
    return list(ctx.config["feature_sets"][ctx.config["feature_set"]])


def column_types(p: Paths, block_names: list[str]) -> dict[str, dict[str, Any]]:
    """Every column's type, all decided on training nodes: the own block's from
    column_spec.json, every other block's from its <block>.spec.json."""
    _require(p.column_spec)
    types = dict(_read_json(p.column_spec)["columns"])
    for name in block_names:
        if blocks.parse_block(name)[0] != C.OWN_BLOCK:  # own_<variant>: own's types
            _require(p.block_spec(name))
            types.update(_read_json(p.block_spec(name))["columns"])
    return types


def run_df_analyze_input(ctx: Context) -> None:
    p = ctx.paths
    _require(p.train_ids, p.test_ids, p.column_spec)
    nodes = read_nodes(p)
    names = feature_set_blocks(ctx)
    for name in gnn_block_names(names):
        check_gnn_embed_settings(ctx, name)
    try:
        features = blocks.load_feature_set(p.features_dir, names, _variants(ctx))
        train_ids, test_ids = pd.read_csv(p.train_ids), pd.read_csv(p.test_ids)
        train, test = blocks.model_tables(features, train_ids, test_ids)
    except blocks.BlockError as e:
        raise StageError(str(e)) from e
    # The labels in the ID files must be the node table's labels.
    by_id = nodes.set_index(C.NODE_ID)[C.LABEL]
    for ids, table in ((train_ids, train), (test_ids, test)):
        if not np.array_equal(by_id.loc[ids[C.NODE_ID]].to_numpy(int), table[C.TARGET]):
            raise StageError("Labels in the split files differ from nodes.parquet")
    spec = column_types(p, names)
    feature_cols = [c for c in train.columns if c != C.TARGET]
    untyped = [c for c in feature_cols if c not in spec]
    if untyped:
        raise StageError(
            f"No column type in {p.column_spec} or the blocks' .spec.json files "
            f"for: {untyped}"
        )
    try:
        renames = df_analyze.df_analyze_names(feature_cols)
    except ValueError as e:
        raise StageError(str(e)) from e
    # Written with the names df-analyze uses, so it renames nothing.
    train, test = train.rename(columns=renames), test.rename(columns=renames)
    info = {
        "feature_set": ctx.config["feature_set"],
        "blocks": names,
        "n_train": len(train),
        "n_test": len(test),
        # df-analyze name -> our block column and its type (column_spec.json)
        "columns": {
            renames[c]: {"block_column": c, "type": spec[c]["type"]} for c in feature_cols
        },
    }
    atomic_write_parquet(p.df_analyze_input_dir / "train.parquet", train)
    atomic_write_parquet(p.df_analyze_input_dir / "test.parquet", test)
    atomic_write_json(p.df_analyze_input_dir / FEATURE_SET_INFO, info)
    print(
        f"Feature set {info['feature_set']} ({', '.join(names)}): "
        f"{len(feature_cols)} features; train {train.shape}, test {test.shape}"
    )


def read_feature_set_info(p: Paths) -> dict[str, Any]:
    path = p.df_analyze_input_dir / FEATURE_SET_INFO
    _require(path)
    return _read_json(path)


def columns_of_type(info: dict[str, Any], kind: str) -> list[str]:
    return [name for name, c in info["columns"].items() if c["type"] == kind]


# --------------------------------------------------------------------------
# Stage 10: run df-analyze; verify its split and its column types
# --------------------------------------------------------------------------
# df-analyze models that train on the GPU when CUDA is available.
GPU_CLASSIFIERS = frozenset({"catboost", "gandalf"})
CUDA_INFO_RUNNER = REPO_ROOT / "scripts" / "dfa" / "cuda_info.py"
CUDA_HELP = """df-analyze's environment cannot see a CUDA GPU. Check that:
  - `nvidia-smi` works in this shell (inside a toolbox/container the NVIDIA
    driver's user-space libraries must be visible there too), and
  - this prints True:
      uv run --python '{python}' --directory {dfa} \\
        python -c "import torch; print(torch.cuda.is_available())"
To run on the CPU instead, set `gpu.require: false` in the config."""
SPLIT_CHECK = "split_check.json"
TYPE_CHECK = "type_check.json"


def _run_dfa(ctx: Context, script: str, args: list[str], log_path: Path) -> None:
    """Run a script in df-analyze's environment; turn failures into StageError."""
    cfg = ctx.config["df_analyze"]
    try:
        run_df_analyze_script(
            df_analyze_dir(ctx.config),
            script,
            args,
            log_path,
            cfg.get("commit"),
            python=cfg.get("python"),
        )
    except ExternalToolError as e:
        raise StageError(str(e)) from e


def check_cuda(ctx: Context, purpose: str) -> bool:
    """Ask df-analyze's environment whether it can use a CUDA GPU, log the answer
    to the run directory, and stop if a GPU is required but missing."""
    report = ctx.run_dir / "cuda_info.json"
    args = ["--report", str(report.resolve())]
    _run_dfa(ctx, str(CUDA_INFO_RUNNER), args, ctx.run_dir / "cuda_info.log")
    info = _read_json(report)
    if info["cuda_available"]:
        print(f"GPU for {purpose}: {info['device_name']} (torch {info['torch']})")
        return True
    message = CUDA_HELP.format(
        python=ctx.config["df_analyze"].get("python"), dfa=df_analyze_dir(ctx.config)
    )
    if ctx.config["gpu"]["require"]:
        raise StageError(message)
    print(f"WARNING: {purpose} will run on the CPU.\n{message}")
    return False


def gnn_rung(ctx: Context) -> bool:
    """True (with a note) for a GNN classifier rung (gnn_models): the
    gnn_classify stage trains and scores it, so a df-analyze stage has
    nothing to do. A fresh run reaches these stages because run.py reruns
    every stage after the first one with missing outputs."""
    if not ctx.paths.is_gnn:
        return False
    print(
        f"{ctx.config['feature_set']} is a GNN classifier rung (gnn_models); the "
        "gnn_classify stage trains and scores it, so there is nothing to do here."
    )
    return True


def _df_analyze_only(
    outputs: Callable[[Paths], list[Path]],
) -> Callable[[Paths], list[Path]]:
    """A df-analyze stage's outputs, or none for a GNN classifier rung."""
    return lambda p: [] if p.is_gnn else outputs(p)


def run_df_analyze(ctx: Context) -> None:
    if gnn_rung(ctx):
        return
    p = ctx.paths
    train_path = p.df_analyze_input_dir / "train.parquet"
    test_path = p.df_analyze_input_dir / "test.parquet"
    _require(train_path, test_path)
    info = read_feature_set_info(p)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    outdir = p.df_analyze_output_dir / stamp
    outdir.mkdir(parents=True, exist_ok=False)
    gpu_models = GPU_CLASSIFIERS.intersection(ctx.config["df_analyze"]["classifiers"])
    if gpu_models:
        # CatBoost and GANDALF switch to the GPU when df-analyze sees CUDA.
        check_cuda(ctx, "df-analyze (" + ", ".join(sorted(gpu_models)) + ")")
    args = df_analyze.df_analyze_args(
        ctx.config["df_analyze"],
        train_path,
        test_path,
        outdir,
        ctx.seed,
        ordinals=columns_of_type(info, own.ORDINAL),
    )
    _run_dfa(ctx, "df-analyze.py", args, outdir / "df_analyze.log")
    verify_df_analyze_run(ctx, outdir, info)
    atomic_write_text(p.df_analyze_output_dir / "latest.txt", stamp + "\n")


def verify_df_analyze_run(ctx: Context, outdir: Path, info: dict[str, Any]) -> None:
    """Check df-analyze's exported split against ours row by row, and its
    inferred column types against column_spec.json; save both reports."""
    p = ctx.paths
    train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    test = pd.read_parquet(p.df_analyze_input_dir / "test.parquet")
    try:
        export = df_analyze.read_export(df_analyze.find_export_dir(outdir))
        split_report = df_analyze.verify_export(
            export, train, test, columns_of_type(info, own.CONTINUOUS)
        )
        inferred = pd.read_csv(df_analyze.find_inferred_types(outdir), index_col=0)
        expected = {name: c["type"] for name, c in info["columns"].items()}
        type_report = df_analyze.check_inferred_types(
            inferred, expected, export.X_train.columns
        )
    except (df_analyze.SplitVerificationError, df_analyze.TypeCheckError) as e:
        raise StageError(f"{e}\nStop and ask before working around this.") from e
    for name, report in ((TYPE_CHECK, type_report), (SPLIT_CHECK, split_report)):
        atomic_write_json(outdir / name, report)
        atomic_write_json(p.model_reports_dir / name, {"run": outdir.name, **report})
    print("Verified: df-analyze's exported train/test rows match our saved split.")
    print(f"Verified: df-analyze kept all {type_report['n_features']} features with our "
          "column types.")  # fmt: skip
    for line in type_report["dropped_or_coerced"]:
        print(f"  df-analyze note: {line}")


def _latest_verified_run(p: Paths) -> tuple[str, Path]:
    """The latest df-analyze run, which must have passed both checks."""
    latest = p.df_analyze_output_dir / "latest.txt"
    _require(latest)
    stamp = latest.read_text(encoding="utf-8").strip()
    outdir = p.df_analyze_output_dir / stamp
    for check in (SPLIT_CHECK, TYPE_CHECK):
        if not (outdir / check).is_file():
            raise StageError(
                f"df-analyze run {outdir} has no {check}, so it was never verified. "
                "Rerun the df_analyze stage."
            )
    return stamp, outdir


# --------------------------------------------------------------------------
# Stage 11: score every tuned model on the saved grouped CV folds (train only)
# --------------------------------------------------------------------------
CV_SELECT_RUNNER = REPO_ROOT / "scripts" / "dfa" / "cv_select.py"
SHARED_CV = "model_selection_cv.csv"
OOF_PREDICTIONS = "oof_predictions.parquet"
TUNING_BUDGET = "tuning_budget.csv"
FOLDS_IN_TRAIN_ORDER = "cv_folds_train_order.csv"


def folds_in_train_order(p: Paths) -> pd.DataFrame:
    """The saved CV fold of every training row, in the df-analyze train table's
    row order (= train_ids.csv order)."""
    _require(p.cv_folds, p.train_ids)
    train_ids = pd.read_csv(p.train_ids)
    folds = pd.read_csv(p.cv_folds).set_index(C.NODE_ID)["fold"]
    missing = set(train_ids[C.NODE_ID]) - set(folds.index)
    if missing:
        raise StageError(f"{len(missing)} training nodes have no CV fold; rerun split")
    return pd.DataFrame(
        {
            C.NODE_ID: train_ids[C.NODE_ID],
            "fold": folds.loc[train_ids[C.NODE_ID]].to_numpy(),
        }
    )


NESTED_SCORES = "nested_scores.parquet"


def nested_scores_file(ctx: Context, outdir: Path, export_dir: Path) -> list[str]:
    """cv_select.py's --nested-file for a feature set with a gnnsup block
    (Model 6; none otherwise): for each outer fold k, the nested score of
    every training row outside fold k, in df-analyze's row order and column
    name. cv_select puts these on the rows a fold's model is fit on, so the
    shared CV never fits on a score built with the held-out fold's labels.
    The held-out rows keep their out-of-fold score. Stops unless df-analyze
    exported the column unchanged (then the values can be put in directly)."""
    names = blocks.blocks_of_kind(feature_set_blocks(ctx), C.GNNSUP_BLOCK)
    if not names:
        return []
    p = ctx.paths
    info = read_feature_set_info(p)
    safe_of = {c["block_column"]: safe for safe, c in info["columns"].items()}
    train_ids = pd.read_csv(p.train_ids)[C.NODE_ID].to_numpy()
    row_of = pd.Series(np.arange(len(train_ids)), index=train_ids)
    exported = pd.read_csv(export_dir / "X_train_00.csv")
    ours = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    frames = []
    for name in names:
        _require(p.nested_scores(name))
        nested = pd.read_parquet(p.nested_scores(name))
        for column in [
            c for c in nested.columns if c.startswith(name + C.BLOCK_SEPARATOR)
        ]:
            safe = safe_of[column]
            if safe not in exported.columns or not np.allclose(
                exported[safe].to_numpy(float), ours[safe].to_numpy(float),
                rtol=0, atol=1e-9,
            ):  # fmt: skip
                raise StageError(
                    f"df-analyze's X_train export changed column {safe}, so its nested "
                    "out-of-fold scores cannot be put in for the shared CV. Ask before "
                    "working around it."
                )
            frames.append(pd.DataFrame({
                "row": row_of.loc[nested[C.NODE_ID].to_numpy()].to_numpy(),
                "outer_fold": nested["outer_fold"].to_numpy(),
                "column": safe,
                "value": nested[column].to_numpy(float),
            }))  # fmt: skip
    path = outdir / NESTED_SCORES
    atomic_write_parquet(path, pd.concat(frames, ignore_index=True))
    print(f"Nested out-of-fold scores for the shared CV: {path}")
    return ["--nested-file", str(path.resolve())]


def run_select_model(ctx: Context) -> None:
    """df-analyze's tuning scores are not comparable across models (and its
    internal CV is not grouped), so refit every tuned combination on the saved
    grouped folds of the training set and score it from its probabilities (see
    scripts/dfa/cv_select.py). The test set is not read. The report stage
    chooses Model A from this table. With a gnnsup block (Model 6), the rows
    each fold's model is fit on carry nested scores (nested_scores_file)."""
    if gnn_rung(ctx):
        return
    p = ctx.paths
    stamp, outdir = _latest_verified_run(p)
    export_dir = df_analyze.find_export_dir(outdir)
    cfg = ctx.config["select_model"]
    folds = folds_in_train_order(p)
    folds_path = outdir / FOLDS_IN_TRAIN_ORDER
    atomic_write_csv(folds_path, folds)
    nested_args = nested_scores_file(ctx, outdir, export_dir)
    cv_path, oof_path = outdir / SHARED_CV, outdir / OOF_PREDICTIONS
    # The CV script saves its tables after every configuration. On --resume it
    # keeps the finished ones; on a fresh run it starts over.
    if not ctx.resume:
        cv_path.unlink(missing_ok=True)
        oof_path.unlink(missing_ok=True)
    args = [
        "--export-dir", str(export_dir.resolve()),
        "--out", str(cv_path.resolve()),
        "--oof-out", str(oof_path.resolve()),
        "--folds-file", str(folds_path.resolve()),
        "--seed", str(ctx.seed),
        "--config-timeout", str(cfg.get("config_timeout_s", 0)),
    ]  # fmt: skip
    if not cfg.get("score_defaults", True):
        args += ["--settings", "tuned"]
    if ctx.resume:
        args += ["--resume"]
    args += nested_args
    _run_dfa(ctx, str(CV_SELECT_RUNNER), args, outdir / "model_selection.log")
    cv = pd.read_csv(cv_path, keep_default_na=False, na_values=[""])
    log = (outdir / "df_analyze.log").read_text(encoding="utf-8", errors="replace")
    budget = selection.parse_tuning_budget(log)
    atomic_write_csv(outdir / TUNING_BUDGET, budget)
    failed = cv["error"].fillna("").astype(str) != ""
    tuned = cv["settings"] == "tuned"
    errors = cv[failed & tuned]
    for r in cv[failed & ~tuned].itertuples():
        print(
            f"WARNING: {r.model_cls} ({r.selection}) with default settings could not be "
            f"scored, so its before-tuning score is missing: {r.error}"
        )
    if not errors.empty:
        raise StageError(
            "These tuned models could not be cross-validated, so the models cannot "
            f"be compared fairly:\n{errors[['model_cls', 'selection', 'error']]}\n"
            f"See {outdir / 'model_selection.log'}; fix and rerun "
            "`uv run run.py --dataset <name> --from select_model`."
        )
    atomic_write_csv(p.model_reports_dir / TUNING_BUDGET, budget)
    # Written last: marks the stage as done.
    atomic_write_csv(p.model_reports_dir / SHARED_CV, cv)
    # embed_selector is empty (read as NaN) except for embedded selection;
    # pivot_table would silently drop those rows.
    shown = (
        cv.assign(embed_selector=cv["embed_selector"].fillna(""))
        .pivot_table(
            index=["model_cls", "selection", "embed_selector"],
            columns="settings",
            values="pr_auc_mean",
        )
        .reset_index()
    )
    print(shown.sort_values("tuned", ascending=False).to_string(index=False))
    print(f"Saved shared-CV scores for run {stamp} to {p.model_reports_dir / SHARED_CV}")


# --------------------------------------------------------------------------
# Stage 12: Model A report: choice by shared CV, threshold from out-of-fold
# training predictions, test metrics, group-bootstrap intervals, sanity band
# --------------------------------------------------------------------------
MODEL_A_REPORT = "model_a_report.md"
MODEL_A_METRICS = "model_a_metrics.csv"
MODEL_A_SUMMARY = "model_a.json"
THRESHOLDS = "thresholds.csv"


def oof_thresholds(oof: pd.DataFrame, y_train: np.ndarray, rule: str) -> pd.DataFrame:
    """Each tuned model's decision threshold from its out-of-fold training
    predictions (select_model's oof_predictions.parquet)."""
    tuned = oof[oof["settings"] == "tuned"]
    rows = []
    for (cls, sel), frame in tuned.groupby(["model_cls", "selection"], sort=True):
        frame = frame.sort_values("row")
        if not np.array_equal(frame["row"].to_numpy(), np.arange(len(y_train))):
            raise StageError(f"{cls} ({sel}): out-of-fold predictions do not cover "
                             "every training row once")  # fmt: skip
        found = evaluation.threshold_from_oof(frame.assign(y=y_train), rule)
        model = df_analyze_report.MODEL_NAMES.get(cls, cls)
        rows.append({"model": model, "selection": sel, **found})
    return pd.DataFrame(rows)


def run_df_analyze_report(ctx: Context) -> None:
    if gnn_rung(ctx):
        return
    p = ctx.paths
    train_path = p.df_analyze_input_dir / "train.parquet"
    test_path = p.df_analyze_input_dir / "test.parquet"
    _require(train_path, test_path, p.test_ids)
    stamp, outdir = _latest_verified_run(p)
    for name in (SHARED_CV, OOF_PREDICTIONS):
        if not (outdir / name).is_file():
            raise StageError(
                f"df-analyze run {outdir} has no {name}. Run the select_model stage "
                "first so every model is scored with the same cross-validation."
            )
    cfg = ctx.config["report"]
    y_train = pd.read_parquet(train_path)[C.TARGET].to_numpy()
    y_test = pd.read_parquet(test_path)[C.TARGET].to_numpy()
    test_ids = pd.read_csv(p.test_ids)
    if not np.array_equal(test_ids[C.LABEL].to_numpy(), y_test):
        raise StageError("Labels in test_ids.csv and the test table disagree")
    shared_cv = pd.read_csv(outdir / SHARED_CV, keep_default_na=False, na_values=[""])
    oof = pd.read_parquet(outdir / OOF_PREDICTIONS)
    budget_path = outdir / TUNING_BUDGET
    budget = pd.read_csv(budget_path) if budget_path.is_file() else None
    try:
        thresholds = oof_thresholds(oof, y_train, str(cfg["threshold_rule"]))
        by_key = {(r.model, r.selection): r.threshold for r in thresholds.itertuples()}
        predictions, tuned, options = df_analyze_report.load_run(outdir)
        table = df_analyze_report.metrics_table(
            predictions, tuned, y_test, shared_cv, by_key
        )
        model_a = df_analyze_report.choose_model_a(table)
        entry = web.find_entry(predictions, model_a, df_analyze_report.MODEL_NAMES)
        prob_a = np.asarray(entry["probs_test"], dtype=float)[:, 1]
        intervals = [
            evaluation.group_bootstrap(
                y_test,
                test_ids[C.GROUP_ID].to_numpy(),
                prob_a,
                metric=metric,
                n_resamples=int(cfg["bootstrap_resamples"]),
                seed=ctx.seed,
            ).as_dict()
            for metric in ("pr_auc", "auroc")
        ]
    except (
        df_analyze_report.ReportError,
        evaluation.EvaluationError,
        web.WebReportError,
    ) as e:
        raise StageError(str(e)) from e
    if not np.isclose(intervals[0]["estimate"], model_a["pr_auc"]):
        raise StageError("Model A's bootstrap estimate differs from its test PR-AUC")
    band = ctx.data["sanity_band"]
    warning = evaluation.sanity_check(
        float(model_a["pr_auc"]), band, float(cfg["sanity_margin"])
    )
    n_folds = int(pd.read_csv(p.cv_folds)["fold"].nunique())
    info = {
        "dataset": ctx.dataset,
        "feature_set": ctx.config["feature_set"],
        "run": stamp,
        "htune_trials": options.get("htune_trials"),
        "seed": options.get("seed"),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "test_positive_rate": float(y_test.mean()),
        "cv_folds": n_folds,
        "selection_rule": "highest mean shared-CV PR-AUC on the training set "
        "(saved grouped folds)",
        "threshold_rule": f"{cfg['threshold_rule']} on out-of-fold training predictions",
        "sanity_band": band,
    }
    report = df_analyze_report.build_report(
        table, model_a, info, budget, intervals, warning
    )
    model_a_dict = {k: v.item() if hasattr(v, "item") else v for k, v in model_a.items()}
    a_key = (model_a["model"], model_a["selection"])
    threshold_a = thresholds.set_index(["model", "selection"]).loc[a_key].to_dict()
    summary = {
        **info,
        "model_a": model_a_dict,
        "model_a_threshold": threshold_a,
        "intervals": intervals,
        "sanity_warning": warning,
    }
    for directory in (outdir, p.model_reports_dir):
        for name, frame in ((MODEL_A_METRICS, table), (THRESHOLDS, thresholds)):
            atomic_write_csv(directory / name, frame)
        atomic_write_text(directory / MODEL_A_REPORT, report)
        atomic_write_json(directory / MODEL_A_SUMMARY, summary)  # last: stage done
    print(report)
    if warning:
        bar = "!" * 78
        print(f"\n{bar}\nWARNING: {warning}\n{bar}")
    print(f"Saved the report to {p.model_reports_dir / MODEL_A_REPORT}")


# --------------------------------------------------------------------------
# Stage 13: static HTML report (parameters, results, confidence cards)
# --------------------------------------------------------------------------
DATASET_LABELS = {"yelpchi": "YelpChi (reviews)", "amazon": "Amazon (users)"}


def run_web_report(ctx: Context) -> None:
    if gnn_rung(ctx):
        return
    p = ctx.paths
    summary_path = p.model_reports_dir / MODEL_A_SUMMARY
    test_path = p.df_analyze_input_dir / "test.parquet"
    train_path = p.df_analyze_input_dir / "train.parquet"
    _require(summary_path, p.model_reports_dir / MODEL_A_METRICS, p.test_ids,
             test_path, train_path)  # fmt: skip
    summary = _read_json(summary_path)
    stamp, outdir = _latest_verified_run(p)
    if summary["run"] != stamp:
        raise StageError(
            f"{summary_path} describes df-analyze run {summary['run']}, but the latest "
            f"run is {stamp}. Rerun `uv run run.py --dataset {ctx.dataset} "
            "--from select_model`."
        )
    cfg = ctx.config["web_report"]
    info = read_feature_set_info(p)
    try:
        predictions, _, _ = df_analyze_report.load_run(outdir)
        entry = web.find_entry(
            predictions, summary["model_a"], df_analyze_report.MODEL_NAMES
        )
        params = web.parse_params(entry["params"])
    except (df_analyze_report.ReportError, web.WebReportError) as e:
        raise StageError(str(e)) from e

    # Test rows are in the order of our saved test IDs (verified by df_analyze).
    test_ids = pd.read_csv(p.test_ids)
    train, test = pd.read_parquet(train_path), pd.read_parquet(test_path)
    y_test = test[C.TARGET].to_numpy()
    if not np.array_equal(test_ids[C.LABEL].to_numpy(), y_test):
        raise StageError("Test labels in test_ids.csv and test.parquet disagree")
    # Show features under their block names (own__f07), not df-analyze's.
    names = {safe: c["block_column"] for safe, c in info["columns"].items()}
    features = [c for c in train.columns if c != C.TARGET]
    prob = np.asarray(entry["probs_test"], dtype=float)[:, 1]
    try:
        top = web.top_features(
            test[features].rename(columns=names),
            train[features].rename(columns=names),
            int(cfg["n_top_features"]),
        )
        meta = pd.DataFrame({C.NODE_ID: test_ids[C.NODE_ID], "top_features": top})
        # Cards use the model's own 0.5 cut-off, so confidence is the
        # probability of the predicted class (0.5 to 1).
        samples = web.confidence_table(prob, (prob >= 0.5).astype(int), y_test, meta)
    except web.WebReportError as e:
        raise StageError(str(e)) from e

    metrics = pd.read_csv(p.model_reports_dir / MODEL_A_METRICS)
    budget_path = p.model_reports_dir / TUNING_BUDGET
    budget = pd.read_csv(budget_path) if budget_path.is_file() else None
    dfa = ctx.config["df_analyze"]
    types = pd.Series([c["type"] for c in info["columns"].values()]).value_counts()
    feature_sets = sorted(
        {web.feature_set_label(web.feature_set_key(s, e))
         for s, e in zip(metrics["selection"], metrics["embed_selector"], strict=True)}
    )  # fmt: skip
    a = summary["model_a"]
    run_config = [
        ("Dataset", DATASET_LABELS.get(ctx.dataset, ctx.dataset)),
        ("Feature set", f"{summary['feature_set']} ({', '.join(info['blocks'])})"),
        ("Seed", str(summary["seed"])),
        ("Nodes", f"{summary['n_train']:,} train · {summary['n_test']:,} test "
                  "(grouped split)"),
        ("Fraud rate", f"{train[C.TARGET].mean():.1%} train · "
                       f"{summary['test_positive_rate']:.1%} test"),
        ("Features", f"{len(features)}: " + ", ".join(f"{n} {k}" for k, n in
                                                   types.items())),
        ("Classifiers", ", ".join(web.model_label(m) for m in dfa["classifiers"])
                        + ", plus dummy baseline"),
        ("Tuning", f"up to {dfa['htune_trials']} trials per model · "
                   f"metric {dfa['htune_cls_metric']}"),
        ("Feature selections", ", ".join(feature_sets)),
        ("Model choice", f"{summary['cv_folds']}-fold grouped shared CV on train, "
                         "PR-AUC"),
        ("Threshold", f"{a['threshold']:.4f} ({summary['threshold_rule']})"),
        ("df-analyze commit", str(dfa.get("commit") or "not pinned")[:12]),
    ]  # fmt: skip
    commit = git_commit(REPO_ROOT)
    inputs = web.ReportInputs(
        summary=summary,
        metrics=metrics,
        params=params,
        n_features=len(entry["selected_cols"]),
        samples=samples,
        header={
            "build": stamp,
            "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "git_commit": commit[:10] if commit else "unknown",
            "run": stamp,
        },
        run_config=run_config,
        budget=budget,
        n_samples=int(cfg["n_samples"]),
        dataset_label=DATASET_LABELS.get(ctx.dataset, ctx.dataset),
    )
    atomic_write_text(p.web_report, web.render(inputs))
    print(f"Saved the web report to {p.web_report}")


# --------------------------------------------------------------------------
# Stage 14: Model A in its native format; check every model df-analyze saved
# --------------------------------------------------------------------------
# Last in STAGES on purpose: run.py reruns every stage after the first one
# with missing outputs, so an earlier position would rerun df-analyze.
EXPORT_RUNNER = REPO_ROOT / "scripts" / "dfa" / "export_model.py"
EXPORT_REQUEST = "request.json"


def export_outputs(p: Paths) -> list[Path]:
    """export_model's outputs: models_check.csv, model_a.json and the model
    file model_a.json names. The stage moves its finished folder into place in
    one step, so they appear together."""
    outputs = [p.model_dir / model_export.MODELS_CHECK,
               p.model_dir / model_export.MODEL_A_META]  # fmt: skip
    try:
        meta = model_export.read_metadata(outputs[1])
    except (model_export.MetadataError, ValueError):  # missing or unreadable
        return outputs
    return [*outputs, p.model_dir / meta["model_file"]]


def _repo_relative(path: Path, root: Path) -> str:
    """`path` relative to the repository root (so it holds in another clone
    that unpacked a bundle), or absolute if it lies outside."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def run_export_model(ctx: Context) -> None:
    """Check that every tuned model df-analyze saved decodes and reproduces its
    saved test probabilities, and save Model A (as chosen by df_analyze_report)
    in its native format with model_a.json, so nobody has to retrain it (see
    scripts/dfa/export_model.py). Nothing is chosen or tuned here."""
    if gnn_rung(ctx):
        return
    p = ctx.paths
    summary_path = p.model_reports_dir / MODEL_A_SUMMARY
    thresholds_path = p.model_reports_dir / THRESHOLDS
    _require(summary_path, thresholds_path)
    summary = _read_json(summary_path)
    stamp, outdir = _latest_verified_run(p)
    if summary["run"] != stamp:
        raise StageError(
            f"{summary_path} describes df-analyze run {summary['run']}, but the latest "
            f"run is {stamp}. Rerun `uv run run.py --dataset {ctx.dataset} "
            "--from select_model`."
        )
    a = summary["model_a"]
    staging = p.model_dir.with_name(p.model_dir.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    request = {
        "export_dir": str(df_analyze.find_export_dir(outdir).resolve()),
        "out_dir": str(staging.resolve()),
        "results_dir": _repo_relative(df_analyze.find_export_dir(outdir), p.root),
        "dataset": ctx.dataset,
        "feature_set": ctx.config["feature_set"],
        "run": stamp,
        "model_a": {k: a[k] for k in ("model", "selection", "embed_selector")},
        "threshold": a["threshold"],
        "threshold_rule": summary["threshold_rule"],
        "thresholds": pd.read_csv(thresholds_path).to_dict("records"),
        "seed": ctx.seed,
        "df_analyze_commit": ctx.config["df_analyze"].get("commit"),
    }
    atomic_write_json(staging / EXPORT_REQUEST, request)
    args = ["--request", str((staging / EXPORT_REQUEST).resolve())]
    _run_dfa(ctx, str(EXPORT_RUNNER), args, outdir / "export_model.log")
    (staging / EXPORT_REQUEST).unlink()
    try:
        meta = model_export.read_metadata(staging / model_export.MODEL_A_META)
    except model_export.MetadataError as e:
        raise StageError(str(e)) from e
    _require(staging / meta["model_file"], staging / model_export.MODELS_CHECK)
    # Replace the previous export in one step: the stage's outputs appear only
    # when all of them are complete.
    shutil.rmtree(p.model_dir, ignore_errors=True)
    os.replace(staging, p.model_dir)

    check = pd.read_csv(p.model_dir / model_export.MODELS_CHECK, keep_default_na=False)
    shown = ["model", "selection_label", "decoded", "intact", "max_abs_diff", "ok"]
    print(check[shown].to_string(index=False))
    failed = check[check["ok"].astype(str) != "True"]
    for r in failed.itertuples():
        print(
            f"WARNING: {r.model} ({r.selection_label}) cannot be loaded from "
            f"df-analyze's saved results: {r.error}. It was not refit."
        )
    print(
        f"Model A ({meta['model']}, {meta['selection_label']}, {meta['source']}) saved "
        f"to {p.model_dir / meta['model_file']}; {len(check) - len(failed)} of "
        f"{len(check)} tuned models load and reproduce their test probabilities."
    )


# --------------------------------------------------------------------------
# Stage 15: GNN classifier rungs (Models 3-4: E5, E5g, E6, N3)
# --------------------------------------------------------------------------
# After export_model: it has outputs only for feature sets in gnn_models, so
# no finished df-analyze run (or bundle) ever reruns anything because of it.
def gnn_classify_outputs(p: Paths) -> list[Path]:
    if not p.is_gnn:
        return []
    r = p.model_reports_dir
    return [r / gnn_outputs.GNN_OOF, r / gnn_outputs.GNN_TEST, r / MODEL_A_METRICS,
            r / MODEL_A_SUMMARY]  # fmt: skip


def _cross_fit_unit(
    store: UnitStore, unit: str, sup: Any, g: Any, data: GNNData, setting: Any,
    cfg: Any, seed: int,
) -> dict[str, Any]:  # fmt: skip
    """One (setting, seed): out-of-fold probabilities of the training rows and
    the fold models' mean for the test nodes; kept on --resume."""
    if store.done(unit):
        print(f"  {unit}: finished in the interrupted run; kept")
        return store.load(unit)
    start = time.monotonic()
    fit = sup.cross_fit(g, data.train_nodes, data.y_train, data.folds, data.groups,
                        setting, cfg, seed)  # fmt: skip
    result = {
        "setting": asdict(setting),
        "seed": seed,
        "oof": fit.oof.tolist(),
        "test": fit.all_nodes[data.test_nodes].tolist(),
        "fold_pr_auc": sup.fold_pr_auc(data.y_train, fit.oof, data.folds),
        "logs": fit.logs,
    }
    store.save(unit, result)
    epochs = [log["best_epoch"] for log in fit.logs]
    print(
        f"  {unit}: out-of-fold PR-AUC {np.mean(result['fold_pr_auc']):.4f} "
        f"(best epochs {epochs}; {time.monotonic() - start:.0f} s)",
        flush=True,
    )
    return result


def run_gnn_classify(ctx: Context) -> None:
    """Train and score a GNN classifier rung (docs/GNN_MODELS_PLAN.md §2):
    the setting from the grid by mean out-of-fold PR-AUC on the saved folds
    (seed 0), or another rung's setting on this split (setting_from); then 5
    seeds, each cross-fit on the saved folds (one unit per setting and seed).
    Probabilities are averaged over seeds; the threshold comes from the
    out-of-fold ones; test metrics and group-bootstrap intervals as for a
    df-analyze Model A. Test labels are read only to score."""
    fs = ctx.config["feature_set"]
    spec = (ctx.config.get("gnn_models") or {}).get(fs)
    if spec is None:
        print(f"Feature set {fs} is not a GNN classifier rung (gnn_models), so there "
              "is nothing to do here.")  # fmt: skip
        return
    sup = _import_supervised()
    cfg = _train_config(ctx, sup, spec)
    data = gnn_data(ctx, feature_set_blocks(ctx)[0])
    g = sup.graph_input(data.x, gnn_edges(ctx, spec["graph"], len(data.x)),
                        spec.get("norm", "mean"), cfg.device, cfg.model)  # fmt: skip
    label = gnn_model_label(spec)
    shape = {"sign": f"{cfg.hops} hops, ", "gat": f"{cfg.heads} heads, "}.get(
        cfg.model, ""
    )
    weights = (
        "attention" if cfg.model == "gat" else f"{spec.get('norm', 'mean')} normalisation"
    )
    print(
        f"{fs}: {label}, {shape}{cfg.layers} layers, hidden {list(cfg.hidden)}, "
        f"relations {g.relations or 'none'} "
        f"({spec['graph']} graph, {weights}), "
        f"{len(data.own_columns)} own features, {len(data.x):,} nodes; cross-fit on "
        f"{len(np.unique(data.folds))} saved folds on {cfg.device}",
        flush=True,
    )
    store = ctx.unit_store("gnn_classify")
    grid_rows: list[dict[str, Any]] = []
    grid_results: dict[str, dict[str, Any]] = {}
    if spec.get("setting_from"):
        source = spec["setting_from"]
        setting = sup.Setting(**gnn_rung_summary(ctx, source)["gnn"]["setting"])
        setting_rule = f"{source}'s chosen setting on this split"
    else:
        for candidate in cfg.grid():
            result = _cross_fit_unit(store, f"{fs}/grid/{candidate.key}", sup, g, data,
                                     candidate, cfg, ctx.seed)  # fmt: skip
            grid_results[candidate.key] = result
            folds_ap = result["fold_pr_auc"]
            grid_rows.append(
                {
                    **asdict(candidate),
                    "setting": candidate.key,
                    **{f"fold{k}_pr_auc": v for k, v in enumerate(folds_ap)},
                    "cv_pr_auc": float(np.mean(folds_ap)),
                    "cv_pr_auc_std": float(np.std(folds_ap, ddof=1)),
                }
            )
        means = [r["cv_pr_auc"] for r in grid_rows]
        setting = cfg.grid()[int(np.argmax(means))]  # ties: the earlier setting
        setting_rule = "grid, highest mean out-of-fold PR-AUC (seed 0, saved folds)"
    seeds = [ctx.seed + s for s in range(cfg.seeds)]
    per_seed = []
    for seed in seeds:
        if seed == ctx.seed and setting.key in grid_results:
            per_seed.append(grid_results[setting.key])  # the grid's fit, same seed
            continue
        per_seed.append(_cross_fit_unit(store, f"{fs}/seed{seed}/{setting.key}", sup, g,
                                        data, setting, cfg, seed))  # fmt: skip
    oof = np.mean([r["oof"] for r in per_seed], axis=0)
    prob = np.mean([r["test"] for r in per_seed], axis=0)
    _write_gnn_rung(ctx, spec, label, data, setting, setting_rule, seeds, per_seed,
                    oof, prob, grid_rows, g, cfg, sup)  # fmt: skip


def _write_gnn_rung(
    ctx: Context, spec: dict[str, Any], label: str, data: GNNData, setting: Any,
    setting_rule: str, seeds: list[int], per_seed: list[dict[str, Any]],
    oof: np.ndarray, prob: np.ndarray, grid_rows: list[dict[str, Any]], g: Any,
    cfg: Any, sup: Any,
) -> None:  # fmt: skip
    """Score a GNN rung as df_analyze_report scores Model A and write its
    outputs (model_a.json last: it marks the stage as done)."""
    p, report = ctx.paths, ctx.config["report"]
    fold_ap = sup.fold_pr_auc(data.y_train, oof, data.folds)
    fold_auroc = [
        float(roc_auc_score(data.y_train[data.folds == k], oof[data.folds == k]))
        for k in sorted(np.unique(data.folds))
    ]
    try:
        threshold = evaluation.threshold_from_oof(
            pd.DataFrame({"fold": data.folds, "prob": oof, "y": data.y_train}),
            str(report["threshold_rule"]),
        )
    except evaluation.EvaluationError as e:
        raise StageError(str(e)) from e
    # Scoring only, after every fit: the test labels and groups.
    test_ids = pd.read_csv(p.test_ids)
    if not np.array_equal(test_ids[C.NODE_ID].to_numpy(), data.test_nodes):
        raise StageError("test_ids.csv changed while the GNN was trained")
    y_test = test_ids[C.LABEL].to_numpy(dtype=int)
    metrics = df_analyze_report.test_metrics(y_test, prob, threshold["threshold"])
    intervals = [
        evaluation.group_bootstrap(
            y_test,
            test_ids[C.GROUP_ID].to_numpy(),
            prob,
            metric=metric,
            n_resamples=int(report["bootstrap_resamples"]),
            seed=ctx.seed,
        ).as_dict()
        for metric in ("pr_auc", "auroc")
    ]
    seed_rows = [
        {"seed": r["seed"], "cv_pr_auc": float(np.mean(r["fold_pr_auc"])),
         "test_pr_auc": float(average_precision_score(y_test, r["test"])),
         "mean_best_epoch": float(np.mean([log["best_epoch"] for log in r["logs"]]))}
        for r in per_seed
    ]  # fmt: skip
    model_a = {
        "model": label,
        "selection": "gnn",
        "embed_selector": "",
        "cv_pr_auc": float(np.mean(fold_ap)),
        "cv_pr_auc_std": float(np.std(fold_ap, ddof=1)),
        "cv_auroc": float(np.mean(fold_auroc)),
        "cv_auroc_std": float(np.std(fold_auroc, ddof=1)),
        "tuning_metric": "out-of-fold PR-AUC (grid); inner-validation PR-AUC "
        "(early stopping)",
        **metrics,
    }
    warning = evaluation.sanity_check(float(metrics["pr_auc"]), ctx.data["sanity_band"],
                                      float(report["sanity_margin"]))  # fmt: skip
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "dataset": ctx.dataset,
        "feature_set": ctx.config["feature_set"],
        "source": gnn_outputs.GNN_SOURCE,
        "run": f"gnn-{stamp}",
        "htune_trials": None,
        "seed": ctx.seed,
        "n_train": len(data.train_nodes),
        "n_test": len(data.test_nodes),
        "test_positive_rate": float(y_test.mean()),
        "cv_folds": len(fold_ap),
        "selection_rule": setting_rule,
        "threshold_rule": f"{report['threshold_rule']} on out-of-fold training "
        "predictions",
        "sanity_band": ctx.data["sanity_band"],
        "model_a": model_a,
        "model_a_threshold": threshold,
        "intervals": intervals,
        "sanity_warning": warning,
        "gnn": {
            "spec": dict(spec),
            "setting": asdict(setting),
            "seeds": seeds,
            "fold_pr_auc": fold_ap,
            "per_seed": seed_rows,
            "relations": g.relations,
            "n_edges": g.n_edges,
            "norm": g.norm,
            "config": asdict(cfg),
            "own_columns": data.own_columns,
            "rewired_graph": _repo_relative(p.rewired_dir(ctx.split_seed), p.root)
            if spec["graph"] == "rewired"
            else None,
            "versions": {k: v for k, v in package_versions().items() if k == "torch"},
        },
    }
    r = p.model_reports_dir
    atomic_write_parquet(r / gnn_outputs.GNN_OOF, pd.DataFrame(
        {"row": np.arange(len(oof)), "fold": data.folds, "prob": oof}))  # fmt: skip
    atomic_write_parquet(r / gnn_outputs.GNN_TEST, pd.DataFrame(
        {C.NODE_ID: data.test_nodes, "prob": prob}))  # fmt: skip
    if grid_rows:
        atomic_write_csv(r / gnn_outputs.GNN_GRID, pd.DataFrame(grid_rows))
    atomic_write_csv(r / gnn_outputs.GNN_SEEDS, pd.DataFrame(seed_rows))
    atomic_write_csv(r / MODEL_A_METRICS, pd.DataFrame([model_a]))
    lines = [
        f"# {label}: {ctx.config['feature_set']} ({ctx.dataset})",
        "",
        f"Setting {setting.key} ({setting_rule}); seeds {seeds}; threshold "
        f"{threshold['threshold']:.4f} ({summary['threshold_rule']}).",
        "",
        f"- Shared-CV PR-AUC (out-of-fold, seed-averaged): {model_a['cv_pr_auc']:.4f} "
        f"± {model_a['cv_pr_auc_std']:.4f}",
        f"- Test PR-AUC: {metrics['pr_auc']:.4f} [{intervals[0]['low']:.4f}, "
        f"{intervals[0]['high']:.4f}] (no skill {summary['test_positive_rate']:.4f})",
        f"- Test AUROC: {metrics['auroc']:.4f} [{intervals[1]['low']:.4f}, "
        f"{intervals[1]['high']:.4f}]",
        f"- F1 / precision / recall at the threshold: {metrics['f1']:.4f} / "
        f"{metrics['precision']:.4f} / {metrics['recall']:.4f}",
        "- Per seed (test PR-AUC): "
        + ", ".join(f"{s['seed']}: {s['test_pr_auc']:.4f}" for s in seed_rows),
    ]
    if warning:
        lines += ["", f"WARNING: {warning}"]
    atomic_write_text(r / MODEL_A_REPORT, "\n".join(lines) + "\n")
    atomic_write_json(r / MODEL_A_SUMMARY, summary)  # last: stage done
    print("\n".join(lines))


STAGES: list[Stage] = [
    # A clone that unpacked a bundle has the load stage's outputs but no raw
    # files; it has nothing left to download.
    Stage("download", "Fetch and checksum the CARE-GNN .mat file", run_download,
          lambda p: [p.mat], skip_when=lambda p: [p.nodes, p.graph_manifest]),
    Stage("load", "Node table, relation edge lists, count checks", run_load,
          lambda p: [p.nodes, p.graph_manifest]),
    Stage("split", "Grouped stratified train/test split and CV folds", run_split,
          lambda p: [p.train_ids, p.test_ids, p.cv_folds, p.split_summary]),
    Stage("features", "Own-feature block and column types (training nodes)",
          run_features, lambda p: [p.block_file(C.OWN_BLOCK), p.column_spec]),
    Stage("audit", "Counts, feature statistics, duplicates, homophily", run_audit,
          lambda p: [p.reports_dir / AUDIT_REPORT,
                     *(p.reports_dir / t for t in AUDIT_TABLES)]),
    # The graph blocks come right after audit and count as done when the
    # active feature set does not use them (see block_outputs).
    Stage("neighbour_agg", "nbr block: label-free neighbour means per relation "
          "(and the degree-only deg control)", run_neighbour_agg,
          lambda p: [f for k in C.NBR_BLOCK_KINDS for f in block_outputs(k)(p)]),
    Stage("gnn_embed", "gnnssl block: self-supervised DGI embeddings per relation "
          "(and the untrained gnnrand control)", run_gnn_embed, gnn_embed_outputs),
    Stage("gnn_oof", "gnnsup block: the best supervised GNN's out-of-fold score "
          "(Model 6, nested cross-fitting)", run_gnn_oof, gnn_oof_outputs),
    Stage("df_analyze_input", "df-analyze train/test tables for the feature set",
          run_df_analyze_input,
          lambda p: [p.df_analyze_input_dir / "train.parquet",
                     p.df_analyze_input_dir / "test.parquet",
                     p.df_analyze_input_dir / FEATURE_SET_INFO]),
    # The df-analyze stages have no outputs for a GNN classifier rung, which
    # gnn_classify scores instead.
    Stage("df_analyze", "Run df-analyze (Model A); verify split and types",
          run_df_analyze,
          _df_analyze_only(lambda p: [p.model_reports_dir / SPLIT_CHECK,
                                      p.model_reports_dir / TYPE_CHECK])),
    Stage("select_model", "Score every tuned model on the grouped CV folds",
          run_select_model,
          _df_analyze_only(lambda p: [p.model_reports_dir / SHARED_CV])),
    Stage("df_analyze_report", "Model A test metrics, threshold, bootstrap CIs",
          run_df_analyze_report,
          _df_analyze_only(lambda p: [p.model_reports_dir / MODEL_A_REPORT,
                                      p.model_reports_dir / MODEL_A_METRICS,
                                      p.model_reports_dir / MODEL_A_SUMMARY])),
    Stage("web_report", "HTML report: parameters, results, confidence cards",
          run_web_report, _df_analyze_only(lambda p: [p.web_report])),
    Stage("export_model", "Save Model A natively; check every saved model",
          run_export_model, _df_analyze_only(export_outputs)),
    Stage("gnn_classify", "GNN classifier rungs (GraphSAGE, R-GCN-style, MLP): "
          "grid, 5 seeds, cross-fit scores", run_gnn_classify, gnn_classify_outputs),
]  # fmt: skip

STAGE_NAMES = [s.name for s in STAGES]


def get_stage(name: str) -> Stage:
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise KeyError(f"Unknown stage {name!r}; choose from {STAGE_NAMES}")
