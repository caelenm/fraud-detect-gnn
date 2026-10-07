"""Pipeline stages, in order. `run.py` and `scripts/NN_*.py` both call these.

Each stage reads its inputs from data/ or outputs/ (under the active
dataset's subfolder) and writes its outputs there, so any stage can be rerun
on its own. New stages (graph features, GNNs, evaluation) are added to STAGES
as they are implemented.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detect import columns as C
from fraud_detect.config import Paths, dataset_config
from fraud_detect.data import audit, care_gnn, download, split
from fraud_detect.features import blocks, own
from fraud_detect.runstate import (
    CHECKPOINT_DIR,
    UnitStore,
    atomic_write,
    atomic_write_json,
    atomic_write_parquet,
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


def _require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            raise StageError(f"Missing input {p}. Run the earlier stages first.")


def _not_implemented(milestone: str) -> Callable[[Context], None]:
    def run(ctx: Context) -> None:
        raise StageError(
            f"This stage is not implemented yet (docs/RESEARCH_PLAN.md, {milestone})."
        )

    return run


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
    _require(p.mat)
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
        nodes = split.split_nodes(nodes, cfg, ctx.seed)
        summary = split.summary(nodes, cfg, ctx.seed)
    except split.SplitError as e:
        raise StageError(str(e)) from e
    atomic_write_parquet(p.nodes, nodes)
    for path, table in (
        (p.train_ids, split.ids_table(nodes, C.TRAIN)),
        (p.test_ids, split.ids_table(nodes, C.TEST)),
        (p.cv_folds, split.folds_table(nodes)),
    ):
        atomic_write(path, lambda tmp, table=table: table.to_csv(tmp, index=False))
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
    _require(p.mat, p.train_ids)
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
        atomic_write(p.reports_dir / name,
                     lambda tmp, table=table: table.to_csv(tmp, index=False))  # fmt: skip
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
# Stage 6: df-analyze input tables for the active feature set
# --------------------------------------------------------------------------
FEATURE_SET_INFO = "feature_set.json"


def feature_set_blocks(ctx: Context) -> list[str]:
    return list(ctx.config["feature_sets"][ctx.config["feature_set"]])


def run_df_analyze_input(ctx: Context) -> None:
    p = ctx.paths
    _require(p.train_ids, p.test_ids, p.column_spec)
    nodes = read_nodes(p)
    names = feature_set_blocks(ctx)
    try:
        features = blocks.load_feature_set(p.features_dir, names)
        train_ids, test_ids = pd.read_csv(p.train_ids), pd.read_csv(p.test_ids)
        train, test = blocks.model_tables(features, train_ids, test_ids)
    except blocks.BlockError as e:
        raise StageError(str(e)) from e
    # The labels in the ID files must be the node table's labels.
    by_id = nodes.set_index(C.NODE_ID)[C.LABEL]
    for ids, table in ((train_ids, train), (test_ids, test)):
        if not np.array_equal(by_id.loc[ids[C.NODE_ID]].to_numpy(int), table[C.TARGET]):
            raise StageError("Labels in the split files differ from nodes.parquet")
    spec = _read_json(p.column_spec)["columns"]
    feature_cols = [c for c in train.columns if c != C.TARGET]
    untyped = [c for c in feature_cols if c not in spec]
    if untyped:
        raise StageError(f"No column type in {p.column_spec} for: {untyped}")
    info = {
        "feature_set": ctx.config["feature_set"],
        "blocks": names,
        "n_train": len(train),
        "n_test": len(test),
        "columns": {c: spec[c]["type"] for c in feature_cols},
    }
    atomic_write_parquet(p.df_analyze_input_dir / "train.parquet", train)
    atomic_write_parquet(p.df_analyze_input_dir / "test.parquet", test)
    atomic_write_json(p.df_analyze_input_dir / FEATURE_SET_INFO, info)
    print(
        f"Feature set {info['feature_set']} ({', '.join(names)}): "
        f"{len(feature_cols)} features; train {train.shape}, test {test.shape}"
    )


STAGES: list[Stage] = [
    Stage("download", "Fetch and checksum the CARE-GNN .mat file", run_download,
          lambda p: [p.mat]),
    Stage("load", "Node table, relation edge lists, count checks", run_load,
          lambda p: [p.nodes, p.graph_manifest]),
    Stage("split", "Grouped stratified train/test split and CV folds", run_split,
          lambda p: [p.train_ids, p.test_ids, p.cv_folds, p.split_summary]),
    Stage("features", "Own-feature block and column types (training nodes)",
          run_features, lambda p: [p.block_file(C.OWN_BLOCK), p.column_spec]),
    Stage("audit", "Counts, feature statistics, duplicates, homophily", run_audit,
          lambda p: [p.reports_dir / AUDIT_REPORT,
                     *(p.reports_dir / t for t in AUDIT_TABLES)]),
    Stage("df_analyze_input", "df-analyze train/test tables for the feature set",
          run_df_analyze_input,
          lambda p: [p.df_analyze_input_dir / "train.parquet",
                     p.df_analyze_input_dir / "test.parquet",
                     p.df_analyze_input_dir / FEATURE_SET_INFO]),
    Stage("df_analyze", "Run df-analyze (Model A); verify split and types",
          _not_implemented("M6"), lambda p: [p.model_reports_dir / "split_check.json"]),
    Stage("select_model", "Score every tuned model on the grouped CV folds",
          _not_implemented("M6"),
          lambda p: [p.model_reports_dir / "model_selection_cv.csv"]),
    Stage("df_analyze_report", "Model A test metrics, threshold, bootstrap CIs",
          _not_implemented("M7"), lambda p: [p.model_reports_dir / "model_a.json"]),
    Stage("web_report", "HTML report: parameters, results, confidence cards",
          _not_implemented("M7"), lambda p: [p.web_report]),
]  # fmt: skip

STAGE_NAMES = [s.name for s in STAGES]


def get_stage(name: str) -> Stage:
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise KeyError(f"Unknown stage {name!r}; choose from {STAGE_NAMES}")
