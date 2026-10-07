"""Pipeline stages, in order. `run.py` and `scripts/NN_*.py` both call these.

Each stage reads its inputs from data/ or outputs/ (under the active
dataset's subfolder) and writes its outputs there, so any stage can be rerun
on its own. New stages (graph features, GNNs, evaluation) are added to STAGES
as they are implemented.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fraud_detect import columns as C
from fraud_detect.config import Paths, dataset_config
from fraud_detect.data import care_gnn, download
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


STAGES: list[Stage] = [
    Stage("download", "Fetch and checksum the CARE-GNN .mat file", run_download,
          lambda p: [p.mat]),
    Stage("load", "Node table, relation edge lists, count checks", run_load,
          lambda p: [p.nodes, p.graph_manifest]),
    Stage("split", "Grouped stratified train/test split and CV folds",
          _not_implemented("M4"),
          lambda p: [p.train_ids, p.test_ids, p.cv_folds, p.split_summary]),
    Stage("features", "Own-feature block and column types (training nodes)",
          _not_implemented("M4"), lambda p: [p.block_file("own"), p.column_spec]),
    Stage("audit", "Counts, feature statistics, duplicates, homophily",
          _not_implemented("M4"), lambda p: [p.reports_dir / "audit.md"]),
    Stage("df_analyze_input", "df-analyze train/test tables for the feature set",
          _not_implemented("M5"),
          lambda p: [p.df_analyze_input_dir / "train.parquet",
                     p.df_analyze_input_dir / "test.parquet"]),
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
