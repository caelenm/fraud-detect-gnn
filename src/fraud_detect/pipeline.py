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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detect import columns as C
from fraud_detect.config import REPO_ROOT, Paths, dataset_config, df_analyze_dir
from fraud_detect.data import audit, care_gnn, download, split
from fraud_detect.external import ExternalToolError, run_df_analyze_script
from fraud_detect.features import blocks, own
from fraud_detect.models import df_analyze, df_analyze_report, evaluation, selection
from fraud_detect.report import web
from fraud_detect.runlog import git_commit
from fraud_detect.runstate import (
    CHECKPOINT_DIR,
    UnitStore,
    atomic_write,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
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
# Stage 7: run df-analyze; verify its split and its column types
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


def run_df_analyze(ctx: Context) -> None:
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
# Stage 8: score every tuned model on the saved grouped CV folds (train only)
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


def run_select_model(ctx: Context) -> None:
    """df-analyze's tuning scores are not comparable across models (and its
    internal CV is not grouped), so refit every tuned combination on the saved
    grouped folds of the training set and score it from its probabilities (see
    scripts/dfa/cv_select.py). The test set is not read. The report stage
    chooses Model A from this table."""
    p = ctx.paths
    stamp, outdir = _latest_verified_run(p)
    export_dir = df_analyze.find_export_dir(outdir)
    cfg = ctx.config["select_model"]
    folds = folds_in_train_order(p)
    folds_path = outdir / FOLDS_IN_TRAIN_ORDER
    atomic_write_csv(folds_path, folds)
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
# Stage 9: Model A report: choice by shared CV, threshold from out-of-fold
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
# Stage 10: static HTML report (parameters, results, confidence cards)
# --------------------------------------------------------------------------
DATASET_LABELS = {"yelpchi": "YelpChi (reviews)", "amazon": "Amazon (users)"}


def run_web_report(ctx: Context) -> None:
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
          run_df_analyze, lambda p: [p.model_reports_dir / SPLIT_CHECK,
                                     p.model_reports_dir / TYPE_CHECK]),
    Stage("select_model", "Score every tuned model on the grouped CV folds",
          run_select_model, lambda p: [p.model_reports_dir / SHARED_CV]),
    Stage("df_analyze_report", "Model A test metrics, threshold, bootstrap CIs",
          run_df_analyze_report,
          lambda p: [p.model_reports_dir / MODEL_A_REPORT,
                     p.model_reports_dir / MODEL_A_METRICS,
                     p.model_reports_dir / MODEL_A_SUMMARY]),
    Stage("web_report", "HTML report: parameters, results, confidence cards",
          run_web_report, lambda p: [p.web_report]),
]  # fmt: skip

STAGE_NAMES = [s.name for s in STAGES]


def get_stage(name: str) -> Stage:
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise KeyError(f"Unknown stage {name!r}; choose from {STAGE_NAMES}")
