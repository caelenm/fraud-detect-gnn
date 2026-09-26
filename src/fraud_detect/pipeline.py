"""Pipeline stages, in order. `run.py` and `scripts/NN_*.py` both call these.

Each stage reads its inputs from data/ or outputs/ and writes its outputs
there, so any stage can be rerun on its own. New stages (graph, GNN,
evaluation, explanation) are added to STAGES as they are implemented.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detect import columns as C
from fraud_detect.config import REPO_ROOT, Paths, df_analyze_dir, load_yaml
from fraud_detect.data import dedup, label, load, sample
from fraud_detect.external import ExternalToolError, run_df_analyze_script
from fraud_detect.features import tabular, text
from fraud_detect.models import df_analyze
from fraud_detect.runlog import write_json


class StageError(RuntimeError):
    """A stage cannot run; the message says what to do."""


@dataclass(frozen=True)
class Context:
    config: dict[str, Any]
    paths: Paths
    run_dir: Path

    @property
    def seed(self) -> int:
        return int(self.config["seed"])


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


def _read_ids(path: Path) -> pd.Series:
    return pd.read_csv(path)[C.COMPLAINT_ID]


# --------------------------------------------------------------------------
# Stage 1: load
# --------------------------------------------------------------------------
def run_load(ctx: Context) -> None:
    cfg = ctx.config["load"]
    files = load.find_raw_files(ctx.paths.raw_dir, cfg["archives"])
    date_min = pd.Timestamp(cfg["date_min"])
    date_max = pd.Timestamp(cfg["date_max"])
    frames, summaries = [], []
    for archive, paths in files.items():
        for path in paths:
            print(f"Reading {archive}/{path.name}", flush=True)
            frame, summary = load.read_archive_csv(
                path, archive, date_min, date_max, int(cfg["chunksize"])
            )
            frames.append(frame)
            summaries.append(summary)
    df, n_exact_dupes = load.combine_archives(frames)

    ctx.paths.complaints.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ctx.paths.complaints, index=False)

    reports = ctx.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([s.__dict__ for s in summaries]).to_csv(
        reports / "raw_files.csv", index=False
    )
    categories = None
    if ctx.paths.categories_file.exists():
        categories = label.CategoryConfig.from_dict(load_yaml(ctx.paths.categories_file))
    label.candidate_report(df, categories).to_csv(
        reports / "category_values.csv", index=False
    )
    df[C.PRODUCT].value_counts(dropna=False).rename("n_complaints").to_csv(
        reports / "product_counts.csv"
    )
    write_json(
        reports / "load_summary.json",
        {
            "n_complaints_with_narrative_in_range": len(df),
            "n_exact_duplicate_rows_dropped": n_exact_dupes,
            "date_min": str(df[C.DATE_RECEIVED].min().date()),
            "date_max": str(df[C.DATE_RECEIVED].max().date()),
        },
    )
    print(
        f"Kept {len(df):,} complaints with narratives. Review "
        f"{reports / 'category_values.csv'} before running the label stage."
    )


# --------------------------------------------------------------------------
# Stage 2: label
# --------------------------------------------------------------------------
def run_label(ctx: Context) -> None:
    _require(ctx.paths.complaints, ctx.paths.categories_file)
    df = pd.read_parquet(ctx.paths.complaints)
    categories = label.CategoryConfig.from_dict(load_yaml(ctx.paths.categories_file))
    try:
        label.validate_categories(df, categories)
    except label.LabelConfigError as e:
        raise StageError(f"{ctx.paths.categories_file}\n{e}") from e
    labeled = label.apply_labels(df, categories)
    labeled.to_parquet(ctx.paths.labeled, index=False)
    summary = (
        labeled.groupby(C.PRODUCT)[C.LABEL]
        .agg(n_complaints="size", n_positive="sum", positive_rate="mean")
        .reset_index()
    )
    summary.to_csv(ctx.paths.reports_dir / "label_summary.csv", index=False)
    print(summary.to_string(index=False))


# --------------------------------------------------------------------------
# Stage 3: deduplicate and sample
# --------------------------------------------------------------------------
def run_sample(ctx: Context) -> None:
    _require(ctx.paths.labeled)
    cfg, dcfg = ctx.config["sample"], ctx.config["dedup"]
    df = pd.read_parquet(ctx.paths.labeled)
    n = int(cfg["n_complaints"])
    pool_n = min(len(df), math.ceil(n * float(cfg["pool_factor"])))
    pool = sample.stratified_sample(df, pool_n, ctx.seed)
    print(f"Deduplicating a stratified pool of {pool_n:,} complaints", flush=True)
    result = dedup.remove_near_duplicates(
        pool,
        shingle_words=int(dcfg["shingle_words"]),
        threshold=float(dcfg["jaccard_threshold"]),
        num_perm=int(dcfg["num_perm"]),
        seed=ctx.seed,
    )
    if len(result.kept) < n:
        raise StageError(
            f"Only {len(result.kept):,} complaints remain after deduplication; "
            f"increase sample.pool_factor (currently {cfg['pool_factor']})."
        )
    final = sample.stratified_sample(result.kept, n, ctx.seed)
    final.to_parquet(ctx.paths.sample, index=False)
    write_json(
        ctx.paths.reports_dir / "sample_summary.json",
        {
            "n_labeled_population": len(df),
            "population_positive_rate": float(df[C.LABEL].mean()),
            "pool_size": pool_n,
            "near_duplicates_removed_from_pool": result.n_removed,
            "duplicate_groups": result.n_groups_with_duplicates,
            "duplicate_groups_with_mixed_labels": result.n_groups_with_mixed_labels,
            "n_sample": len(final),
            "sample_positive_rate": float(final[C.LABEL].mean()),
        },
    )
    print(f"Sampled {len(final):,} complaints; positive rate {final[C.LABEL].mean():.4f}")


# --------------------------------------------------------------------------
# Stage 4: train/test split
# --------------------------------------------------------------------------
def run_split(ctx: Context) -> None:
    _require(ctx.paths.sample)
    df = pd.read_parquet(ctx.paths.sample)
    split = sample.stratified_split(df, float(ctx.config["split"]["test_size"]), ctx.seed)
    ctx.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    split.to_csv(ctx.paths.split, index=False)
    sample.ids_for(split, "train").to_frame().to_csv(ctx.paths.train_ids, index=False)
    sample.ids_for(split, "test").to_frame().to_csv(ctx.paths.test_ids, index=False)
    merged = split.merge(df[[C.COMPLAINT_ID, C.LABEL]], on=C.COMPLAINT_ID)
    summary = merged.groupby("split")[C.LABEL].agg(n="size", positive_rate="mean")
    summary.to_csv(ctx.paths.reports_dir / "split_summary.csv")
    print(summary.to_string())


# --------------------------------------------------------------------------
# Stage 5: embed narratives with df-embed's code (on the GPU by default)
# --------------------------------------------------------------------------
# df-analyze models that train on the GPU when CUDA is available.
GPU_CLASSIFIERS = frozenset({"catboost", "gandalf"})
GPU_EMBED_RUNNER = REPO_ROOT / "scripts" / "dfa" / "embed_on_device.py"
CUDA_HELP = """df-analyze's environment cannot see a CUDA GPU. Check that:
  - `nvidia-smi` works in this shell (inside a toolbox/container the NVIDIA
    driver's user-space libraries must be visible there too), and
  - this prints True:
      uv run --python '{python}' --directory {dfa} \\
        python -c "import torch; print(torch.cuda.is_available())"
To run on the CPU instead, set `gpu.require: false` in the config."""


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


def check_cuda(ctx: Context, key: str, purpose: str) -> bool:
    """Ask df-analyze's environment whether it can use a CUDA GPU, log the answer
    to the run directory as cuda_info_<key>.json, and stop if a GPU is required
    but missing."""
    report = ctx.run_dir / f"cuda_info_{key}.json"
    args = ["--cuda-info", "--report", str(report.resolve())]
    _run_dfa(ctx, str(GPU_EMBED_RUNNER), args, ctx.run_dir / f"cuda_info_{key}.log")
    info = json.loads(report.read_text(encoding="utf-8"))
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


def run_embed(ctx: Context) -> None:
    _require(ctx.paths.sample)
    dfa = df_analyze_dir(ctx.config)
    python = ctx.config["df_analyze"].get("python")
    if not (dfa / text.DF_EMBED_MODEL_DIR).is_dir():
        raise StageError(
            f"The df-embed NLP model is not downloaded in {dfa}. Run once:\n"
            f"  uv run --python '{python}' --directory {dfa} "
            "python df-embed.py --download --modality nlp"
        )
    cfg = ctx.config["embed"]
    df = pd.read_parquet(ctx.paths.sample)
    ctx.paths.embed_dir.mkdir(parents=True, exist_ok=True)
    input_path = ctx.paths.embed_dir / "embed_input.parquet"
    output_path = ctx.paths.embed_dir / "embed_output.parquet"
    text.df_embed_input(df).to_parquet(input_path, index=False)
    output_path.unlink(missing_ok=True)
    io_args = [
        "--data", str(input_path.resolve()),
        "--out", str(output_path.resolve()),
        "--batch-size", str(cfg["batch_size"]),
    ]  # fmt: skip
    if cfg["runner"] == "gpu":
        # df-embed's own code, run on the GPU when available (see the script).
        use_gpu = check_cuda(ctx, "embed", "embedding")
        report = ctx.run_dir / "embed_report.json"
        args = [
            *io_args,
            "--device", "cuda" if use_gpu else "cpu",
            "--verify-rows", str(cfg["verify_rows"]),
            "--report", str(report.resolve()),
        ]  # fmt: skip
        _run_dfa(ctx, str(GPU_EMBED_RUNNER), args, ctx.run_dir / "embed.log")
        write_json(
            ctx.paths.reports_dir / "embed_report.json", json.loads(report.read_text())
        )
    elif cfg["runner"] == "df-embed":
        args = ["--modality", "nlp", *io_args]
        _run_dfa(ctx, "df-embed.py", args, ctx.run_dir / "df_embed.log")
    else:
        raise StageError(
            f"Unknown embed.runner {cfg['runner']!r}; use 'gpu' or 'df-embed'"
        )
    embedded = pd.read_parquet(output_path)
    table = text.attach_ids_to_embeddings(embedded, df[C.COMPLAINT_ID], df[C.LABEL])
    table.to_parquet(ctx.paths.embeddings, index=False)
    print(f"Saved {len(table):,} embeddings of width {table.shape[1] - 1}")


# --------------------------------------------------------------------------
# Stage 6: PCA on text embeddings (fit on training complaints only)
# --------------------------------------------------------------------------
def run_pca(ctx: Context) -> None:
    _require(ctx.paths.embeddings, ctx.paths.train_ids)
    embeddings = pd.read_parquet(ctx.paths.embeddings)
    comps, variance = text.fit_pca(
        embeddings,
        _read_ids(ctx.paths.train_ids),
        int(ctx.config["pca"]["n_components"]),
        ctx.seed,
    )
    comps.to_parquet(ctx.paths.text_pca, index=False)
    variance.to_csv(ctx.paths.reports_dir / "pca_explained_variance.csv", index=False)
    explained = variance["cumulative"].iloc[-1]
    print(f"{comps.shape[1] - 1} components explain {explained:.3f} of the variance")


# --------------------------------------------------------------------------
# Stage 7: tabular features (company statistics from training complaints only)
# --------------------------------------------------------------------------
def run_features(ctx: Context) -> None:
    _require(ctx.paths.sample, ctx.paths.train_ids)
    df = pd.read_parquet(ctx.paths.sample)
    features, groups, merged = tabular.build_tabular_features(
        df,
        _read_ids(ctx.paths.train_ids),
        min_level_count=int(ctx.config["features"]["min_level_count"]),
    )
    write_json(ctx.paths.reports_dir / "rare_levels_merged.json", merged)
    for col, levels in merged.items():
        print(
            f"{col}: merged {len(levels)} rare or unseen levels into {tabular.RARE_LEVEL}"
        )
    features.to_parquet(ctx.paths.tabular_features, index=False)
    write_json(ctx.paths.tabular_feature_groups, groups)
    print(f"Built {features.shape[1] - 1} tabular features")


# --------------------------------------------------------------------------
# Stage 8: df-analyze input tables
# --------------------------------------------------------------------------
def run_df_analyze_input(ctx: Context) -> None:
    p = ctx.paths
    _require(
        p.tabular_features,
        p.tabular_feature_groups,
        p.text_pca,
        p.sample,
        p.train_ids,
        p.test_ids,
    )
    features = pd.read_parquet(p.tabular_features)
    comps = pd.read_parquet(p.text_pca)
    labels = pd.read_parquet(p.sample, columns=[C.COMPLAINT_ID, C.LABEL])
    train, test = df_analyze.build_tables(
        features, comps, labels, _read_ids(p.train_ids), _read_ids(p.test_ids)
    )
    p.df_analyze_input_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(p.df_analyze_input_dir / "train.parquet", index=False)
    test.to_parquet(p.df_analyze_input_dir / "test.parquet", index=False)
    groups = json.loads(p.tabular_feature_groups.read_text(encoding="utf-8"))
    groups["text"] = [c for c in comps.columns if c != C.COMPLAINT_ID]
    write_json(p.feature_groups, groups)
    print(f"df-analyze tables: train {train.shape}, test {test.shape}")


# --------------------------------------------------------------------------
# Stage 9: run df-analyze and verify its exported split
# --------------------------------------------------------------------------
def run_df_analyze(ctx: Context) -> None:
    p = ctx.paths
    train_path = p.df_analyze_input_dir / "train.parquet"
    test_path = p.df_analyze_input_dir / "test.parquet"
    _require(train_path, test_path, p.text_pca)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    outdir = p.df_analyze_output_dir / stamp
    outdir.mkdir(parents=True, exist_ok=False)
    gpu_models = GPU_CLASSIFIERS.intersection(ctx.config["df_analyze"]["classifiers"])
    if gpu_models:
        # CatBoost and GANDALF switch to the GPU when df-analyze sees CUDA.
        check_cuda(
            ctx, "df_analyze", "df-analyze (" + ", ".join(sorted(gpu_models)) + ")"
        )
    args = df_analyze.df_analyze_args(
        ctx.config["df_analyze"], train_path, test_path, outdir, ctx.seed
    )
    _run_dfa(ctx, "df-analyze.py", args, outdir / "df_analyze.log")
    verify_df_analyze_split(ctx, outdir)
    (p.df_analyze_output_dir / "latest.txt").write_text(stamp + "\n", encoding="utf-8")


def verify_df_analyze_split(ctx: Context, outdir: Path) -> None:
    p = ctx.paths
    train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    test = pd.read_parquet(p.df_analyze_input_dir / "test.parquet")
    pca_cols = [c for c in train.columns if c.startswith(text.PCA_PREFIX)]
    export = df_analyze.read_export(df_analyze.find_export_dir(outdir))
    try:
        report = df_analyze.verify_export(export, train, test, pca_cols)
    except df_analyze.SplitVerificationError as e:
        raise StageError(str(e)) from e
    write_json(outdir / "split_check.json", report)
    write_json(p.reports_dir / "df_analyze_split_check.json", report)
    print("Verified: df-analyze's exported train/test rows match our saved split.")


STAGES: list[Stage] = [
    Stage("load", "Read archives 2-4, keep narratives, report categories", run_load,
          lambda p: [p.complaints, p.reports_dir / "category_values.csv"]),
    Stage("label", "Apply the reviewed label allow-list; drop Issue/Sub-issue",
          run_label, lambda p: [p.labeled]),
    Stage("sample", "Remove near-duplicate narratives; stratified sample", run_sample,
          lambda p: [p.sample]),
    Stage("split", "Stratified train/test split; save Complaint IDs", run_split,
          lambda p: [p.split, p.train_ids, p.test_ids]),
    Stage("embed", "Embed narratives with df-embed's code (GPU by default)", run_embed,
          lambda p: [p.embeddings]),
    Stage("pca", "PCA on embeddings, fit on training complaints", run_pca,
          lambda p: [p.text_pca]),
    Stage("features", "Tabular and company features (training stats only)",
          run_features, lambda p: [p.tabular_features, p.tabular_feature_groups]),
    Stage("df_analyze_input", "Write df-analyze train/test tables",
          run_df_analyze_input,
          lambda p: [p.df_analyze_input_dir / "train.parquet",
                     p.df_analyze_input_dir / "test.parquet", p.feature_groups]),
    Stage("df_analyze", "Run df-analyze (Model A) and verify its split", run_df_analyze,
          lambda p: [p.reports_dir / "df_analyze_split_check.json"]),
]  # fmt: skip

STAGE_NAMES = [s.name for s in STAGES]


def get_stage(name: str) -> Stage:
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise KeyError(f"Unknown stage {name!r}; choose from {STAGE_NAMES}")
