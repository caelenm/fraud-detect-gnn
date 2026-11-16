"""Configuration loading and project paths."""

from __future__ import annotations

import copy
import os
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fraud_detect.columns import BLOCK_KINDS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    """The configuration is incomplete or inconsistent; the message says why."""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the pipeline config. Relative paths inside it are resolved against the
    repository root, not the current working directory."""
    return load_yaml(path)


def apply_overrides(config: dict[str, Any], pairs: list[str]) -> dict[str, Any]:
    """Apply `section.key=value` overrides (values parsed as YAML) to a copy of
    `config`. Keys must already exist, so a typo cannot silently add a setting."""
    out = copy.deepcopy(config)
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"Override {pair!r} must look like section.key=value")
        *parents, leaf = key.split(".")
        node = out
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise ValueError(f"Unknown config section {part!r} in {pair!r}")
            node = node[part]
        if leaf not in node:
            raise ValueError(f"Unknown config key {key!r}")
        node[leaf] = yaml.safe_load(raw)
    return out


def dataset_names(config: dict[str, Any]) -> list[str]:
    return list(config["datasets"])


def with_dataset(config: dict[str, Any], dataset: str | None) -> dict[str, Any]:
    """A copy of `config` with the active dataset set (from --dataset, else the
    config's `dataset:` key), checked against the configured datasets."""
    out = copy.deepcopy(config)
    name = dataset or out.get("dataset")
    if not name:
        raise ConfigError(
            "Choose a dataset with --dataset (one of: "
            + ", ".join(dataset_names(out))
            + ") or set `dataset:` in the config."
        )
    if name not in out["datasets"]:
        raise ConfigError(
            f"Unknown dataset {name!r}; choose from {', '.join(dataset_names(out))}"
        )
    out["dataset"] = name
    repeat, n_repeats = int(out["split"]["repeat"]), int(out["split"]["n_repeats"])
    if not 0 <= repeat < max(n_repeats, 1):
        raise ConfigError(f"split.repeat {repeat} is outside 0..{n_repeats - 1}")
    feature_set = out.get("feature_set")
    if feature_set not in out["feature_sets"]:
        raise ConfigError(
            f"Unknown feature set {feature_set!r}; choose from "
            + ", ".join(out["feature_sets"])
        )
    needed = {b.partition("_")[2] for b in out["feature_sets"][feature_set]} - {""}
    out["feature_variants"] = resolve_variants(
        out.get("feature_variants") or {}, repeat, needed
    )
    check_feature_sets(out)
    return out


KEEP_BY_REPEAT = "keep_by_repeat"
# gnn_models graphs: every relation, N2's rewired graph, or no edges (an MLP).
GNN_GRAPHS = ("real", "rewired", "none")
GNN_RUNG_KEYS = {"graph", "norm", "setting_from", "train"}
# What a rung's own `train:` block may set on top of gnn_classify (the larger
# rungs, docs/BIGGER_GNN_PLAN.md); graph/supervised.TrainConfig checks values.
GNN_TRAIN_KEYS = {
    "layers", "hidden", "dropout", "lr", "weight_decay", "max_epochs", "patience",
    "model", "residual", "layer_norm", "jk", "hops", "heads",
}  # fmt: skip
# gnn_embed_sizes value for a GNN block that keeps its raw embedding (no PCA).
RAW_SIZE = "raw"


def resolve_variants(
    variants: dict[str, Any], repeat: int, needed: Collection[str] = ()
) -> dict[str, Any]:
    """A {keep_by_repeat: {0: [...], 1: [...]}} variant keeps other own columns
    on each split repeat (F5: a top 4 ranked on each split's own training
    folds). It becomes {keep: [...]} for the active repeat; others are kept.
    One that lists no columns for the active repeat is an error only if the
    active feature set uses it (`needed`); otherwise it stays unresolved, so a
    block built from it fails loudly (blocks.variant_spec)."""
    out: dict[str, Any] = {}
    for name, spec in variants.items():
        if isinstance(spec, dict) and set(spec) == {KEEP_BY_REPEAT}:
            by_repeat = {int(k): v for k, v in (spec[KEEP_BY_REPEAT] or {}).items()}
            if repeat in by_repeat:
                spec = {"keep": list(by_repeat[repeat])}
            elif name in needed:
                raise ConfigError(
                    f"feature_variants.{name} lists no columns for split repeat {repeat}"
                )
        out[name] = spec
    return out


def check_feature_sets(config: dict[str, Any]) -> None:
    """Every block of every feature set is a known kind, optionally with a
    variant listed in feature_variants (e.g. own_no36, gnnssl_no36)."""
    variants = config.get("feature_variants") or {}
    for variant, spec in variants.items():
        keeps = isinstance(spec, dict) and set(spec) == {"keep"} and spec["keep"]
        # Left unresolved by resolve_variants: no columns for this split repeat.
        unresolved = isinstance(spec, dict) and set(spec) == {KEEP_BY_REPEAT}
        if not (keeps or unresolved or isinstance(spec, list)) or "_" in variant:
            raise ConfigError(
                f"feature_variants.{variant}: a variant (a name without '_') is a list "
                "of own columns to leave out, or {keep: [own columns to keep]}"
            )
    for variant, size in (config.get("gnn_embed_sizes") or {}).items():
        number = isinstance(size, int) and not isinstance(size, bool) and size > 0
        if variant not in variants or not (number or size == RAW_SIZE):
            raise ConfigError(
                f"gnn_embed_sizes.{variant}: maps a variant of feature_variants to a "
                f"positive number of PCA components per relation, or {RAW_SIZE!r}"
            )
    for name, spec in (config.get("gnn_models") or {}).items():
        own_only = [b.partition("_")[0] for b in config["feature_sets"].get(name, [])]
        if own_only != ["own"]:
            raise ConfigError(
                f"gnn_models.{name}: a GNN classifier rung needs a feature set of "
                "exactly one own block"
            )
        if (
            not isinstance(spec, dict)
            or spec.get("graph") not in GNN_GRAPHS
            or (spec.get("norm", "mean") not in ("mean", "sym"))
        ):
            raise ConfigError(
                f"gnn_models.{name}: graph must be one of {GNN_GRAPHS} and norm "
                "mean or sym"
            )
        unknown = sorted(set(spec) - GNN_RUNG_KEYS)
        if unknown:
            raise ConfigError(f"gnn_models.{name}: unknown keys {unknown}")
        train = spec.get("train")
        if train is not None and (
            not isinstance(train, dict) or not set(train) <= GNN_TRAIN_KEYS
        ):
            raise ConfigError(
                f"gnn_models.{name}.train: a mapping of gnn_classify settings, from "
                f"{sorted(GNN_TRAIN_KEYS)}"
            )
        source = spec.get("setting_from")
        if source is not None and source not in config["gnn_models"]:
            raise ConfigError(f"gnn_models.{name}: setting_from {source!r} is not a rung")
        if source is not None and (spec.get("train") or {}) != (
            config["gnn_models"][source].get("train") or {}
        ):
            raise ConfigError(
                f"gnn_models.{name}: setting_from {source!r} needs the same train: "
                "block (a setting is only meaningful for the same network)"
            )
    for name, block_names in config["feature_sets"].items():
        for block in block_names:
            kind, _, variant = block.partition("_")
            if kind not in BLOCK_KINDS:
                raise ConfigError(f"Feature set {name}: unknown block {block!r}")
            if variant and variant not in variants:
                raise ConfigError(
                    f"Feature set {name}: block {block!r} needs feature_variants."
                    f"{variant}"
                )


def dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    """The active dataset's section of the config (see `with_dataset`)."""
    return config["datasets"][config["dataset"]]


@dataclass(frozen=True)
class Paths:
    """All file locations used by the pipeline for one dataset and feature set.

    data/raw/care_gnn/               downloaded .zip and .mat files (shared)
    data/processed/<dataset>/        node table, graph, split, feature blocks
    outputs/<dataset>/               dataset-level reports, run state, run logs,
                                     GNN embeddings (gnn_embed/), comparison
                                     pages (compare/)
    outputs/<dataset>/<feature_set>/ df-analyze runs, model reports, web report,
                                     exported Model A (model/)
    """

    root: Path
    raw_dir: Path
    mat: Path  # the active dataset's .mat file in raw_dir
    processed_dir: Path
    outputs_dir: Path
    feature_set: str
    # The active feature set's blocks, so a block stage's outputs can be only
    # the blocks this feature set uses (none for m1_own).
    blocks: tuple[str, ...] = ()
    # True for a GNN classifier rung (gnn_models): gnn_classify scores it, and
    # the df-analyze stages have no outputs for it.
    is_gnn: bool = False

    # ---- Dataset level -----------------------------------------------------
    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def runs_dir(self) -> Path:
        return self.outputs_dir / "runs"

    @property
    def nodes(self) -> Path:
        return self.processed_dir / "nodes.parquet"

    @property
    def graph_dir(self) -> Path:
        return self.processed_dir / "graph"

    @property
    def graph_manifest(self) -> Path:
        return self.graph_dir / "manifest.json"

    def relation_file(self, relation: str) -> Path:
        return self.graph_dir / f"{relation}.npz"

    def rewired_dir(self, seed: int) -> Path:
        """Every relation rewired so each node keeps exactly its degree (the
        null graph of control N2), for one rewiring seed."""
        return self.processed_dir / "graph_rewired" / f"seed{seed}"

    @property
    def train_ids(self) -> Path:
        return self.processed_dir / "train_ids.csv"

    @property
    def test_ids(self) -> Path:
        return self.processed_dir / "test_ids.csv"

    @property
    def cv_folds(self) -> Path:
        return self.processed_dir / "cv_folds.csv"

    @property
    def split_summary(self) -> Path:
        return self.processed_dir / "split_summary.json"

    @property
    def features_dir(self) -> Path:
        return self.processed_dir / "features"

    def block_file(self, block: str) -> Path:
        return self.features_dir / f"{block}.parquet"

    def block_spec(self, block: str) -> Path:
        """Column types of a block other than `own` (whose types are in
        column_spec.json), decided on training nodes like column_spec.json."""
        return self.features_dir / f"{block}.spec.json"

    @property
    def column_spec(self) -> Path:
        return self.processed_dir / "column_spec.json"

    @property
    def gnn_embed_dir(self) -> Path:
        """Raw self-supervised embeddings and their training log (gnn_embed)."""
        return self.embed_dir("gnnssl")

    def embed_dir(self, block: str) -> Path:
        """gnn_embed's folder for one block: gnn_embed/ for `gnnssl`,
        gnn_embed_<variant>/ for `gnnssl_<variant>`, gnn_embed_<block>/ for the
        other kinds (gnn_embed_gnnrand/, gnn_embed_gnnrand_no36/)."""
        if block.startswith("gnnssl"):
            return self.outputs_dir / ("gnn_embed" + block.removeprefix("gnnssl"))
        return self.outputs_dir / f"gnn_embed_{block}"

    def raw_embedding(self, relation: str, block: str = "gnnssl") -> Path:
        """One relation's raw (n_nodes, out_dim) GNN embedding, before PCA."""
        return self.embed_dir(block) / f"raw_{relation}.npy"

    def nested_scores(self, block: str) -> Path:
        """A gnnsup block's nested out-of-fold scores (node_id, outer_fold,
        <block>__score), which select_model puts on the rows it fits on."""
        return self.features_dir / f"{block}.nested.parquet"

    @property
    def gnn_oof_dir(self) -> Path:
        """The gnn_oof stage's meta.json (which GNN, its setting and seeds)."""
        return self.outputs_dir / "gnn_oof"

    @property
    def probe_dir(self) -> Path:
        return self.outputs_dir / "probe"

    def compare_dir(self, a: str, b: str) -> Path:
        """Comparison page of feature sets `a` and `b` (scripts/compare_report.py)."""
        return self.outputs_dir / "compare" / f"{a}__vs__{b}"

    # ---- Feature-set level (one df-analyze run per feature set) -------------
    @property
    def df_analyze_input_dir(self) -> Path:
        return self.processed_dir / "df_analyze" / self.feature_set

    @property
    def feature_set_dir(self) -> Path:
        return self.outputs_dir / self.feature_set

    @property
    def model_reports_dir(self) -> Path:
        return self.feature_set_dir / "reports"

    @property
    def df_analyze_output_dir(self) -> Path:
        return self.feature_set_dir / "df_analyze"

    @property
    def web_report(self) -> Path:
        return self.feature_set_dir / "report" / "index.html"

    @property
    def model_dir(self) -> Path:
        """Model A saved in its native format, with model_a.json."""
        return self.feature_set_dir / "model"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_folder(dataset: str, repeat: int = 0) -> str:
    """Folder name of one dataset and split repeat (repeat 0: the dataset)."""
    return dataset if repeat == 0 else f"{dataset}_repeat{repeat}"


def get_paths(config: dict[str, Any], root: Path = REPO_ROOT) -> Paths:
    """Paths for the active dataset, split repeat and feature set."""
    p = config["paths"]
    dataset = config.get("dataset")
    if not dataset:
        raise ConfigError("No dataset selected (use --dataset or `dataset:`)")
    folder = run_folder(dataset, int(config["split"].get("repeat", 0)))
    feature_set = str(config["feature_set"])
    return Paths(
        root=root,
        raw_dir=_resolve(root, p["raw_dir"]),
        mat=_resolve(root, p["raw_dir"]) / config["datasets"][dataset]["mat"],
        processed_dir=_resolve(root, p["processed_dir"]) / folder,
        outputs_dir=_resolve(root, p["outputs_dir"]) / folder,
        feature_set=feature_set,
        blocks=tuple(config["feature_sets"].get(feature_set, ())),
        is_gnn=feature_set in (config.get("gnn_models") or {}),
    )


def df_analyze_dir(config: dict[str, Any], root: Path = REPO_ROOT) -> Path:
    """Location of the separate df-analyze clone.

    Precedence: config `df_analyze.dir`, then the DF_ANALYZE_DIR environment
    variable, then `../df-analyze` next to this repository.
    """
    configured = config.get("df_analyze", {}).get("dir")
    if configured:
        return _resolve(root, configured).resolve()
    env = os.environ.get("DF_ANALYZE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (root.parent / "df-analyze").resolve()
