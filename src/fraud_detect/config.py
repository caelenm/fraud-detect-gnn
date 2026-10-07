"""Configuration loading and project paths."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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
    feature_set = out.get("feature_set")
    if feature_set not in out["feature_sets"]:
        raise ConfigError(
            f"Unknown feature set {feature_set!r}; choose from "
            + ", ".join(out["feature_sets"])
        )
    return out


def dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    """The active dataset's section of the config (see `with_dataset`)."""
    return config["datasets"][config["dataset"]]


@dataclass(frozen=True)
class Paths:
    """All file locations used by the pipeline for one dataset and feature set.

    data/raw/care_gnn/               downloaded .zip and .mat files (shared)
    data/processed/<dataset>/        node table, graph, split, feature blocks
    outputs/<dataset>/               dataset-level reports, run state, run logs
    outputs/<dataset>/<feature_set>/ df-analyze runs, model reports, web report
    """

    root: Path
    raw_dir: Path
    mat: Path  # the active dataset's .mat file in raw_dir
    processed_dir: Path
    outputs_dir: Path
    feature_set: str

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

    @property
    def column_spec(self) -> Path:
        return self.processed_dir / "column_spec.json"

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


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def get_paths(config: dict[str, Any], root: Path = REPO_ROOT) -> Paths:
    """Paths for the active dataset and feature set (see `with_dataset`)."""
    p = config["paths"]
    dataset = config.get("dataset")
    if not dataset:
        raise ConfigError("No dataset selected (use --dataset or `dataset:`)")
    return Paths(
        root=root,
        raw_dir=_resolve(root, p["raw_dir"]),
        mat=_resolve(root, p["raw_dir"]) / config["datasets"][dataset]["mat"],
        processed_dir=_resolve(root, p["processed_dir"]) / dataset,
        outputs_dir=_resolve(root, p["outputs_dir"]) / dataset,
        feature_set=str(config["feature_set"]),
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
