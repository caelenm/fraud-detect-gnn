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


@dataclass(frozen=True)
class Paths:
    """All file locations used by the pipeline, derived from the config."""

    root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    outputs_dir: Path
    categories_file: Path

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    # Stage outputs. Keeping them here makes each stage's inputs explicit.
    @property
    def complaints(self) -> Path:
        return self.interim_dir / "complaints.parquet"

    @property
    def labeled(self) -> Path:
        return self.interim_dir / "labeled.parquet"

    @property
    def sample(self) -> Path:
        return self.interim_dir / "sample.parquet"

    @property
    def split(self) -> Path:
        return self.processed_dir / "split.csv"

    @property
    def train_ids(self) -> Path:
        return self.processed_dir / "train_ids.csv"

    @property
    def test_ids(self) -> Path:
        return self.processed_dir / "test_ids.csv"

    @property
    def embed_dir(self) -> Path:
        return self.interim_dir / "embed"

    @property
    def embeddings(self) -> Path:
        return self.processed_dir / "embeddings.parquet"

    @property
    def text_pca(self) -> Path:
        return self.processed_dir / "text_pca.parquet"

    @property
    def tabular_features(self) -> Path:
        return self.processed_dir / "tabular_features.parquet"

    @property
    def tabular_feature_groups(self) -> Path:
        return self.processed_dir / "tabular_feature_groups.json"

    @property
    def feature_groups(self) -> Path:
        return self.processed_dir / "feature_groups.json"

    @property
    def df_analyze_input_dir(self) -> Path:
        return self.processed_dir / "df_analyze"

    @property
    def df_analyze_output_dir(self) -> Path:
        return self.outputs_dir / "df_analyze"

    @property
    def runs_dir(self) -> Path:
        return self.outputs_dir / "runs"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def get_paths(config: dict[str, Any], root: Path = REPO_ROOT) -> Paths:
    p = config["paths"]
    return Paths(
        root=root,
        raw_dir=_resolve(root, p["raw_dir"]),
        interim_dir=_resolve(root, p["interim_dir"]),
        processed_dir=_resolve(root, p["processed_dir"]),
        outputs_dir=_resolve(root, p["outputs_dir"]),
        categories_file=_resolve(root, p["categories_file"]),
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
