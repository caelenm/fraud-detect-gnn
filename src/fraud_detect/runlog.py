"""Seeding and run metadata logging."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Packages whose versions are recorded for every run (if installed).
LOGGED_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "datasketch",
    "pyyaml",
    "torch",
    "torch-geometric",
)


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch global RNGs.

    Stages also pass explicit seeds to every randomised function; this is a
    second line of defence for code that uses global RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415  (optional dependency)
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in LOGGED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def new_run_dir(runs_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = runs_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_info(
    run_dir: Path,
    config: dict[str, Any],
    stages: list[str],
    root: Path,
    argv: list[str] | None = None,
) -> None:
    """Record config, package versions, seed and git commit for a run."""
    info = {
        "started_utc": datetime.now(UTC).isoformat(),
        "argv": argv if argv is not None else sys.argv,
        "stages": stages,
        "seed": config.get("seed"),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(root),
        "packages": package_versions(),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
