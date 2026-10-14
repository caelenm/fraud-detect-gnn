"""Reading df-analyze's saved models and tables; saving and loading Model A.

Shared by export_model.py and predict.py. LIKE THEM, THIS RUNS INSIDE
DF-ANALYZE'S ENVIRONMENT, never this project's. Call `setup_paths()` before
importing anything from df-analyze or fraud_detect, and before decoding
df-analyze's results (they refer to df-analyze's classes).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import joblib
import jsonpickle
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_dfa_root() -> Path:
    """The df-analyze clone: $DF_ANALYZE_DIR, else the current directory if it
    is one (the pipeline runs these scripts from there), else ../df-analyze
    next to this repository."""
    env = os.environ.get("DF_ANALYZE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "df_analyze").is_dir():
        return cwd
    return (REPO_ROOT.parent / "df-analyze").resolve()


def setup_paths() -> Path:
    """Make df-analyze (from its source tree, as cv_select.py does) and this
    repository's src/ importable. Returns the df-analyze clone."""
    dfa = find_dfa_root()
    if not (dfa / "src" / "df_analyze").is_dir():
        raise SystemExit(
            f"df-analyze not found at {dfa}. Set DF_ANALYZE_DIR to your df-analyze clone."
        )
    for i, path in enumerate((dfa / "src", dfa, REPO_ROOT / "src")):
        if str(path) not in sys.path:
            sys.path.insert(i, str(path))
    return dfa


def read_table(path: Path) -> pd.DataFrame:
    """A feature table from .csv (df-analyze's exports) or .parquet. CSV floats
    are parsed exactly (round_trip), so a model sees the values df-analyze
    wrote, not ones that may differ in the last bit."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, float_precision="round_trip")


def read_target(path: Path) -> pd.Series:
    """df-analyze's y_*_00.csv export (index, then one target column)."""
    frame = pd.read_csv(path, index_col=0)
    return frame.iloc[:, 0].astype(int).reset_index(drop=True).rename("target")


def decode_results(path: Path) -> list[Any]:
    """Every tuned combination (df-analyze HtuneResult objects, each holding
    its model refit on the full training set) from
    eval_htune_results_jsonpickle_00.json, in the order df-analyze saved them
    (the same order as prediction_results_00.json)."""
    decoded = jsonpickle.decode(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict) or "results" not in decoded:
        raise SystemExit(f"{path} does not hold df-analyze evaluation results")
    return list(decoded["results"])


def save_native(estimator: Any, fmt: str, path: Path) -> None:
    """Save a fitted estimator in its library's own format."""
    if fmt == "joblib":
        joblib.dump(estimator, path)
    elif fmt == "cbm":
        estimator.save_model(str(path), format="cbm")
    else:
        raise ValueError(f"Unknown model format {fmt!r}")


def load_native(fmt: str, path: Path) -> Any:
    """Load an estimator saved by save_native; it has predict_proba(X)."""
    if fmt == "joblib":
        return joblib.load(path)
    if fmt == "cbm":
        from catboost import CatBoostClassifier  # noqa: PLC0415  (only if needed)

        model = CatBoostClassifier()
        model.load_model(str(path), format="cbm")
        return model
    raise ValueError(f"Unknown model format {fmt!r}")
