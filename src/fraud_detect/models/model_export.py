"""Exported df-analyze models: checks, metadata and input columns.

df-analyze saves every tuned (model, feature selection) combination, refit on
the full training set, in `eval_htune_results_jsonpickle_00.json`. The
`export_model` stage (scripts/dfa/export_model.py) decodes that file, checks
that each model reproduces df-analyze's saved test probabilities, and saves
Model A in its library's own format next to `model_a.json`, which records
everything needed to use it. scripts/dfa/predict.py loads the result.

Those two scripts run in df-analyze's environment and import this module from
src/, so it uses only the standard library, numpy and pandas, which both
environments have. Everything here is plain logic, tested in the project
environment.
"""

from __future__ import annotations

import json
import math
import platform
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1
MODEL_A_META = "model_a.json"
MODELS_CHECK = "models_check.csv"

# df-analyze estimator classes -> the classifier names used in the config.
MODEL_NAMES = {
    "CatBoostClassifier": "catboost",
    "DummyClassifier": "dummy",
    "GandalfEstimator": "gandalf",
    "KNNClassifier": "knn",
    "LRClassifier": "lr",
    "LightGBMClassifier": "lgbm",
    "LightGBMRFClassifier": "rf",
    "MLPEstimator": "mlp",
}

# How Model A is saved, per config model name: scikit-learn-style estimators
# (LightGBM's included) with joblib, CatBoost in its own .cbm format.
NATIVE_FORMATS = {
    "dummy": "joblib",
    "knn": "joblib",
    "lgbm": "joblib",
    "lr": "joblib",
    "rf": "joblib",
    "catboost": "cbm",
}
MODEL_FILES = {"joblib": "model_a.joblib", "cbm": "model_a.cbm"}

# Largest allowed |difference| between a reloaded model's test probabilities
# and those df-analyze saved. CPU models must reproduce them to rounding
# error. CatBoost and GANDALF train on the GPU, whose floating-point sums may
# differ slightly from a CPU re-evaluation.
CPU_TOLERANCE = 1e-9
GPU_TOLERANCE = 1e-5
GPU_MODELS = frozenset({"catboost", "gandalf"})

# Packages whose versions decide whether a saved model loads and predicts the
# same way; recorded in model_a.json and compared by predict.py.
PACKAGES = ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "torch",
            "jsonpickle", "joblib")  # fmt: skip

# Keys model_a.json must have (see export_model.py for their meaning).
META_KEYS = (
    "format_version", "dataset", "feature_set", "run", "results_dir", "model",
    "model_cls", "selection", "embed_selector", "input_columns",
    "prepared_columns", "tuned_params", "threshold", "threshold_rule", "seed",
    "df_analyze_commit", "source", "model_file", "model_format", "verification",
    "package_versions",
)  # fmt: skip


class ColumnMismatchError(ValueError):
    """An input table's columns are not the ones the model was trained on."""


class MetadataError(ValueError):
    """model_a.json is missing, incomplete or of an unknown format."""


def model_name(model_cls: str) -> str:
    """Config name (lgbm, catboost, ...) of a df-analyze estimator class."""
    return MODEL_NAMES.get(model_cls, model_cls)


def selection_label(selection: str, embed_selector: str | None) -> str:
    """Feature selection as df-analyze names it in its tables: `embed_linear`
    for embedded selection with a linear model, otherwise the selection."""
    if selection == "embed" and embed_selector:
        return f"embed_{embed_selector}"
    return str(selection)


def tolerance_for(model: str) -> float:
    return GPU_TOLERANCE if model in GPU_MODELS else CPU_TOLERANCE


def check_columns(
    found: Sequence[str], expected: Sequence[str], prepared: Sequence[str] = ()
) -> list[str]:
    """The columns to give the model, after checking an input table's columns.

    The table must have exactly the model's input columns in the same order
    (`expected`), or exactly every column of df-analyze's prepared feature
    table in the same order (`prepared`, e.g. its X_test export), from which
    the model's columns are then taken. Anything else (a missing, extra,
    renamed or reordered column) raises ColumnMismatchError, because a model
    given columns in the wrong order silently predicts nonsense."""
    found, expected, prepared = list(found), list(expected), list(prepared)
    if found == expected or (prepared and found == prepared):
        return expected
    # Explain the mismatch against the prepared table if the input has more
    # than the model's columns, else against the model's columns.
    reference = prepared if prepared and set(found) - set(expected) else expected
    missing = [c for c in reference if c not in found]
    extra = [c for c in found if c not in reference]
    lines = []
    if missing:
        lines.append(f"missing columns: {missing}")
    if extra:
        lines.append(f"unexpected columns: {extra}")
    if not missing and not extra:
        # zip stops at the shorter list: a repeated column shows up there.
        pairs = enumerate(zip(found, reference, strict=False))
        first = next(
            (i for i, (a, b) in pairs if a != b), min(len(found), len(reference))
        )
        lines.append(
            f"the right columns, but reordered or repeated (first difference at "
            f"position {first})"
        )
    raise ColumnMismatchError(
        "The input table's columns do not match the model's. It must have exactly "
        f"the model's {len(expected)} input columns in order, or all "
        f"{len(prepared)} columns of df-analyze's prepared table (X_test_00.csv) "
        "in order.\n- " + "\n- ".join(lines)
    )


def is_fitted_model(obj: Any) -> bool:
    """True if a decoded df-analyze model is a real object holding a fit.

    When jsonpickle cannot encode an object, df-analyze saves `str(obj)`
    instead (`fail_safe=str` in EvaluationResults.save), so a model can come
    back as a placeholder string, or as an object whose fitted estimator
    (`tuned_model`) is one."""
    if obj is None or isinstance(obj, str | bytes):
        return False
    if not callable(getattr(obj, "predict_proba", None)):
        return False
    tuned = getattr(obj, "tuned_model", obj)
    return tuned is not None and not isinstance(tuned, str | bytes)


def max_abs_diff(a: Any, b: Any) -> float:
    """Largest absolute difference between two probability arrays; infinite
    if their shapes differ or either has a NaN."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or np.isnan(a).any() or np.isnan(b).any():
        return math.inf
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def check_combination(
    model_obj: Any,
    predict: Callable[[Any], Any],
    saved_probs: Any,
    tolerance: float,
) -> dict[str, Any]:
    """Whether one decoded combination is a fitted model (`decoded`) and
    reproduces df-analyze's saved test probabilities (`ok`). `predict(model)`
    returns its test probabilities (n_rows x 2)."""
    if not is_fitted_model(model_obj):
        shown = repr(model_obj)[:80]
        return {"decoded": False, "max_abs_diff": math.nan, "ok": False,
                "error": f"not a fitted model: {shown}"}  # fmt: skip
    try:
        probs = predict(model_obj)
    except Exception as e:  # record the failure; the caller decides what to do
        return {"decoded": True, "max_abs_diff": math.nan, "ok": False,
                "error": f"{type(e).__name__}: {e}"[:300]}  # fmt: skip
    diff = max_abs_diff(probs, saved_probs)
    error = "" if diff <= tolerance else f"probabilities differ by up to {diff:.3g}"
    return {"decoded": True, "max_abs_diff": diff, "ok": diff <= tolerance,
            "error": error}  # fmt: skip


def model_a_source(check: dict[str, Any]) -> str:
    """'decoded' when Model A's saved model reproduces its test probabilities,
    else 'refit' (refit with its tuned parameters on the full training set)."""
    return "decoded" if check["decoded"] and check["ok"] else "refit"


def native_format(model: str) -> str:
    """How Model A is saved for config model `model`."""
    if model not in NATIVE_FORMATS:
        raise ValueError(
            f"Saving a {model!r} model in its native format is not implemented "
            f"(supported: {', '.join(sorted(NATIVE_FORMATS))})."
        )
    return NATIVE_FORMATS[model]


def package_versions(names: Sequence[str] = PACKAGES) -> dict[str, str | None]:
    """Installed versions of `names` (None if not installed), plus Python."""
    found: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def version_warnings(
    recorded: dict[str, str | None], installed: dict[str, str | None]
) -> list[str]:
    """One warning per package whose installed version differs from the one
    the model was saved with."""
    return [
        f"{name} {installed.get(name)} is installed, but the model was saved with "
        f"{version}; predictions may differ or loading may fail."
        for name, version in recorded.items()
        if installed.get(name) != version
    ]


def write_metadata(path: Path, meta: dict[str, Any]) -> None:
    """Write model_a.json (directly; the caller makes the write atomic)."""
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise MetadataError(f"model_a.json would lack {missing}")
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MetadataError(
            f"{path} does not exist. Run the export_model stage, or unpack a bundle "
            "that contains it."
        )
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("format_version") != FORMAT_VERSION:
        raise MetadataError(
            f"{path} has format {meta.get('format_version')}, expected {FORMAT_VERSION}"
        )
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise MetadataError(f"{path} lacks {missing}")
    return meta
