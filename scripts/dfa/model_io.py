"""Reading df-analyze's saved models and tables; saving and loading Model A.

Shared by export_model.py and predict.py. LIKE THEM, THIS RUNS INSIDE
DF-ANALYZE'S ENVIRONMENT, never this project's. Call `setup_paths()` before
importing anything from df-analyze or fraud_detect, and before decoding
df-analyze's results (they refer to df-analyze's classes).
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from jsonpickle import tags
from jsonpickle.unpickler import Unpickler, _Proxy

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


def _load_tensors_on_cpu_without_cuda() -> None:
    """jsonpickle stores torch tensors (GANDALF's weights) as torch.save bytes,
    restored with torch.storage._load_from_bytes. Weights saved from a GPU
    cannot be restored that way on a machine without CUDA (e.g. a Mac), and
    one such tensor stops the whole file from decoding, so map them to the
    CPU there."""
    import torch  # noqa: PLC0415  (heavy; only needed here)

    if torch.cuda.is_available():
        return

    def load_on_cpu(data: bytes) -> Any:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)

    torch.storage._load_from_bytes = load_on_cpu


class Unrestorable:
    """Stands in for a saved object that could not be rebuilt, or a reference
    that cannot be trusted (see TolerantUnpickler)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __repr__(self) -> str:
        return f"<not restorable: {self.reason}>"


class TolerantUnpickler(Unpickler):
    """jsonpickle's Unpickler, made safe for df-analyze's saved results.

    jsonpickle numbers objects in the order it writes them, and a later
    reference to an object (`{"py/id": n}`) is that number. On reading it
    numbers objects as it rebuilds them, so the numbers only agree while
    every object is rebuilt exactly as written. GANDALF's saved training
    state cannot be: some objects cannot be created (torch.dtype) or fail in
    `__setstate__` (the torch optimizer, Lightning internals). By default the
    first failure stops the whole decode, so no model in the file loads; and
    if failures are skipped, the numbering drifts and references in every
    model saved after GANDALF point at the wrong objects, silently.

    Here a failure does not stop the decode (the object becomes an
    `Unrestorable` placeholder), and from the first failure on, the
    numbering is no longer trusted: a reference to any object numbered from
    the start of the saved result in which the first failure happened
    becomes an `Unrestorable` placeholder instead of a possibly wrong object.
    References to objects numbered before that point are exact. A model
    holding a placeholder is reported as not intact; no model is used without
    also reproducing df-analyze's saved test predictions.

    In df-analyze's results the models are saved in alphabetical order, so
    CatBoost and the dummy (before GANDALF) are unaffected, and later models
    refer mostly to objects CatBoost and the dummy saved first (such as numpy
    dtypes). A reference a later model makes to one of its own objects is not
    trusted; _repair_lightgbm restores LightGBM's two known cases. This relies
    on jsonpickle internals, pinned by df-analyze's lock file (4.1.1).
    """

    def __init__(self, result_nodes: list[Any]) -> None:
        super().__init__()
        self._result_starts = {id(node) for node in result_nodes}
        self._current_start = 0  # objects numbered before the current result
        self.trusted_below: int | None = None  # None: no failure yet
        self.failures: list[str] = []
        self._seen_errors: set[int] = set()

    def _failed(self, reason: str) -> None:
        self.failures.append(reason[:200])
        if self.trusted_below is None:
            self.trusted_below = self._current_start

    def _restore(self, obj: Any, *args: Any) -> Any:
        if id(obj) in self._result_starts:
            self._current_start = len(self._objs)
        try:
            return super()._restore(obj, *args)
        except Exception as e:  # recorded once, then handled by a caller
            if id(e) not in self._seen_errors:
                self._seen_errors.add(id(e))
                self._failed(f"{type(e).__name__}: {e}")
            raise

    def _process_missing(self, class_name: str) -> None:
        # jsonpickle keeps such an object as raw JSON and skips its contents.
        self._failed(f"cannot create {class_name}")
        super()._process_missing(class_name)

    def _restore_id(self, obj: Any) -> Any:
        number = obj[tags.ID]
        if self.trusted_below is not None and number >= self.trusted_below:
            return Unrestorable(f"reference {number} after a part that was not rebuilt")
        return super()._restore_id(obj)

    def _restore_reduce(self, obj: Any) -> Any:
        index, depth = len(self._objs), len(self._namestack)
        try:
            return super()._restore_reduce(obj)
        except Exception as e:
            del self._namestack[depth:]  # the failed call may not have popped its names
            self._failed(f"{type(e).__name__}: {e}")
            placeholder = Unrestorable(f"{type(e).__name__}: {e}"[:200])
            proxy = self._objs[index]  # numbered first by the call that failed
            if isinstance(proxy, _Proxy):
                proxy.reset(placeholder)
                self._swapref(proxy, placeholder)
            return placeholder

    def _restore_state(self, obj: Any, instance: Any) -> Any:
        depth = len(self._namestack)
        try:
            return super()._restore_state(obj, instance)
        except Exception as e:
            del self._namestack[depth:]
            self._failed(f"{type(e).__name__}: {e}")
            # The half-restored object is replaced, so the failure stays visible.
            placeholder = Unrestorable(f"{type(e).__name__}: {e}"[:200])
            index = self._obj_to_idx.pop(id(instance), None)
            if index is not None:
                self._objs[index] = placeholder
                self._obj_to_idx[id(placeholder)] = index
            return placeholder


def _repair_lightgbm(result: Any) -> None:
    """LightGBM's scikit-learn classifier holds two objects twice, by
    construction in LGBMModel.fit / LGBMClassifier.fit: `self._classes =
    self._le.classes_` and `self._best_score = self._Booster.best_score`.
    jsonpickle saves the second copy of each as a reference; if that
    reference could not be trusted, take the object from its first copy.
    The export check then verifies the model's probabilities and predicted
    labels against df-analyze's saved ones."""
    estimator = getattr(getattr(result, "model", None), "tuned_model", None)
    pairs = (
        (getattr(estimator, "_le", None), "classes_", estimator, "_classes"),
        (estimator, "_best_score", getattr(estimator, "_Booster", None), "best_score"),
    )
    for holder, name, source, source_name in pairs:
        value = getattr(source, source_name, None)
        if isinstance(getattr(holder, name, None), Unrestorable) and not isinstance(
            value, Unrestorable | type(None)
        ):
            setattr(holder, name, value)


def count_placeholders(obj: Any, _seen: set[int] | None = None, _depth: int = 0) -> int:
    """How many Unrestorable placeholders or unresolved jsonpickle proxies an
    object holds (in its attributes, lists and dicts, 12 levels deep)."""
    seen = set() if _seen is None else _seen
    if id(obj) in seen or _depth > 12:
        return 0
    seen.add(id(obj))
    if isinstance(obj, Unrestorable | _Proxy):
        return 1
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, list | tuple):
        children = list(obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        children = list(vars(obj).values())
    else:
        return 0
    return sum(count_placeholders(c, seen, _depth + 1) for c in children)


def decode_results(path: Path) -> list[Any]:
    """Every tuned combination (df-analyze HtuneResult objects, each holding
    its model refit on the full training set) from
    eval_htune_results_jsonpickle_00.json, in the order df-analyze saved them
    (the same order as prediction_results_00.json). Parts that cannot be
    rebuilt are placeholders (see TolerantUnpickler and count_placeholders)."""
    _load_tensors_on_cpu_without_cuda()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise SystemExit(f"{path} does not hold df-analyze evaluation results")
    unpickler = TolerantUnpickler(data["results"])
    results = list(unpickler.restore(data)["results"])
    for result in results:
        _repair_lightgbm(result)
    if unpickler.failures:
        kinds = sorted({f.split(":")[0] for f in unpickler.failures})
        print(
            f"Note: {len(unpickler.failures)} saved objects could not be rebuilt "
            f"({', '.join(kinds[:4])}); references after the first one are not "
            "trusted.",
            flush=True,
        )
    return results


def without_checkpoint_files(model: Any) -> Any:
    """Make a decoded df-analyze model predict only from what it holds.

    df-analyze's GANDALF predicts from its best checkpoint file when that file
    exists (in the df-analyze clone's __GANDALF_INTERNAL_LOGS__, which no
    bundle contains), and silently from the weights it holds otherwise. This
    forces the second, so a check on this machine gives what anyone else gets.
    Other models are returned unchanged."""
    trainer = getattr(model, "tuned_trainer", None)
    callback = getattr(trainer, "checkpoint_callback", None)
    if callback is not None and getattr(callback, "best_model_path", ""):
        callback.best_model_path = ""
    return model


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
