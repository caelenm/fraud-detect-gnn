"""Check scripts/dfa/model_io.decode_results on a file with a broken object.

RUNS INSIDE DF-ANALYZE'S ENVIRONMENT (tests/test_export_model_dfa.py calls it
with `uv run --directory <df-analyze>`). Not collected by pytest. Exits 0 when
every check passes. Random numbers only; no data.

The file mimics df-analyze's results: a first result holding an object that
cannot be rebuilt (like GANDALF's training state), then a LightGBM model,
whose label encoder and best score jsonpickle saves as references.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dfa"))

import jsonpickle  # noqa: E402
import numpy as np  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from model_io import Unrestorable, count_placeholders, decode_results  # noqa: E402


class FailsOnLoad:
    """Like GANDALF's optimizer: saves fine, fails in __setstate__ on load."""

    def __getstate__(self) -> dict:
        return {"x": 1}

    def __setstate__(self, state: dict) -> None:
        raise TypeError("cannot be rebuilt")


def main() -> int:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = (rng.random(120) < 0.3).astype(int)
    lgbm = LGBMClassifier(n_estimators=5, verbosity=-1).fit(X, y)
    shared = np.arange(4.0)  # first saved in a result before the failing one
    own = np.arange(3.0)  # first saved after the failure
    # As in df-analyze's file, models saved before the failure (CatBoost,
    # dummy) have already saved the numpy dtypes the later models refer to.
    dtypes = [np.zeros(1, dtype=t) for t in (np.float64, np.int64, np.int32)]
    data = {
        "results": [
            SimpleNamespace(early=shared, dtypes=dtypes),
            SimpleNamespace(broken=FailsOnLoad()),
            SimpleNamespace(
                model=SimpleNamespace(tuned_model=lgbm),
                later=shared,
                own=own,
                own_again=own,
            ),
        ],
        "is_classification": True,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eval_htune_results_jsonpickle_00.json"
        path.write_text(jsonpickle.encode(data, unpicklable=True), encoding="utf-8")
        _, failing, second = decode_results(path)

    model = second.model.tuned_model
    same_predictions = np.array_equal(
        model.predict_proba(X), lgbm.predict_proba(X)
    ) and np.array_equal(model.predict(X), lgbm.predict(X))
    checks = {
        "the broken object is a placeholder": isinstance(failing.broken, Unrestorable),
        "a reference to an object saved before the failure resolves": (
            isinstance(second.later, np.ndarray) and np.array_equal(second.later, shared)
        ),
        "a reference to an object saved after the failure is a placeholder": (
            isinstance(second.own_again, Unrestorable)
        ),
        "objects written out in full after the failure are rebuilt": (
            np.array_equal(second.own, own)
        ),
        "LightGBM's shared parts are restored": count_placeholders(model) == 0,
        "the repaired LightGBM model predicts as the original": same_predictions,
    }
    for name, passed in checks.items():
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
