"""SHAP on tiny models, in df-analyze's environment (see tests/test_explain_dfa.py).

Fits LightGBM and CatBoost through df-analyze's own classes on synthetic rows, then
runs shap_explain.py's functions on them, which only this environment can do. It
prints one JSON line of results. Everything is in memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "dfa"))

import shap_explain as S  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)
    # Synthetic: two correlated columns and one noise column.
    a = rng.normal(size=600)
    x = pd.DataFrame(
        {"a": a, "b": a + rng.normal(0, 0.1, size=600), "c": rng.normal(size=600)}
    )
    y = (a + rng.normal(0, 0.5, size=600) > 1).astype(int)
    results = {}
    for model_cls, params in (
        ("LightGBMClassifier", {"n_estimators": 20, "num_leaves": 7, "device": "cpu"}),
        ("CatBoostClassifier", {"iterations": 20, "depth": 3, "task_type": "CPU"}),
    ):
        model = S.make_fit(model_cls, params)(x, y, 1)
        rows = x.to_numpy(dtype=float)
        background = rows[:300]
        margin = S.margin_function(model)
        prob = S.prob_of(model, x)
        gap = float(np.abs(1 / (1 + np.exp(-margin(rows))) - prob).max())
        values, base, check = S.checked_shap(model, rows[:50], background, 2)
        path_values, path_base, _ = S.checked_shap(model, rows[:50], None, 2)
        results[model_cls] = {
            "margin_vs_prob": gap,
            "shape": list(values.shape),
            "base": base,
            "background_mean": float(margin(background).mean()),
            "additivity": check["max_additivity_error"],
            "path_base": path_base,
            "modes_differ": bool(np.abs(values - path_values).max() > 1e-6),
            "top_column": ["a", "b", "c"][int(np.abs(values).mean(axis=0).argmax())],
        }
    print("RESULT " + json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
