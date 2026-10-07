"""
=====================================================================
SYNTHETIC TEST DATA. NOT REAL YELPCHI OR AMAZON DATA.
=====================================================================

Every row used by the test suite is built by a function in this module and
nowhere else. All values are invented: random feature values, made-up node
IDs and labels, and graphs drawn at random. Nothing describes a real review,
user or model.

Nothing here is written to data/ or outputs/. Tests that need files write
them to pytest's temporary directory (`tmp_path`), which is deleted
automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detect import columns as C

# (config name, df-analyze estimator class) for fake df-analyze results.
FAKE_DFA_MODELS = [("dummy", "DummyClassifier"), ("lgbm", "LightGBMClassifier"),
                   ("catboost", "CatBoostClassifier")]  # fmt: skip


def fake_df_analyze_results(
    y_test: np.ndarray, cv_scores: dict[tuple[str, str], float], seed: int = 0
) -> tuple[list[dict], pd.DataFrame]:
    """Invented df-analyze outputs: `prediction_results` entries and a
    `tuned_models` table, one per (model, selection) key in `cv_scores`.
    Probabilities are random noise plus a model-specific signal; they describe
    no real model."""
    rng = np.random.default_rng(seed)
    classes = dict(FAKE_DFA_MODELS)
    entries, tuned = [], []
    for i, ((model, selection), score) in enumerate(cv_scores.items()):
        signal = 0.0 if model == "dummy" else 0.15 * (i + 1)
        p1 = np.clip(rng.uniform(0, 0.6, len(y_test)) + signal * y_test, 0, 1)
        entries.append(
            {
                "model_cls": classes[model],
                "selection": selection,
                "embed_select_model": "linear" if selection == "embed" else None,
                "metric": "Accuracy",
                "score": score,
                "probs_test": np.column_stack([1 - p1, p1]).tolist(),
                "preds_test": (p1 > 0.5).astype(int).tolist(),
            }
        )
        tuned.append(
            {
                "selection": selection,
                "embed_selector": "linear" if selection == "embed" else "none",
                "model": model,
                "params": "{}",
                "metric": "acc",
                "score": score,
                "test_idx": 0,
            }
        )
    return entries, pd.DataFrame(tuned)


def fake_test_meta(n: int, n_features: int = 4, seed: int = 0) -> pd.DataFrame:
    """Invented test-node details for the web report, in test order: node IDs
    and each node's top features as (name, value, percentile) tuples."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            C.NODE_ID: [9000 + i for i in range(n)],
            "top_features": [
                [(f"own__f{j:02d}", float(rng.uniform()), float(rng.uniform()))
                 for j in range(n_features)]
                for _ in range(n)
            ],
        }
    )  # fmt: skip


def fake_shared_cv(
    pr_auc: dict[tuple[str, str], float],
    std: float = 0.01,
    default_offset: float = -0.05,
) -> pd.DataFrame:
    """Invented output of scripts/dfa/cv_select.py: for each (model, selection)
    key, a tuned row and a default-settings row whose PR-AUC is the tuned one
    plus `default_offset`. The scores describe no real model."""
    classes = dict(FAKE_DFA_MODELS)
    rows = []
    for (model, selection), score in pr_auc.items():
        for settings, value in (("tuned", score), ("default", score + default_offset)):
            rows.append(
                {
                    "model_cls": classes[model],
                    "selection": selection,
                    "embed_selector": "linear" if selection == "embed" else "",
                    "settings": settings,
                    "pr_auc_mean": value,
                    "pr_auc_std": std,
                    "auroc_mean": 0.5 + value / 2,
                    "bal_acc_mean": 0.5,
                    "error": "",
                }
            )
    return pd.DataFrame(rows)


def fake_model_tables(
    n_train: int = 180, n_test: int = 120, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Invented df-analyze input tables: continuous, ordinal and binary `own__`
    columns plus `target`, shaped like the df_analyze_input stage's output.
    Features are random noise, so none predicts the label."""
    rng = np.random.default_rng(seed)

    def table(n: int) -> pd.DataFrame:
        y = (rng.random(n) < 0.2).astype(int)
        y[:2] = [0, 1]  # both classes
        return pd.DataFrame(
            {
                "own__f00": rng.normal(size=n),
                "own__f01": rng.uniform(size=n),
                "own__f02": rng.integers(0, 6, n),
                "own__f03": rng.integers(0, 2, n),
                "target": y,
            }
        )

    return table(n_train), table(n_test)
