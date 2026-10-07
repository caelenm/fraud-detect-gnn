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


def fake_care_gnn_mat(
    n_nodes: int = 60, n_features: int = 5, unlabelled_prefix: int = 8, seed: int = 0
) -> dict:
    """Invented contents of a CARE-GNN-style .mat file (for scipy.io.savemat):
    sparse `features`, a 1 x n `label` row (0 for the unlabelled prefix, as in
    Amazon), two symmetric relations and their union `homo`.

    Planted structure for the tests: labelled nodes p and p+1 have identical
    feature rows (a duplicate group); relation `net_aaa` links nodes in pairs
    (2k, 2k+1) for k < 10, so those pairs are its connected components."""
    import scipy.sparse as sp  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    features = rng.integers(0, 9, size=(n_nodes, n_features)).astype(float)
    features[:, 0] = np.arange(n_nodes)  # unique rows unless planted below
    p = unlabelled_prefix
    features[p + 1] = features[p]
    labels = (rng.random(n_nodes) < 0.3).astype(float)
    labels[:p] = 0  # unlabelled nodes are stored as 0
    labels[p], labels[p + 2] = 1, 0  # both classes among labelled nodes

    def symmetric(pairs: list[tuple[int, int]]) -> sp.csc_matrix:
        rows = [a for a, b in pairs] + [b for a, b in pairs]
        cols = [b for a, b in pairs] + [a for a, b in pairs]
        return sp.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes,) * 2)

    pairs_a = [(2 * k, 2 * k + 1) for k in range(10)]
    pairs_b = [(int(a), int(b)) for a, b in rng.integers(0, n_nodes, (30, 2)) if a != b]
    a, b = symmetric(pairs_a), symmetric(pairs_b)
    homo = ((a + b) > 0).astype(float)
    return {
        "features": sp.csc_matrix(features),
        "label": labels.reshape(1, -1),
        "net_aaa": a,
        "net_bbb": b,
        "homo": sp.csc_matrix(homo),
    }


def fake_node_table(
    n: int = 400, unlabelled: int = 0, group_size: int = 3, seed: int = 0
) -> pd.DataFrame:
    """Invented node table shaped like the load stage's output: the first
    `unlabelled` nodes have no label; labelled nodes come in groups of
    `group_size` consecutive nodes that share one label (like a YelpChi user's
    reviews), about 15% positive."""
    rng = np.random.default_rng(seed)
    groups = np.arange(n) // group_size
    group_label = (rng.random(groups.max() + 1) < 0.15).astype(int)
    label = pd.array(group_label[groups], dtype="Int8")
    is_labelled = np.arange(n) >= unlabelled
    label[~is_labelled] = pd.NA
    split = pd.array([None] * n, dtype="string")
    split[~is_labelled] = C.UNLABELLED
    return pd.DataFrame(
        {
            C.NODE_ID: np.arange(n, dtype=np.int64),
            C.LABEL: label,
            C.IS_LABELLED: is_labelled,
            C.GROUP_ID: groups.astype(np.int64),
            C.SPLIT: split,
            C.CV_FOLD: pd.array([None] * n, dtype="Int8"),
        }
    )


def fake_block(name: str, n: int = 10, width: int = 3, seed: int = 0) -> pd.DataFrame:
    """Invented feature block: node_id plus <name>__x0 ... columns of noise."""
    rng = np.random.default_rng(seed)
    block = pd.DataFrame(rng.normal(size=(n, width))).add_prefix(f"{name}__x")
    block.insert(0, C.NODE_ID, np.arange(n, dtype=np.int64))
    return block


def fake_inferred_types(kinds: dict[str, str]) -> pd.DataFrame:
    """Invented df-analyze inspection/inferred_types.csv: index feature_name,
    columns user, inferred, reason (as written by InspectionResults.basic_df)."""
    frame = pd.DataFrame(
        {
            "user": ["ord" if k.startswith("user-ord") else "" for k in kinds.values()],
            "inferred": list(kinds.values()),
            "reason": ["fake reason"] * len(kinds),
        },
        index=pd.Index(list(kinds), name="feature_name"),
    )
    return frame
