"""A quick probe of feature blocks on the training set's grouped CV folds.

Before spending a df-analyze run on a feature set, fit one fixed, untuned
HistGradientBoostingClassifier per saved CV fold (the folds select_model
uses) on own, own+nbr and own+gnnssl, and score the held-out fold by PR-AUC.
It gives an early, cheap signal; it is not a result, and nothing is chosen
from it on this branch.

Only training nodes are read: the feature table is restricted to the IDs in
train_ids.csv before anything is fit, so test rows cannot influence it
(tests/test_probe.py fills test rows with garbage and expects the same
scores). Test results are never used here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detect import columns as C
from fraud_detect.features import blocks

# Fixed in advance and never tuned: sklearn's defaults, without early stopping.
PROBE_PARAMS = {"max_iter": 100, "learning_rate": 0.1, "early_stopping": False}
BLOCK_SETS: dict[str, list[str]] = {
    "own": [C.OWN_BLOCK],
    "own+nbr": [C.OWN_BLOCK, C.NBR_BLOCK],
    "own+gnnssl": [C.OWN_BLOCK, C.GNNSSL_BLOCK],
}


def block_sets(variants: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """BLOCK_SETS, then the same sets for each input variant (own_no36, ...)."""
    out = dict(BLOCK_SETS)
    for v in variants:
        for names in BLOCK_SETS.values():
            renamed = [f"{b}_{v}" for b in names]
            out["+".join(renamed)] = renamed
    return out


def training_rows(
    features: pd.DataFrame, train_ids: pd.DataFrame, folds: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """X, y and the CV fold of every training node (train_ids.csv order);
    nothing from any other node."""
    table = features.set_index(C.NODE_ID)
    ids = train_ids[C.NODE_ID].to_numpy()
    fold_of = folds.set_index(C.NODE_ID)["fold"]
    missing = set(ids) - set(fold_of.index)
    if missing:
        raise ValueError(f"{len(missing)} training nodes have no CV fold")
    columns = list(table.columns)
    x = table.loc[ids, columns].to_numpy(dtype=float)
    y = train_ids[C.LABEL].to_numpy(dtype=int)
    return x, y, fold_of.loc[ids].to_numpy(dtype=int), columns


def cv_scores(x: np.ndarray, y: np.ndarray, fold: np.ndarray, seed: int) -> dict:
    """Mean and SD (ddof=1, as select_model) of the held-out folds' PR-AUC and
    AUROC for one fixed HistGradientBoostingClassifier per fold."""
    pr, roc = [], []
    for k in np.unique(fold):
        held_out = fold == k
        model = HistGradientBoostingClassifier(**PROBE_PARAMS, random_state=seed)
        model.fit(x[~held_out], y[~held_out])
        prob = model.predict_proba(x[held_out])[:, 1]
        pr.append(average_precision_score(y[held_out], prob))
        roc.append(roc_auc_score(y[held_out], prob))
    return {
        "n_folds": len(pr),
        "cv_pr_auc_mean": float(np.mean(pr)),
        "cv_pr_auc_std": float(np.std(pr, ddof=1)),
        "cv_auroc_mean": float(np.mean(roc)),
        "cv_auroc_std": float(np.std(roc, ddof=1)),
        "cv_pr_auc_folds": " ".join(f"{v:.4f}" for v in pr),
    }


def probe(
    features_dir: Path,
    train_ids: pd.DataFrame,
    folds: pd.DataFrame,
    seed: int,
    block_sets: Mapping[str, Sequence[str]] = BLOCK_SETS,
    variants: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """One row per block set whose blocks all exist; sets with a missing block
    are listed with a note instead of scores. own_<variant> blocks are own
    without the variant's columns (`variants`, the config's feature_variants)."""
    rows = []
    for name, names in block_sets.items():
        stored = [C.OWN_BLOCK if b.startswith(C.OWN_BLOCK) else b for b in names]
        absent = [b for b in stored if not (features_dir / f"{b}.parquet").is_file()]
        if absent:
            rows.append(
                {"blocks": name, "note": f"missing block(s): {', '.join(absent)}"}
            )
            continue
        features = blocks.load_feature_set(
            features_dir, list(names), {k: list(v) for k, v in (variants or {}).items()}
        )
        x, y, fold, columns = training_rows(features, train_ids, folds)
        rows.append(
            {
                "blocks": name,
                "n_features": len(columns),
                "n_train": len(y),
                **cv_scores(x, y, fold, seed),
                "note": "",
            }
        )
    return pd.DataFrame(rows)
