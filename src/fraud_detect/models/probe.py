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
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detect import columns as C
from fraud_detect.features import blocks

# Fixed in advance and never tuned: sklearn's defaults, without early stopping.
PROBE_PARAMS = {"max_iter": 100, "learning_rate": 0.1, "early_stopping": False}
BLOCK_SETS: dict[str, list[str]] = {
    "own": [C.OWN_BLOCK],
    "own+nbr": [C.OWN_BLOCK, C.NBR_BLOCK],
    "own+gnnssl": [C.OWN_BLOCK, C.GNNSSL_BLOCK],
    "own+gnnrand": [C.OWN_BLOCK, C.GNNRAND_BLOCK],
}
# Permutation importance (feature_ranking): shuffles per feature and fold.
PERMUTATION_REPEATS = 10


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


def single_feature_check(
    own: pd.DataFrame, train_ids: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """How well each column alone separates the TRAINING classes: AUROC and
    PR-AUC of the column oriented so that higher means fraud, and the best
    balanced accuracy of a single threshold on it (issue #36)."""
    table = own.set_index(C.NODE_ID).loc[train_ids[C.NODE_ID].to_numpy()]
    y = train_ids[C.LABEL].to_numpy(dtype=int)
    rows = []
    for col in columns:
        v = table[col].to_numpy(dtype=float)
        sign = 1.0 if roc_auc_score(y, v) >= 0.5 else -1.0
        score = sign * v
        best = 0.0
        for t in np.unique(score):
            pred = score >= t
            tpr = pred[y == 1].mean()
            tnr = (~pred[y == 0]).mean()
            best = max(best, (tpr + tnr) / 2)
        rows.append(
            {
                "column": col,
                "direction": "higher = fraud" if sign > 0 else "lower = fraud",
                "auroc": float(roc_auc_score(y, score)),
                "pr_auc": float(average_precision_score(y, score)),
                "best_threshold_balanced_accuracy": float(best),
                "n_unique": int(len(np.unique(v))),
                "n_train": len(y),
            }
        )
    return pd.DataFrame(rows)


def vote_ratio_check(
    own: pd.DataFrame, train_ids: pd.DataFrame, votes: str, low: float = 0.2,
    high: float = 0.8,
) -> pd.DataFrame:  # fmt: skip
    """Issue #36: is some column c the 'helpful votes' to `votes`'s 'unhelpful
    votes'? The label rule is helpful share < `low` = fraud, > `high` =
    benign, so for that c the ratio c / (c + votes) separates the training
    classes almost perfectly. One row per other non-negative column, best
    first: AUROC of the ratio (higher = benign), the share of fraud users
    below `low` and of benign users above `high` (users with c + votes = 0
    count as neither), and the smallest c + votes. Training rows only."""
    table = own.set_index(C.NODE_ID).loc[train_ids[C.NODE_ID].to_numpy()]
    y = train_ids[C.LABEL].to_numpy(dtype=int)
    v = table[votes].to_numpy(dtype=float)
    rows = []
    for col in table.columns:
        c = table[col].to_numpy(dtype=float)
        if col == votes or (c < 0).any():
            continue
        total = c + v
        ratio = np.divide(c, total, out=np.full_like(c, np.nan), where=total > 0)
        known = ~np.isnan(ratio)
        auroc = (
            float(roc_auc_score(1 - y[known], ratio[known]))
            if len(np.unique(y[known])) == 2 else float("nan")
        )  # fmt: skip
        rows.append(
            {
                "column": col,
                "ratio_auroc_benign": auroc,
                "fraud_below_low": float((ratio[y == 1] < low).mean()),
                "benign_above_high": float((ratio[y == 0] > high).mean()),
                "share_with_no_votes": float((total == 0).mean()),
                "min_total": float(total.min()),
                "n_train": len(y),
            }
        )
    return pd.DataFrame(rows).sort_values("ratio_auroc_benign", ascending=False)


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


def feature_ranking(
    own: pd.DataFrame, train_ids: pd.DataFrame, folds: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """A training-only ranking of the own features (phase 3, Task 5): for each
    saved CV fold, the probe's fixed HistGradientBoosting is fit on the other
    training folds, and each feature's permutation importance is the drop in
    the held-out fold's PR-AUC when that feature is shuffled; the importance
    is averaged over folds. Next to it, each feature's univariate AUROC on the
    training nodes. Never reads a test row."""
    x, y, fold, columns = training_rows(own, train_ids, folds)
    per_fold = []
    for k in np.unique(fold):
        held_out = fold == k
        model = HistGradientBoostingClassifier(**PROBE_PARAMS, random_state=seed)
        model.fit(x[~held_out], y[~held_out])
        result = permutation_importance(
            model, x[held_out], y[held_out], scoring="average_precision",
            n_repeats=PERMUTATION_REPEATS, random_state=seed,
        )  # fmt: skip
        per_fold.append(result.importances_mean)
    imp = np.vstack(per_fold)
    table = pd.DataFrame(
        {
            "column": columns,
            "perm_importance_mean": imp.mean(axis=0),
            "perm_importance_sd": imp.std(axis=0, ddof=1),
            "perm_importance_folds": [
                " ".join(f"{v:.4f}" for v in imp[:, j]) for j in range(len(columns))
            ],
        }
    )
    single = single_feature_check(own, train_ids, columns).set_index("column")
    table["univariate_auroc"] = table["column"].map(single["auroc"])
    table["rank_permutation"] = (
        table["perm_importance_mean"].rank(ascending=False, method="first").astype(int)
    )
    table["rank_univariate_auroc"] = (
        table["univariate_auroc"].rank(ascending=False, method="first").astype(int)
    )
    table["n_train"] = len(y)
    return table.sort_values("rank_permutation").reset_index(drop=True)


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
