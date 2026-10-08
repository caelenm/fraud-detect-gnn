"""Grouped, stratified train/test split and shared CV folds (plan §4).

Every labelled node belongs to a split group (a YelpChi user, an Amazon set
of identical feature rows). Whole groups go to train or test, and whole
training groups go to one CV fold, so the same user's reviews (or copies of
the same row) can never be on both sides of any comparison.

`StratifiedGroupKFold` splits the labelled nodes into `n_folds` folds that
keep each fold's positive rate close to the overall one; the first
`test_folds` folds are the test set (2 of 5 = 40%). The training nodes are
then split again into `cv_folds` grouped, stratified folds, which every model
choice (select_model) and any later out-of-fold feature uses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from fraud_detect import columns as C


class SplitError(RuntimeError):
    """The split is inconsistent; the message says how."""


def grouped_folds(
    y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int
) -> np.ndarray:
    """Fold number (0 .. n_folds-1) of every row, whole groups per fold,
    stratified by `y`. Deterministic for a given seed."""
    folds = np.full(len(y), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, held_out) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        folds[held_out] = k
    if (folds < 0).any():
        raise SplitError("Some rows were not assigned to a fold")
    return folds


def split_nodes(nodes: pd.DataFrame, cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """A copy of the node table with `split` (train/test/unlabelled) and
    `cv_fold` (training nodes only) filled in."""
    out = nodes.copy()
    labelled = out[C.IS_LABELLED].to_numpy()
    y = out.loc[labelled, C.LABEL].to_numpy(dtype=np.int64)
    groups = out.loc[labelled, C.GROUP_ID].to_numpy()
    outer = grouped_folds(y, groups, int(cfg["n_folds"]), seed)
    is_test = outer < int(cfg["test_folds"])

    split = np.full(len(out), C.UNLABELLED, dtype=object)
    split[np.flatnonzero(labelled)] = np.where(is_test, C.TEST, C.TRAIN)
    out[C.SPLIT] = pd.array(split, dtype="string")

    train_rows = np.flatnonzero(labelled)[~is_test]
    cv = grouped_folds(y[~is_test], groups[~is_test], int(cfg["cv_folds"]), seed)
    fold = pd.array([None] * len(out), dtype="Int8")
    fold[train_rows] = cv
    out[C.CV_FOLD] = fold
    check_split(out)
    return out


def check_split(nodes: pd.DataFrame) -> dict[str, int]:
    """Invariant checks (leakage invariants 2 and 3). Raises SplitError on any
    violation; returns the counts it checked (all zero) for the summary."""
    labelled = nodes[C.IS_LABELLED]
    split = nodes[C.SPLIT]
    checks = {
        "labelled_nodes_without_split": int(
            (labelled & ~split.isin([C.TRAIN, C.TEST])).sum()
        ),
        "unlabelled_nodes_in_train_or_test": int(
            (~labelled & split.isin([C.TRAIN, C.TEST])).sum()
        ),
        "groups_in_both_train_and_test": int(
            nodes[labelled].groupby(C.GROUP_ID)[C.SPLIT].nunique().gt(1).sum()
        ),
        "training_nodes_without_cv_fold": int(
            ((split == C.TRAIN) & nodes[C.CV_FOLD].isna()).sum()
        ),
        "non_training_nodes_with_cv_fold": int(
            ((split != C.TRAIN) & nodes[C.CV_FOLD].notna()).sum()
        ),
        "groups_in_more_than_one_cv_fold": int(
            nodes[split == C.TRAIN].groupby(C.GROUP_ID)[C.CV_FOLD].nunique().gt(1).sum()
        ),
    }
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        raise SplitError(f"The split violates the grouping rules: {bad}")
    return checks


def ids_table(nodes: pd.DataFrame, part: str) -> pd.DataFrame:
    """node_id, label, group_id of one part, in node_id order."""
    rows = nodes[nodes[C.SPLIT] == part].sort_values(C.NODE_ID)
    return rows[[C.NODE_ID, C.LABEL, C.GROUP_ID]].astype("int64").reset_index(drop=True)


def folds_table(nodes: pd.DataFrame) -> pd.DataFrame:
    rows = nodes[nodes[C.SPLIT] == C.TRAIN].sort_values(C.NODE_ID)
    return pd.DataFrame(
        {
            "node_id": rows[C.NODE_ID].astype("int64"),
            "fold": rows[C.CV_FOLD].astype("int64"),
        }
    ).reset_index(drop=True)


def summary(nodes: pd.DataFrame, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Sizes, positive rates and group counts per part, and the checks."""

    def part(frame: pd.DataFrame) -> dict[str, Any]:
        y = frame[C.LABEL].astype("float64")
        return {
            "n_nodes": len(frame),
            "n_positive": int(y.sum()),
            "positive_rate": float(y.mean()) if len(frame) else None,
            "n_groups": int(frame[C.GROUP_ID].nunique()),
        }

    train = nodes[nodes[C.SPLIT] == C.TRAIN]
    return {
        "seed": seed,
        "method": "StratifiedGroupKFold: test = the first "
        f"{cfg['test_folds']} of {cfg['n_folds']} folds; {cfg['cv_folds']} grouped "
        "CV folds on train",
        "train": part(train),
        "test": part(nodes[nodes[C.SPLIT] == C.TEST]),
        "unlabelled": int((nodes[C.SPLIT] == C.UNLABELLED).sum()),
        "cv_folds": {str(k): part(f) for k, f in train.groupby(C.CV_FOLD, sort=True)},
        "checks": check_split(nodes),
    }


def check_against_saved_ids(
    nodes: pd.DataFrame, train_ids: pd.DataFrame, test_ids: pd.DataFrame
) -> None:
    """Check that the node table's split columns agree with the saved ID
    files (they disagree if `load` was rerun without `split`)."""
    for part, ids in ((C.TRAIN, train_ids), (C.TEST, test_ids)):
        mine = nodes.loc[nodes[C.SPLIT] == part, C.NODE_ID].sort_values().to_numpy()
        if not np.array_equal(mine, np.sort(ids[C.NODE_ID].to_numpy())):
            raise SplitError(
                f"nodes.parquet and the saved {part} IDs disagree; rerun the split "
                "stage (uv run run.py --dataset <name> --from split --force)."
            )
