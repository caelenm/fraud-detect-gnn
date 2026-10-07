"""Canonical column names used throughout the pipeline.

Every dataset is a graph whose nodes are the examples (YelpChi: reviews;
Amazon: users). The node table has one row per node, keyed by `node_id`, the
node's row index in the source `.mat` file.
"""

from __future__ import annotations

NODE_ID = "node_id"
LABEL = "label"  # nullable: null for unlabelled nodes (Amazon's first 3,305)
IS_LABELLED = "is_labelled"
GROUP_ID = "group_id"  # split group: a YelpChi user, an Amazon duplicate set
SPLIT = "split"  # train / test / unlabelled
CV_FOLD = "cv_fold"  # shared CV fold of a training node; null otherwise

TRAIN, TEST, UNLABELLED = "train", "test", "unlabelled"

# Feature columns are "<block>__<name>", e.g. own__f00 (see features/blocks.py).
BLOCK_SEPARATOR = "__"
OWN_BLOCK = "own"

# Columns that identify, label or place a node. They must never appear in any
# feature block or model input (AGENTS.md leakage invariant 7).
FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    NODE_ID,
    LABEL,
    IS_LABELLED,
    GROUP_ID,
    SPLIT,
    CV_FOLD,
)


def feature_name(index: int, n_features: int) -> str:
    """Anonymous feature name in .mat column order: f00, f01, ... (wide enough
    that names sort in column order)."""
    width = max(2, len(str(n_features - 1)))
    return f"f{index:0{width}d}"
