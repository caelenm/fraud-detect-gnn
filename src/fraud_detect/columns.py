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

# The label column in the df-analyze input tables.
TARGET = "target"

# Feature columns are "<block>__<name>", e.g. own__f00 (see features/blocks.py).
BLOCK_SEPARATOR = "__"
OWN_BLOCK = "own"
# Graph-derived blocks, one per kind, covering every relation of the dataset:
# nbr__upu_mean_f00 (label-free neighbour means, Model 2), gnnssl__upu_pc01
# (self-supervised GNN embeddings, Model 5) and gnnrand__upu_pc01 (the same
# GNN encoder with its seeded initial weights and no training: the untrained
# control of phase 3, Task 3a).
NBR_BLOCK = "nbr"
GNNSSL_BLOCK = "gnnssl"
GNNRAND_BLOCK = "gnnrand"
# The blocks the gnn_embed stage builds.
GNN_BLOCK_KINDS: tuple[str, ...] = (GNNSSL_BLOCK, GNNRAND_BLOCK)
# Every block name is one of these kinds, optionally followed by _<variant>
# (see features/blocks.py).
BLOCK_KINDS: tuple[str, ...] = (OWN_BLOCK, NBR_BLOCK, *GNN_BLOCK_KINDS)

# Columns that identify, label or place a node. They must never appear in any
# feature block or model input (AGENTS.md leakage invariant 7).
FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    NODE_ID,
    LABEL,
    IS_LABELLED,
    GROUP_ID,
    SPLIT,
    CV_FOLD,
    TARGET,
)


def feature_name(index: int, n_features: int) -> str:
    """Anonymous feature name in .mat column order: f00, f01, ... (wide enough
    that names sort in column order)."""
    width = max(2, len(str(n_features - 1)))
    return f"f{index:0{width}d}"
