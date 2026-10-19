"""Model 2's `nbr` block: label-free neighbour means per relation.

For each relation (an undirected edge list from the load stage) and every
node: the mean of its neighbours' own features, and log1p of its degree (the
number of distinct neighbours). A node without neighbours gets 0 for every
mean. Columns are named nbr__<relation>_mean_<feature> and
nbr__<relation>_log1p_degree, e.g. nbr__upu_mean_f00.

The builder follows the graph-feature convention (features/blocks.py): it
takes the own block and edge lists and never sees a label. It uses the
features of every node, test and unlabelled ones included, as AGENTS.md
invariant 5 allows; tests/test_graph_blocks.py checks it with permuted labels.
Means are a row-normalised sparse adjacency times the feature matrix (scipy),
so no per-edge copies of the features are made.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fraud_detect import columns as C


def adjacency(edges: np.ndarray, n_nodes: int) -> sp.csr_matrix:
    """Symmetric 0/1 adjacency of an undirected (2, E) edge list that holds each
    edge once (src < dst, as the load stage writes it)."""
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"Edges must have shape (2, E), got {edges.shape}")
    if edges.size and (edges.min() < 0 or edges.max() >= n_nodes):
        raise ValueError(f"Edges refer to nodes outside 0..{n_nodes - 1}")
    rows = np.concatenate([edges[0], edges[1]])
    cols = np.concatenate([edges[1], edges[0]])
    a = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float64
    )
    a.sum_duplicates()
    a.data[:] = 1.0  # a repeated edge still counts once
    return a


def own_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """The own block as a matrix in node order (row i = node i), and its
    feature names without the block prefix (f00, f01, ...)."""
    ordered = features.sort_values(C.NODE_ID)
    ids = ordered[C.NODE_ID].to_numpy()
    if not np.array_equal(ids, np.arange(len(ordered))):
        raise ValueError("The own block must have exactly one row per node 0..n-1")
    prefix = C.OWN_BLOCK + C.BLOCK_SEPARATOR
    cols = [c for c in ordered.columns if c != C.NODE_ID]
    if not cols or not all(c.startswith(prefix) for c in cols):
        raise ValueError(f"Expected only {prefix}* feature columns, got {cols}")
    return ordered[cols].to_numpy(dtype=np.float64), [c[len(prefix) :] for c in cols]


def build_nbr_block(
    features: pd.DataFrame,
    edges_by_relation: Mapping[str, np.ndarray],
    log1p_degree: bool = True,
) -> pd.DataFrame:
    """The `nbr` block: for each relation in order, the mean of every own
    feature over a node's neighbours (0 without neighbours) and, if
    `log1p_degree`, log1p of its degree. One row per node."""
    x, names = own_matrix(features)
    n = len(x)
    sep = C.BLOCK_SEPARATOR
    out: dict[str, np.ndarray] = {C.NODE_ID: np.arange(n, dtype=np.int64)}
    for relation, edges in edges_by_relation.items():
        a = adjacency(edges, n)
        degree = np.asarray(a.sum(axis=1)).ravel()
        sums = a @ x
        means = np.divide(
            sums, degree[:, None], out=np.zeros_like(sums), where=degree[:, None] > 0
        )
        for j, name in enumerate(names):
            out[f"{C.NBR_BLOCK}{sep}{relation}_mean_{name}"] = means[:, j]
        if log1p_degree:
            out[f"{C.NBR_BLOCK}{sep}{relation}_log1p_degree"] = np.log1p(degree)
    return pd.DataFrame(out)
