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


def rewire_edges(edges: np.ndarray, n_nodes: int, seed: int) -> np.ndarray:
    """A random graph with (nearly) the same degrees, for a negative control.

    Every edge end ("stub") is shuffled and consecutive stubs are paired, so
    each node keeps its degree; self-loops and repeated pairs are then dropped,
    which loses a few edges on dense relations. Returns an edge list in the
    load stage's format (src < dst, int32)."""
    stubs = np.concatenate([edges[0], edges[1]]).astype(np.int64)
    rng = np.random.default_rng(seed)
    stubs = stubs[rng.permutation(len(stubs))]
    pairs = stubs[: len(stubs) // 2 * 2].reshape(-1, 2).T
    pairs = np.sort(pairs, axis=0)
    pairs = pairs[:, pairs[0] != pairs[1]]
    out = np.unique(pairs, axis=1).astype(np.int32)
    if out.size and out.max() >= n_nodes:
        raise ValueError("Rewired edges refer to nodes outside the graph")
    return out


def _contains(sorted_keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_keys, query).clip(max=max(len(sorted_keys) - 1, 0))
    return sorted_keys[idx] == query if len(sorted_keys) else np.zeros(len(query), bool)


def swap_rewire(
    edges: np.ndarray,
    n_nodes: int,
    seed: int,
    swaps_per_edge: float = 10.0,
    max_rounds: int = 500,
) -> tuple[np.ndarray, int]:
    """A random graph in which every node keeps exactly its degree: the null
    graph of control N2 (2026-10-26).

    Double-edge swaps (a-b, c-d -> a-d, c-b) are repeated until about
    `swaps_per_edge` x E were made, or `max_rounds` batches ran out. A swap is
    made only if it creates no self-loop and no edge that already exists, so
    the graph stays simple and the edge count stays E. Each batch proposes
    swaps on disjoint pairs of edges and drops any two that would create the
    same edge, so the work is vectorised. Takes only the edge list, never a
    feature or a label. Returns the new edge list in the load stage's format
    (src < dst, sorted, int32) and the number of swaps made."""
    edges = np.asarray(edges)
    src, dst = edges[0].astype(np.int64), edges[1].astype(np.int64)
    m = len(src)
    target = int(round(swaps_per_edge * m))
    rng = np.random.default_rng(seed)

    def keys(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b) * n_nodes + np.maximum(a, b)

    existing = np.sort(keys(src, dst))
    if len(np.unique(existing)) != m or (src == dst).any():
        raise ValueError("Rewiring needs a simple graph: no repeated edges or self-loops")
    done = 0
    for _ in range(max_rounds):
        if done >= target or m < 2:
            break
        perm = rng.permutation(m)
        half = m // 2
        i, j = perm[:half], perm[half : 2 * half]
        flip = rng.random(half) < 0.5  # both ways of reconnecting a pair
        a, b = src[i], dst[i]
        c, d = np.where(flip, dst[j], src[j]), np.where(flip, src[j], dst[j])
        k1, k2 = keys(a, d), keys(c, b)
        ok = (a != d) & (c != b) & (k1 != k2)
        ok &= ~_contains(existing, k1) & ~_contains(existing, k2)
        # No two swaps of one batch may create the same edge.
        idx = np.flatnonzero(ok)
        new = np.concatenate([k1[idx], k2[idx]])
        _, inverse, counts = np.unique(new, return_inverse=True, return_counts=True)
        repeated = counts[inverse] > 1
        ok[idx[repeated[: len(idx)] | repeated[len(idx) :]]] = False
        accepted = np.flatnonzero(ok)[: target - done]
        src[i[accepted]], dst[i[accepted]] = a[accepted], d[accepted]
        src[j[accepted]], dst[j[accepted]] = c[accepted], b[accepted]
        done += len(accepted)
        existing = np.sort(keys(src, dst))
    out = np.sort(np.vstack([src, dst]), axis=0)
    out = out[:, np.lexsort((out[1], out[0]))].astype(np.int32)
    return out, done


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
    name: str = C.NBR_BLOCK,
) -> pd.DataFrame:
    """The `nbr` block (or `name`, e.g. nbr_no36 when built from fewer own
    features): for each relation in order, the mean of every own feature over
    a node's neighbours (0 without neighbours) and, if `log1p_degree`, log1p of
    its degree. One row per node."""
    x, features_used = own_matrix(features)
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
        for j, feature in enumerate(features_used):
            out[f"{name}{sep}{relation}_mean_{feature}"] = means[:, j]
        if log1p_degree:
            out[f"{name}{sep}{relation}_log1p_degree"] = np.log1p(degree)
    return pd.DataFrame(out)


def build_deg_block(
    features: pd.DataFrame,
    edges_by_relation: Mapping[str, np.ndarray],
    name: str = C.DEG_BLOCK,
) -> pd.DataFrame:
    """The `deg` block (null-graph control N1): log1p of each node's degree per
    relation, with the same values as the nbr block's log1p_degree columns.
    The own block only fixes the nodes; none of its values is used."""
    x, _ = own_matrix(features)
    n = len(x)
    out: dict[str, np.ndarray] = {C.NODE_ID: np.arange(n, dtype=np.int64)}
    for relation, edges in edges_by_relation.items():
        degree = np.asarray(adjacency(edges, n).sum(axis=1)).ravel()
        out[f"{name}{C.BLOCK_SEPARATOR}{relation}_log1p_degree"] = np.log1p(degree)
    return pd.DataFrame(out)
