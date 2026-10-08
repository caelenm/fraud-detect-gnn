"""Reading the CARE-GNN .mat files: node table, relation edge lists, groups.

Each .mat holds `features` (sparse, one row per node), `label`, one
adjacency matrix per relation (`net_*`) and their union `homo`. The matrices
are symmetric, so each undirected edge is stored twice.

Two dataset quirks are handled here (docs/RESEARCH_PLAN.md §2):
- Amazon's first 3,305 nodes have no label but are stored as label 0. They
  become `label = null, is_labelled = False`, never negatives.
- Split groups: YelpChi labels are per user, so a group is a connected
  component of the same-user relation; on Amazon a group is a set of nodes
  with identical feature rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from fraud_detect import columns as C

HOMO = "homo"  # the union of all relations, as stored in the .mat


class DatasetError(RuntimeError):
    """The data does not match what the config says it should be."""


@dataclass(frozen=True)
class RawGraph:
    features: np.ndarray  # (n_nodes, n_features), float64
    labels: np.ndarray  # (n_nodes,), int, as stored (0 for unlabelled nodes too)
    adjacency: dict[str, sp.csr_matrix]  # relation name (and HOMO) -> matrix

    @property
    def n_nodes(self) -> int:
        return int(self.features.shape[0])


def read_mat(path: Path, relations: dict[str, str]) -> RawGraph:
    """Read features, labels and the relations named in `relations`
    (short name -> .mat key), plus `homo`."""
    keys = ["features", "label", HOMO, *relations.values()]
    mat = scipy.io.loadmat(str(path), variable_names=keys, spmatrix=False)
    missing = [k for k in keys if k not in mat]
    if missing:
        raise DatasetError(f"{path.name} has no {missing}")
    features = mat["features"]
    features = features.toarray() if sp.issparse(features) else np.asarray(features)
    labels = np.asarray(mat["label"]).ravel()
    if not np.isin(labels, (0, 1)).all():
        raise DatasetError(f"{path.name}: labels other than 0/1")
    adjacency = {name: sp.csr_matrix(mat[key]) for name, key in relations.items()}
    adjacency[HOMO] = sp.csr_matrix(mat[HOMO])
    return RawGraph(
        features=features.astype(np.float64),
        labels=labels.astype(np.int64),
        adjacency=adjacency,
    )


def read_features(path: Path) -> np.ndarray:
    """Only the feature matrix (for the features stage)."""
    mat = scipy.io.loadmat(str(path), variable_names=["features"], spmatrix=False)
    features = mat["features"]
    features = features.toarray() if sp.issparse(features) else np.asarray(features)
    return features.astype(np.float64)


def edge_list(adjacency: sp.spmatrix) -> np.ndarray:
    """Undirected edges of a (normally symmetric) adjacency matrix as a (2, E)
    int32 array: each edge once with src < dst, no self-loops, sorted.

    Any stored non-zero entry (i, j) is an edge between i and j, so an
    asymmetric matrix is treated as undirected rather than losing edges."""
    coo = sp.coo_matrix(adjacency)
    keep = (coo.data != 0) & (coo.row != coo.col)
    rows, cols = coo.row[keep].astype(np.int64), coo.col[keep].astype(np.int64)
    src, dst = np.minimum(rows, cols), np.maximum(rows, cols)
    pairs = np.unique(np.stack([src, dst], axis=1), axis=0)
    if len(pairs) and pairs.max() > np.iinfo(np.int32).max:
        raise DatasetError("Node IDs do not fit in int32")
    return pairs.T.astype(np.int32).reshape(2, -1)


def component_groups(edges: np.ndarray, n_nodes: int) -> np.ndarray:
    """Connected-component ID of every node (isolated nodes are their own
    component). IDs are numbered by each component's smallest node ID."""
    adj = sp.coo_matrix(
        (np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n_nodes, n_nodes)
    )
    _, labels = connected_components(adj, directed=False)
    return _renumber(labels)


def duplicate_groups(features: np.ndarray) -> np.ndarray:
    """Group ID of every node: nodes with identical feature rows share one.
    IDs are numbered by each group's smallest node ID."""
    _, inverse = np.unique(features, axis=0, return_inverse=True)
    return _renumber(inverse.ravel())


def _renumber(labels: np.ndarray) -> np.ndarray:
    """Relabel groups 0, 1, 2, ... in order of their first node, so group IDs
    do not depend on how a library happened to number them."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.argsort(np.argsort(first, kind="stable"), kind="stable")
    return rank[inverse.ravel()].astype(np.int64)


def group_ids(
    rule: dict[str, Any], features: np.ndarray, edges: dict[str, np.ndarray]
) -> np.ndarray:
    """Split group of every node, by the dataset's grouping rule."""
    kind = rule["rule"]
    if kind == "components":
        return component_groups(edges[rule["relation"]], len(features))
    if kind == "duplicate_features":
        return duplicate_groups(features)
    raise DatasetError(f"Unknown grouping rule {kind!r}")


def node_table(
    labels: np.ndarray, unlabelled_prefix: int, groups: np.ndarray
) -> pd.DataFrame:
    """One row per node. Nodes 0 .. unlabelled_prefix-1 are unlabelled: their
    stored label (always 0) is discarded, never used as a negative.
    `split` and `cv_fold` are filled in by the split stage."""
    n = len(labels)
    if not 0 <= unlabelled_prefix <= n:
        raise DatasetError(f"unlabelled_prefix {unlabelled_prefix} outside 0..{n}")
    if labels[:unlabelled_prefix].any():
        raise DatasetError(
            f"Nodes below {unlabelled_prefix} should be unlabelled (stored as 0), "
            "but some are labelled 1; the file is not the one the config describes."
        )
    is_labelled = np.arange(n) >= unlabelled_prefix
    label = pd.array(np.where(is_labelled, labels, 0), dtype="Int8")
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


def count_mismatches(
    found: dict[str, Any], expected: dict[str, Any], prefix: str = ""
) -> list[str]:
    """Differences between observed and expected counts (nested for edges)."""
    problems = []
    for key, want in expected.items():
        name = prefix + key
        got = found.get(key)
        if isinstance(want, dict):
            problems += count_mismatches(got or {}, want, name + ".")
        elif got != want:
            problems.append(f"{name}: expected {want}, found {got}")
    return problems


def observed_counts(
    graph: RawGraph, nodes: pd.DataFrame, edges: dict[str, np.ndarray]
) -> dict[str, Any]:
    labelled = nodes[C.IS_LABELLED]
    return {
        "n_nodes": graph.n_nodes,
        "n_labelled": int(labelled.sum()),
        "n_positive": int(nodes.loc[labelled, C.LABEL].sum()),
        "n_features": int(graph.features.shape[1]),
        "n_edges": {name: int(e.shape[1]) for name, e in edges.items()},
    }


def check_counts(found: dict[str, Any], expected: dict[str, Any], name: str) -> None:
    problems = count_mismatches(found, expected)
    if problems:
        raise DatasetError(
            f"{name} does not match the expected counts (docs/RESEARCH_PLAN.md §2):\n- "
            + "\n- ".join(problems)
            + "\nStop and report this rather than working around it."
        )


def check_edges_in_range(edges: dict[str, np.ndarray], n_nodes: int) -> None:
    for name, e in edges.items():
        if e.size and (e.min() < 0 or e.max() >= n_nodes):
            raise DatasetError(f"Relation {name} has node IDs outside 0..{n_nodes - 1}")


def relation_names(relations: dict[str, str]) -> Sequence[str]:
    """The relation files written by `load`: each relation, then `homo`."""
    return [*relations, HOMO]
