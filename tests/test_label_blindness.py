"""The label-blindness convention for graph feature builders (plan §6).

A graph feature builder is called as `builder(features, edges)`: the node
feature table and an edge list, never labels. Any real builder added later
(neighbour aggregates, GNN embeddings) must pass `assert_label_blind`: its
output must be identical when every label is permuted between two calls, so
it cannot read labels by any route (an argument, a closure, a file).

The example builder lives here in tests/, not in src/, so the convention
exists before any real graph block does. All data is synthetic.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from synthetic import fake_block, fake_node_table

from fraud_detect import columns as C
from fraud_detect.features import blocks

LABEL_WORDS = ("label", "target", "y_")


def mean_neighbour_block(features: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """Example builder: the mean of each node's neighbours' own features over
    an undirected edge list (0 for isolated nodes), as block `nbr_example`."""
    X = features.set_index(C.NODE_ID).sort_index()
    n = len(X)
    src = np.concatenate([edges[0], edges[1]])
    dst = np.concatenate([edges[1], edges[0]])
    sums = np.zeros((n, X.shape[1]))
    np.add.at(sums, src, X.to_numpy()[dst])
    degree = np.bincount(src, minlength=n)[:, None]
    means = np.divide(sums, degree, out=np.zeros_like(sums), where=degree > 0)
    names = [c.replace("own__", "nbr_example__", 1) for c in X.columns]
    out = pd.DataFrame(means, columns=names)
    out.insert(0, C.NODE_ID, X.index.to_numpy())
    return out


def assert_label_blind(
    builder: Callable[[pd.DataFrame, np.ndarray], pd.DataFrame],
    nodes: pd.DataFrame,
    features: pd.DataFrame,
    edges: np.ndarray,
    seed: int = 0,
) -> None:
    """Fail unless `builder` takes no label argument and gives identical output
    after the labels in `nodes` are permuted in place."""
    params = [p.lower() for p in inspect.signature(builder).parameters]
    assert not [p for p in params if any(w in p for w in LABEL_WORDS)], params
    assert not set(features.columns) & (set(C.FORBIDDEN_FEATURE_COLUMNS) - {C.NODE_ID})
    before = builder(features.copy(), edges.copy())
    original = nodes[C.LABEL].copy()
    rng = np.random.default_rng(seed)
    try:
        nodes[C.LABEL] = original.to_numpy()[rng.permutation(len(nodes))]
        assert not nodes[C.LABEL].equals(original)  # the permutation changed labels
        after = builder(features.copy(), edges.copy())
    finally:
        nodes[C.LABEL] = original
    pd.testing.assert_frame_equal(before, after)


def graph(n: int = 50, seed: int = 0):
    nodes = fake_node_table(n=n, seed=seed)
    features = fake_block("own", n=n, width=3, seed=seed)
    features.columns = [C.NODE_ID, "own__f00", "own__f01", "own__f02"]
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, n, size=(2, 120))
    edges = np.unique(np.sort(pairs[:, pairs[0] != pairs[1]], axis=0), axis=1)
    return nodes, features, edges


def test_example_builder_is_label_blind_and_a_valid_block():
    nodes, features, edges = graph()
    assert_label_blind(mean_neighbour_block, nodes, features, edges)
    blocks.validate_block("nbr_example", mean_neighbour_block(features, edges))


def test_the_check_catches_a_builder_that_reads_labels():
    nodes, features, edges = graph()

    def leaky(features: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
        out = mean_neighbour_block(features, edges)
        # Reads labels through a closure: a neighbour fraud rate in disguise.
        out["nbr_example__fraud_rate"] = mean_neighbour_block(
            features.assign(own__f00=nodes[C.LABEL].fillna(0).to_numpy(float)), edges
        )["nbr_example__f00"]
        return out

    with pytest.raises(AssertionError):
        assert_label_blind(leaky, nodes, features, edges)


def test_the_check_rejects_a_label_argument():
    nodes, features, edges = graph()

    def takes_labels(features, edges, labels=None):
        return mean_neighbour_block(features, edges)

    with pytest.raises(AssertionError):
        assert_label_blind(takes_labels, nodes, features, edges)
