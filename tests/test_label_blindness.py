"""The label-blindness convention for graph feature builders (plan §6).

A graph feature builder is called as `builder(features, edges)`: the node
feature table and its edges, never labels. Every real builder (neighbour
aggregates, GNN embeddings; tests/test_graph_blocks.py, tests/test_gnn_ssl.py)
must pass `assert_label_blind` (tests/helpers.py): its output must be
identical when every label is permuted between two calls, so it cannot read
labels by any route (an argument, a closure, a file).

The example builder lives here in tests/, not in src/; it shows the
convention on its own. All data is synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import assert_label_blind, label_like
from synthetic import fake_block, fake_edges, fake_node_table

from fraud_detect import columns as C
from fraud_detect.features import blocks


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


def graph(n: int = 50, seed: int = 0):
    nodes = fake_node_table(n=n, seed=seed)
    features = fake_block("own", n=n, width=3, seed=seed)
    features.columns = [C.NODE_ID, "own__f00", "own__f01", "own__f02"]
    return nodes, features, fake_edges(n, 120, seed)


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


def test_label_like_parameter_names():
    for name in ("labels", "y", "y_train", "is_labelled", "target", "train_targets"):
        assert label_like(name), name
    for name in ("features", "edges", "edges_by_relation", "fit_mask", "config"):
        assert not label_like(name), name
