"""The block probe (models/probe.py) reads training nodes only (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from synthetic import fake_feature_graph

from fraud_detect import columns as C
from fraud_detect.features.graph import build_nbr_block
from fraud_detect.models import probe


def setup(tmp_path, n: int = 240):
    nodes, own, edges = fake_feature_graph(n=n, width=3, seed=1)
    own.to_parquet(tmp_path / "own.parquet", index=False)
    build_nbr_block(own, edges).to_parquet(tmp_path / "nbr.parquet", index=False)
    train = nodes[nodes[C.SPLIT] == C.TRAIN]
    train_ids = pd.DataFrame(
        {C.NODE_ID: train[C.NODE_ID], C.LABEL: train[C.LABEL].astype(int),
         C.GROUP_ID: train[C.GROUP_ID]}
    )  # fmt: skip
    folds = pd.DataFrame({C.NODE_ID: train[C.NODE_ID], "fold": np.arange(len(train)) % 3})
    test_rows = nodes[C.SPLIT].to_numpy() != C.TRAIN
    return own, train_ids, folds, test_rows


def test_probe_scores_each_block_set_and_notes_missing_blocks(tmp_path):
    _, train_ids, folds, _ = setup(tmp_path)
    table = probe.probe(tmp_path, train_ids, folds, seed=0)
    by_set = table.set_index("blocks")
    assert by_set.loc["own", "n_features"] == 3
    assert by_set.loc["own+nbr", "n_features"] == 3 + 2 * (3 + 1)
    assert by_set.loc["own", "n_train"] == len(train_ids)
    assert 0 <= by_set.loc["own+nbr", "cv_pr_auc_mean"] <= 1
    assert "missing" in by_set.loc["own+gnnssl", "note"]


def test_probe_covers_variant_sets(tmp_path):
    _, train_ids, folds, _ = setup(tmp_path)
    variants = {"x01": ["own__f01"]}
    sets = probe.block_sets(variants)
    assert sets["own_x01+nbr_x01"] == ["own_x01", "nbr_x01"]
    table = probe.probe(tmp_path, train_ids, folds, 0, sets, variants).set_index("blocks")
    assert table.loc["own_x01", "n_features"] == 2  # own without f01
    assert "missing" in table.loc["own_x01+nbr_x01", "note"]  # nbr_x01 not built


def test_probe_never_reads_test_or_unlabelled_rows(tmp_path):
    own, train_ids, folds, other_rows = setup(tmp_path)
    before = probe.probe(tmp_path, train_ids, folds, seed=0)
    garbage = own.copy()
    cols = [c for c in own.columns if c != C.NODE_ID]
    rng = np.random.default_rng(0)
    garbage.loc[other_rows, cols] = rng.normal(1e6, 1e5, (int(other_rows.sum()), 3))
    garbage.to_parquet(tmp_path / "own.parquet", index=False)
    pd.testing.assert_frame_equal(probe.probe(tmp_path, train_ids, folds, seed=0), before)
