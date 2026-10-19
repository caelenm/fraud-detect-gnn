"""Grouped split, column types and the audit, on synthetic node tables.

All rows come from tests/synthetic.py; nothing is real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import fake_dataset
from synthetic import fake_node_table

from fraud_detect import columns as C
from fraud_detect.cli import main
from fraud_detect.config import get_paths
from fraud_detect.data import audit, split
from fraud_detect.features import own

CFG = {"n_folds": 5, "test_folds": 2, "cv_folds": 5}


def test_no_group_crosses_train_test_or_folds():
    nodes = split.split_nodes(fake_node_table(), CFG, seed=3)
    labelled = nodes[nodes[C.IS_LABELLED]]
    assert labelled.groupby(C.GROUP_ID)[C.SPLIT].nunique().max() == 1
    train = nodes[nodes[C.SPLIT] == C.TRAIN]
    assert train.groupby(C.GROUP_ID)[C.CV_FOLD].nunique().max() == 1
    assert nodes.loc[nodes[C.SPLIT] != C.TRAIN, C.CV_FOLD].isna().all()
    assert set(train[C.CV_FOLD].astype(int)) == set(range(5))


def test_unlabelled_nodes_stay_out_of_every_split():
    nodes = split.split_nodes(fake_node_table(unlabelled=30), CFG, seed=3)
    unlabelled = nodes[~nodes[C.IS_LABELLED]]
    assert (unlabelled[C.SPLIT] == C.UNLABELLED).all() and len(unlabelled) == 30
    assert unlabelled[C.CV_FOLD].isna().all()


def test_split_is_stratified_and_about_60_40():
    nodes = split.split_nodes(fake_node_table(n=2000), CFG, seed=3)
    labelled = nodes[nodes[C.IS_LABELLED]]
    rates = labelled.groupby(C.SPLIT)[C.LABEL].mean()
    overall = labelled[C.LABEL].mean()
    assert abs(rates[C.TRAIN] - overall) < 0.02 and abs(rates[C.TEST] - overall) < 0.02
    share_test = (labelled[C.SPLIT] == C.TEST).mean()
    assert 0.35 < share_test < 0.45


def test_split_is_deterministic_for_a_seed():
    a = split.split_nodes(fake_node_table(), CFG, seed=3)
    b = split.split_nodes(fake_node_table(), CFG, seed=3)
    c = split.split_nodes(fake_node_table(), CFG, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a[C.SPLIT].equals(c[C.SPLIT])


def test_duplicate_groups_are_kept_together():
    """Amazon-style: identical feature rows form a group and must not straddle."""
    nodes = fake_node_table(n=300, group_size=1)
    # Plant 20 sets of 3 identical rows: give them one group each.
    for k in range(20):
        nodes.loc[[3 * k, 3 * k + 1, 3 * k + 2], C.GROUP_ID] = 10_000 + k
    out = split.split_nodes(nodes, CFG, seed=0)
    planted = out[out[C.GROUP_ID] >= 10_000]
    assert planted.groupby(C.GROUP_ID)[C.SPLIT].nunique().max() == 1
    assert (
        planted[planted[C.SPLIT] == C.TRAIN]
        .groupby(C.GROUP_ID)[C.CV_FOLD]
        .nunique()
        .max()
        == 1
    )


def test_check_split_catches_a_group_on_both_sides():
    nodes = split.split_nodes(fake_node_table(), CFG, seed=3)
    train_row = nodes.index[nodes[C.SPLIT] == C.TRAIN][0]
    test_row = nodes.index[nodes[C.SPLIT] == C.TEST][0]
    nodes.loc[test_row, C.GROUP_ID] = nodes.loc[train_row, C.GROUP_ID]
    with pytest.raises(split.SplitError, match="groups_in_both_train_and_test"):
        split.check_split(nodes)


def test_saved_ids_must_match_the_node_table():
    nodes = split.split_nodes(fake_node_table(), CFG, seed=3)
    train, test = split.ids_table(nodes, C.TRAIN), split.ids_table(nodes, C.TEST)
    split.check_against_saved_ids(nodes, train, test)
    with pytest.raises(split.SplitError, match="rerun the split"):
        split.check_against_saved_ids(nodes, train.iloc[1:], test)


# ---- Column types ----------------------------------------------------------
def test_column_type_rules():
    assert own.column_type(np.array([0.0, 1.0, 1.0])) == own.BINARY
    assert own.column_type(np.array([0.2, 0.7])) == own.BINARY  # 2 values, any
    assert own.column_type(np.array([0.0, 1.0, 5.0, 2.0])) == own.ORDINAL
    assert own.column_type(np.array([-1.0, 0.0, 3.0])) == own.ORDINAL
    assert own.column_type(np.array([0.5, 1.0, 2.0])) == own.CONTINUOUS
    assert own.column_type(np.array([4.0, 4.0])) == own.CONSTANT


def test_column_spec_uses_training_nodes_only():
    features = np.array(
        [[0.0, 1.0], [1.0, 2.0], [0.0, 3.0],  # train: col 0 binary, col 1 ordinal
         [7.5, 3.5], [2.0, 9.0]]  # test values would change both types
    )  # fmt: skip
    block = own.own_block(features)
    assert list(block.columns) == [C.NODE_ID, "own__f00", "own__f01"]
    spec = own.column_spec(block, train_ids=np.array([0, 1, 2]))
    assert spec["columns"]["own__f00"]["type"] == own.BINARY
    assert spec["columns"]["own__f01"]["type"] == own.ORDINAL
    full = own.column_spec(block, train_ids=np.arange(5))
    assert full["columns"]["own__f00"]["type"] == own.CONTINUOUS  # test would change it


def test_feature_names_sort_in_column_order():
    names = list(own.own_block(np.zeros((1, 12))).columns[1:])
    assert names == sorted(names) and names[10] == "own__f10"


# ---- Audit -------------------------------------------------------------------
def test_audit_flags_a_planted_shortcut_feature():
    nodes = split.split_nodes(fake_node_table(n=600), CFG, seed=1)
    rng = np.random.default_rng(0)
    label = nodes[C.LABEL].fillna(0).to_numpy(dtype=float)
    features = np.column_stack(
        [rng.normal(size=len(nodes)),  # noise
         label + rng.normal(scale=0.3, size=len(nodes)),  # planted shortcut
         -label + rng.normal(scale=0.3, size=len(nodes))]  # same, reversed
    )  # fmt: skip
    block = own.own_block(features)
    train_ids = nodes.loc[nodes[C.SPLIT] == C.TRAIN, C.NODE_ID].to_numpy()
    table = audit.feature_table(block, nodes, own.column_spec(block, train_ids), 0.85)
    flags = dict(zip(table["feature"], table["possible_shortcut"], strict=True))
    assert flags == {"own__f00": False, "own__f01": True, "own__f02": True}
    assert (table["auroc_train"] >= 0.5).all()  # oriented


def test_audit_counts_duplicates_and_homophily():
    nodes = split.split_nodes(fake_node_table(n=200), CFG, seed=1)
    features = np.arange(200 * 2, dtype=float).reshape(200, 2)
    train = nodes.index[nodes[C.SPLIT] == C.TRAIN]
    features[train[1]] = features[train[0]]  # one duplicate within train
    dups = audit.duplicate_counts(own.own_block(features), nodes)
    assert dups == {"within_train": 1, "within_test": 0,
                    "test_rows_identical_to_a_train_row": 0}  # fmt: skip

    pos = nodes.index[(nodes[C.SPLIT] == C.TRAIN) & (nodes[C.LABEL] == 1)][:3]
    neg = nodes.index[(nodes[C.SPLIT] == C.TRAIN) & (nodes[C.LABEL] == 0)][:1]
    test = nodes.index[nodes[C.SPLIT] == C.TEST][:1]
    # pos0-pos1 (same, both positive), pos0-neg0 (differ), pos2-test0 (ignored)
    edges = np.array([[pos[0], pos[0], pos[2]], [pos[1], neg[0], test[0]]])
    row = audit.relation_row("r", edges, nodes)
    assert row["n_train_train_edges"] == 2
    assert row["edge_homophily_train"] == pytest.approx(0.5)
    # Edges leaving a positive node: pos0->pos1, pos1->pos0, pos0->neg0.
    assert row["positive_homophily_train"] == pytest.approx(2 / 3)


def test_split_features_and_audit_stages_end_to_end(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=120)
    assert main(["--config", str(path), "--to", "audit"]) == 0
    p = get_paths(config)
    nodes = pd.read_parquet(p.nodes)
    train, test = pd.read_csv(p.train_ids), pd.read_csv(p.test_ids)
    assert list(train.columns) == [C.NODE_ID, C.LABEL, C.GROUP_ID]
    assert not set(train[C.NODE_ID]) & set(test[C.NODE_ID])
    assert len(train) + len(test) == int(nodes[C.IS_LABELLED].sum())
    folds = pd.read_csv(p.cv_folds)
    assert folds[C.NODE_ID].tolist() == train[C.NODE_ID].tolist()
    block = pd.read_parquet(p.block_file(C.OWN_BLOCK))
    assert len(block) == len(nodes)  # unlabelled nodes keep their features
    assert not set(block.columns) & (set(C.FORBIDDEN_FEATURE_COLUMNS) - {C.NODE_ID})
    report = (p.reports_dir / "audit.md").read_text(encoding="utf-8")
    assert "Test rows identical to a training row: 0" in report
