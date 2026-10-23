"""Model 2's `nbr` block and the neighbour_agg stage (synthetic data only)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from helpers import assert_label_blind, fake_dataset
from synthetic import fake_feature_graph

from fraud_detect import columns as C
from fraud_detect.cli import main
from fraud_detect.config import (
    DEFAULT_CONFIG,
    ConfigError,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.features import blocks
from fraud_detect.features import graph as graph_module
from fraud_detect.features.graph import build_nbr_block
from fraud_detect.models import probe
from fraud_detect.models.df_analyze import df_analyze_names


def five_nodes() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Two features on five nodes. Relation aaa: 0-1, 0-2, 2-3, node 4
    isolated. Relation bbb: 0-4 only."""
    own = pd.DataFrame(
        {
            C.NODE_ID: np.arange(5),
            "own__f00": [1.0, 2.0, 3.0, 4.0, 5.0],
            "own__f01": [10.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    edges = {
        "aaa": np.array([[0, 0, 2], [1, 2, 3]], dtype=np.int32),
        "bbb": np.array([[0], [4]], dtype=np.int32),
    }
    return own, edges


def test_hand_checked_means_and_degrees():
    own, edges = five_nodes()
    block = build_nbr_block(own, edges)
    assert list(block.columns) == [
        C.NODE_ID,
        "nbr__aaa_mean_f00", "nbr__aaa_mean_f01", "nbr__aaa_log1p_degree",
        "nbr__bbb_mean_f00", "nbr__bbb_mean_f01", "nbr__bbb_log1p_degree",
    ]  # fmt: skip
    b = block.set_index(C.NODE_ID)
    # Node 0's aaa neighbours are 1 and 2; node 2's are 0 and 3; node 4 has none.
    assert b["nbr__aaa_mean_f00"].tolist() == [2.5, 1.0, 2.5, 3.0, 0.0]
    assert b["nbr__aaa_mean_f01"].tolist() == [0.0, 10.0, 5.0, 0.0, 0.0]
    assert np.allclose(b["nbr__aaa_log1p_degree"], np.log1p([2, 1, 2, 1, 0]))
    assert b["nbr__bbb_mean_f00"].tolist() == [5.0, 0.0, 0.0, 0.0, 1.0]
    assert np.allclose(b["nbr__bbb_log1p_degree"], np.log1p([1, 0, 0, 0, 1]))


def test_row_order_of_the_own_block_does_not_matter():
    own, edges = five_nodes()
    shuffled = own.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(build_nbr_block(shuffled, edges),
                                  build_nbr_block(own, edges))  # fmt: skip


def test_bad_inputs_fail():
    own, edges = five_nodes()
    with pytest.raises(ValueError, match="outside"):
        build_nbr_block(own, {"aaa": np.array([[0], [5]])})
    with pytest.raises(ValueError, match="one row per node"):
        build_nbr_block(own.iloc[1:], edges)
    with pytest.raises(ValueError, match="own__"):
        build_nbr_block(own.assign(extra=1.0), edges)


def test_nbr_block_is_label_blind_valid_and_complete():
    nodes, own, edges = fake_feature_graph(n=80, width=3)
    assert_label_blind(build_nbr_block, nodes, own, edges)
    block = build_nbr_block(own, edges)
    blocks.validate_block(C.NBR_BLOCK, block)  # prefixed, numeric, no nulls
    assert len(block) == len(own) and block[C.NODE_ID].is_unique
    assert block.shape[1] == 1 + 2 * (3 + 1)


def test_rewiring_keeps_degrees_roughly_and_changes_partners():
    _, _, edges = fake_feature_graph(n=300, width=2, seed=3)
    e = edges["aaa"]
    r = graph_module.rewire_edges(e, 300, seed=1)
    assert np.array_equal(r, graph_module.rewire_edges(e, 300, seed=1))  # seeded
    assert (r[0] < r[1]).all() and r.dtype == np.int32
    assert r.shape[1] >= 0.95 * e.shape[1]  # few lost to self-loops / repeats
    deg = np.bincount(e.ravel(), minlength=300)
    deg_r = np.bincount(r.ravel(), minlength=300)
    assert np.abs(deg - deg_r).sum() <= 2 * (e.shape[1] - r.shape[1])
    shared = {tuple(p) for p in e.T} & {tuple(p) for p in r.T}
    assert len(shared) < 0.1 * e.shape[1]


def test_single_feature_check_uses_training_rows_only():
    own = pd.DataFrame({C.NODE_ID: range(6), "own__f00": [0, 1, 5, 6, 100, -100.0]})
    train = pd.DataFrame({C.NODE_ID: [0, 1, 2, 3], C.LABEL: [0, 0, 1, 1]})
    out = probe.single_feature_check(own, train, ["own__f00"]).iloc[0]
    assert out["auroc"] == 1.0 and out["best_threshold_balanced_accuracy"] == 1.0
    assert out["n_train"] == 4 and out["direction"] == "higher = fraud"


def test_graph_block_names_reach_df_analyze_without_collisions():
    # The real column names of both datasets (Amazon 25 features, YelpChi 32).
    for relations, width in ((("upu", "usu", "uvu"), 25), (("rur", "rtr", "rsr"), 32)):
        names = [f"own__f{j:02d}" for j in range(width)]
        for r in relations:
            names += [f"nbr__{r}_mean_f{j:02d}" for j in range(width)]
            names += [f"nbr__{r}_log1p_degree"]
            names += [f"gnnssl__{r}_pc{k:02d}" for k in range(1, 9)]
        renamed = df_analyze_names(names)  # raises on a collision
        assert len(set(renamed.values())) == len(names)
        r = relations[0]
        assert renamed[f"nbr__{r}_mean_f00"] == f"nbr_{r}_mean_f00"
        assert renamed[f"gnnssl__{r}_pc01"] == f"gnnssl_{r}_pc01"


def with_variant(path, config) -> dict:
    """The fake dataset's config plus a variant x01 (without own__f01) and
    feature sets using it, written back to `path`."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["feature_variants"]["x01"] = ["own__f01"]
    raw["feature_sets"]["t1_x01"] = ["own_x01"]
    raw["feature_sets"]["t2_x01"] = ["own_x01", "nbr_x01"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return {**config, "feature_variants": raw["feature_variants"],
            "feature_sets": raw["feature_sets"]}  # fmt: skip


def test_a_variant_leaves_its_columns_out_of_own_and_graph_inputs(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=120)
    config = with_variant(path, config)
    args = ["--config", str(path), "--to", "df_analyze_input", "--set"]
    assert main([*args, "feature_set=m2_own_nbr"]) == 0
    assert main([*args, "feature_set=t2_x01"]) == 0
    p = get_paths({**config, "feature_set": "t2_x01"})
    full = pd.read_parquet(p.block_file("nbr"))
    reduced = pd.read_parquet(p.block_file("nbr_x01"))
    assert "nbr_x01__aaa_mean_f01" not in reduced.columns
    assert reduced.shape[1] == full.shape[1] - 2  # one mean per relation fewer
    assert np.allclose(reduced["nbr_x01__aaa_mean_f00"], full["nbr__aaa_mean_f00"])
    train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    assert "own_f01" not in train.columns and "own_f00" in train.columns
    assert not [c for c in train.columns if c.endswith("_f01")]
    info = json.loads((p.df_analyze_input_dir / "feature_set.json").read_text())
    assert info["blocks"] == ["own_x01", "nbr_x01"]
    # own_x01 alone: own's columns and types, minus the variant's.
    assert main([*args, "feature_set=t1_x01"]) == 0
    t1 = get_paths({**config, "feature_set": "t1_x01"})
    assert pd.read_parquet(t1.df_analyze_input_dir / "test.parquet").shape[1] == 5


def test_a_keep_variant_uses_only_its_columns_everywhere(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=120)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["feature_variants"]["k01"] = {"keep": ["own__f01", "own__f03"]}
    raw["feature_sets"]["t2_k01"] = ["own_k01", "nbr_k01"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    args = ["--config", str(path), "--to", "df_analyze_input", "--set"]
    assert main([*args, "feature_set=t2_k01"]) == 0
    p = get_paths({**config, "feature_sets": raw["feature_sets"],
                   "feature_variants": raw["feature_variants"],
                   "feature_set": "t2_k01"})  # fmt: skip
    nbr = pd.read_parquet(p.block_file("nbr_k01"))
    assert list(nbr.columns)[1:4] == [
        "nbr_k01__aaa_mean_f01", "nbr_k01__aaa_mean_f03", "nbr_k01__aaa_log1p_degree",
    ]  # fmt: skip
    train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    own_cols = [c for c in train.columns if c.startswith("own_")]
    assert own_cols == ["own_f01", "own_f03"]
    assert train.shape[1] == 2 + 2 * (2 + 1) + 1  # own, nbr per relation, target


def test_keep_and_leave_out_variants_list_the_same_columns():
    own, _ = five_nodes()
    keep = {"k": {"keep": ["own__f00"]}, "x": ["own__f01"]}
    assert blocks.excluded_columns("own_k", keep, list(own.columns)) == ["own__f01"]
    assert blocks.excluded_columns("own_x", keep) == ["own__f01"]
    pd.testing.assert_frame_equal(blocks.own_view(own, "nbr_k", keep),
                                  blocks.own_view(own, "nbr_x", keep))  # fmt: skip
    with pytest.raises(blocks.BlockError, match="are needed"):
        blocks.excluded_columns("own_k", keep)
    with pytest.raises(blocks.BlockError, match="lacks"):
        blocks.own_view(own, "own_k", {"k": {"keep": ["own__f09"]}})
    config = load_config(DEFAULT_CONFIG)
    for bad in ({"keep": []}, {"drop": ["own__f00"]}, "own__f00"):
        config["feature_variants"]["bad"] = bad
        with pytest.raises(ConfigError, match="feature_variants.bad"):
            with_dataset(config, "amazon")


def test_feature_ranking_reads_training_rows_only_and_finds_the_signal():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    own = pd.DataFrame(
        {
            C.NODE_ID: np.arange(n),
            "own__f00": y + rng.normal(0, 0.3, n),  # the signal
            "own__f01": rng.normal(size=n),
            "own__f02": rng.normal(size=n),
        }
    )
    train = pd.DataFrame({C.NODE_ID: np.arange(300), C.LABEL: y[:300]})
    folds = pd.DataFrame({C.NODE_ID: np.arange(300), "fold": np.arange(300) % 3})
    table = probe.feature_ranking(own, train, folds, seed=0)
    assert (
        table.iloc[0]["column"] == "own__f00" and table.iloc[0]["rank_permutation"] == 1
    )
    assert table["univariate_auroc"].iloc[0] > 0.9
    garbage = own.copy()
    garbage.loc[300:, ["own__f00", "own__f01", "own__f02"]] = 1e6  # test rows
    pd.testing.assert_frame_equal(table, probe.feature_ranking(garbage, train, folds, 0))


def test_unknown_blocks_and_variants_are_refused():
    with pytest.raises(blocks.BlockError, match="Unknown block"):
        blocks.parse_block("xyz_no36")
    assert blocks.parse_block("gnnssl_no36") == ("gnnssl", "no36")
    with pytest.raises(blocks.BlockError, match="feature_variants"):
        blocks.excluded_columns("own_zz", {"no36": ["own__f13"]})
    own, _ = five_nodes()
    with pytest.raises(blocks.BlockError, match="lacks"):
        blocks.own_view(own, "own_no36", {"no36": ["own__f13"]})
    config = load_config(DEFAULT_CONFIG)
    config["feature_sets"]["bad"] = ["own_nope"]
    with pytest.raises(ConfigError, match="feature_variants"):
        with_dataset(config, "amazon")


def test_neighbour_agg_stage_builds_the_block_only_when_used(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=120)
    assert main(["--config", str(path), "--to", "neighbour_agg"]) == 0
    p = get_paths(config)
    assert not p.block_file(C.NBR_BLOCK).exists()  # m1_own does not use it

    args = ["--config", str(path), "--to", "df_analyze_input",
            "--set", "feature_set=m2_own_nbr"]  # fmt: skip
    assert main(args) == 0
    block = pd.read_parquet(p.block_file(C.NBR_BLOCK))
    spec = json.loads(p.block_spec(C.NBR_BLOCK).read_text(encoding="utf-8"))
    n_own = 5  # the fake dataset's features
    assert block.shape == (120, 1 + 2 * (n_own + 1))  # relations aaa and bbb
    assert set(spec["columns"]) == set(block.columns) - {C.NODE_ID}
    m2 = get_paths({**config, "feature_set": "m2_own_nbr"})
    train = pd.read_parquet(m2.df_analyze_input_dir / "train.parquet")
    assert train.shape[1] == n_own + 2 * (n_own + 1) + 1  # + target
    assert "nbr_aaa_mean_f00" in train.columns
