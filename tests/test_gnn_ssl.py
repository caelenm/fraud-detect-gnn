"""Model 5's gnnssl block (graph/ssl.py) and the gnn_embed stage.

Synthetic data only, on the CPU: CUDA's sparse products are not
bit-reproducible, so the determinism and label-blindness tests force the CPU.
Skipped when the optional gnn extra (torch, torch_geometric) is missing.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml
from helpers import assert_label_blind, fake_dataset
from synthetic import fake_feature_graph

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from fraud_detect import columns as C  # noqa: E402
from fraud_detect.cli import main  # noqa: E402
from fraud_detect.config import get_paths  # noqa: E402
from fraud_detect.features import blocks  # noqa: E402
from fraud_detect.graph import ssl  # noqa: E402
from fraud_detect.pipeline import split_masks  # noqa: E402

pytestmark = pytest.mark.gnn

CFG = ssl.SSLConfig(
    hidden=8, out_dim=8, max_epochs=15, patience=5, pca_components=3, device="cpu", seed=3
)
SMALL = [
    "--set", "gnn_embed.device=cpu", "--set", "gnn_embed.max_epochs=12",
    "--set", "gnn_embed.hidden=8", "--set", "gnn_embed.out_dim=8",
    "--set", "gnn_embed.pca_components=3",
]  # fmt: skip


def small_graph(seed: int = 0):
    nodes, own, edges = fake_feature_graph(n=70, width=4, seed=seed)
    fit_mask, loss_mask = split_masks(nodes)
    return nodes, own, edges, fit_mask, loss_mask


def test_embeddings_are_label_blind():
    nodes, own, edges, fit_mask, loss_mask = small_graph()

    def builder(features: pd.DataFrame, edges: dict) -> pd.DataFrame:
        return ssl.gnnssl_block(features, edges, fit_mask, loss_mask, CFG).block

    assert_label_blind(builder, nodes, own, edges)


def test_same_seed_same_embeddings_on_the_cpu():
    _, own, edges, fit_mask, loss_mask = small_graph()
    first = ssl.gnnssl_block(own, edges, fit_mask, loss_mask, CFG)
    second = ssl.gnnssl_block(own, edges, fit_mask, loss_mask, CFG)
    for relation in edges:
        assert np.array_equal(first.embeddings[relation], second.embeddings[relation])
    pd.testing.assert_frame_equal(first.block, second.block)
    other = ssl.gnnssl_block(own, edges, fit_mask, loss_mask, replace(CFG, seed=4))
    assert not np.array_equal(first.embeddings["aaa"], other.embeddings["aaa"])


def test_block_shape_names_and_training_log():
    _, own, edges, fit_mask, loss_mask = small_graph()
    result = ssl.gnnssl_block(own, edges, fit_mask, loss_mask, CFG)
    block = result.block
    expected = [f"gnnssl__{r}_pc{k:02d}" for r in ("aaa", "bbb") for k in (1, 2, 3)]
    assert list(block.columns) == [C.NODE_ID, *expected]
    assert len(block) == len(own) and not block.isna().any().any()
    blocks.validate_block(C.GNNSSL_BLOCK, block)
    for relation, log in result.logs.items():
        assert result.embeddings[relation].shape == (len(own), CFG.out_dim)
        assert 1 <= log["best_epoch"] <= log["epochs_run"] <= CFG.max_epochs
        assert len(log["losses"]) == log["epochs_run"] and log["device"] == "cpu"
        assert 0 < log["explained_variance_total"] <= 1


def test_scaler_and_pca_are_fit_on_training_nodes_only():
    _, own, _, fit_mask, _ = small_graph()
    rng = np.random.default_rng(0)
    x = own.drop(columns=C.NODE_ID).to_numpy()
    changed = x.copy()
    changed[~fit_mask] = rng.normal(50, 9, size=changed[~fit_mask].shape)
    a, b = ssl.fit_scaler(x, fit_mask), ssl.fit_scaler(changed, fit_mask)
    assert np.array_equal(a.mean_, b.mean_) and np.array_equal(a.scale_, b.scale_)

    z = rng.normal(size=(len(fit_mask), 8))
    z_changed = z.copy()
    z_changed[~fit_mask] = rng.normal(5, 3, size=z_changed[~fit_mask].shape)
    p, q = ssl.fit_pca(z, fit_mask, 3, 0), ssl.fit_pca(z_changed, fit_mask, 3, 0)
    assert np.array_equal(p.components_, q.components_)
    assert np.array_equal(p.mean_, q.mean_)
    # ...but the transform is applied to every node, test nodes included.
    assert not np.allclose(p.transform(z), q.transform(z_changed))


def test_masks_and_edge_sampling_rules():
    nodes, *_ = small_graph()
    fit_mask, loss_mask = split_masks(nodes)
    split = nodes.sort_values(C.NODE_ID)[C.SPLIT].to_numpy()
    assert (fit_mask == (split == C.TRAIN)).all()
    assert not loss_mask[split == C.TEST].any() and loss_mask[split == C.UNLABELLED].all()
    with pytest.raises(ValueError, match="inside loss_mask"):
        ssl.check_masks(len(fit_mask), fit_mask, fit_mask & False)

    edges = np.arange(20).reshape(2, 10)
    half = ssl.sample_edges(edges, 0.5, seed=1)
    assert half.shape == (2, 5) and np.array_equal(half, ssl.sample_edges(edges, 0.5, 1))
    cfg = ssl.SSLConfig(usu_edge_fraction=0.5)
    assert cfg.edge_fraction("usu") == 0.5 and cfg.edge_fraction("upu") == 1.0


def test_gnnrand_is_label_blind_deterministic_and_shaped_like_gnnssl():
    nodes, own, edges, fit_mask, _ = small_graph()

    def builder(features: pd.DataFrame, edges: dict) -> pd.DataFrame:
        return ssl.gnnrand_block(features, edges, fit_mask, CFG).block

    assert_label_blind(builder, nodes, own, edges)
    first = ssl.gnnrand_block(own, edges, fit_mask, CFG)
    pd.testing.assert_frame_equal(first.block, builder(own, edges))
    expected = [f"gnnrand__{r}_pc{k:02d}" for r in ("aaa", "bbb") for k in (1, 2, 3)]
    assert list(first.block.columns) == [C.NODE_ID, *expected]
    blocks.validate_block(C.GNNRAND_BLOCK, first.block)
    other = ssl.gnnrand_block(own, edges, fit_mask, replace(CFG, seed=4))
    assert not np.array_equal(first.embeddings["aaa"], other.embeddings["aaa"])
    assert not first.logs["aaa"]["trained"]


def test_gnnrand_is_where_dgi_training_starts():
    # With one epoch, DGI keeps the weights that produced its first loss: the
    # initial ones. gnnrand must give exactly those embeddings.
    _, own, edges, fit_mask, loss_mask = small_graph()
    start = ssl.gnnssl_block(own, edges, fit_mask, loss_mask, replace(CFG, max_epochs=1))
    rand = ssl.gnnrand_block(own, edges, fit_mask, CFG)
    trained = ssl.gnnssl_block(own, edges, fit_mask, loss_mask, CFG)
    for relation in edges:
        assert np.array_equal(start.embeddings[relation], rand.embeddings[relation])
        assert not np.array_equal(trained.embeddings[relation], rand.embeddings[relation])


def test_gnn_embed_stage_builds_a_gnnrand_block(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=120)
    args = ["--config", str(path), "--set", "feature_set=pilot_m5r_own_gnnrand", *SMALL]
    assert main([*args, "--to", "df_analyze_input"]) == 0
    p = get_paths({**config, "feature_set": "pilot_m5r_own_gnnrand"})
    block = pd.read_parquet(p.block_file(C.GNNRAND_BLOCK))
    assert block.shape == (120, 1 + 2 * 3)
    meta = json.loads((p.embed_dir("gnnrand") / "meta.json").read_text(encoding="utf-8"))
    assert p.embed_dir("gnnrand").name == "gnn_embed_gnnrand"
    assert meta["method"].startswith("untrained") and "corruption" not in meta
    assert np.load(p.raw_embedding("aaa", "gnnrand")).shape == (120, 8)
    assert not p.block_file(C.GNNSSL_BLOCK).exists()  # only the blocks it uses
    train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    assert "gnnrand_bbb_pc03" in train.columns
    # Rebuilt with the same settings: identical (no training, fixed seed).
    assert main([*args, "--only", "gnn_embed", "--force"]) == 0
    pd.testing.assert_frame_equal(pd.read_parquet(p.block_file(C.GNNRAND_BLOCK)), block)


def stage_args(path) -> list[str]:
    return ["--config", str(path), "--set", "feature_set=m5_own_gnnssl", *SMALL]


def test_gnn_embed_stage_writes_block_spec_meta_and_raw_embeddings(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=120)
    assert main([*stage_args(path), "--to", "df_analyze_input"]) == 0
    p = get_paths({**config, "feature_set": "m5_own_gnnssl"})
    block = pd.read_parquet(p.block_file(C.GNNSSL_BLOCK))
    spec = json.loads(p.block_spec(C.GNNSSL_BLOCK).read_text(encoding="utf-8"))
    meta = json.loads((p.gnn_embed_dir / "meta.json").read_text(encoding="utf-8"))
    assert block.shape == (120, 1 + 2 * 3)
    assert set(spec["columns"]) == set(block.columns) - {C.NODE_ID}
    assert set(meta["relations"]) == {"aaa", "bbb"} and meta["config"]["device"] == "cpu"
    assert meta["n_fit_nodes"] < meta["n_loss_nodes"] < meta["n_nodes"] == 120
    assert np.load(p.raw_embedding("aaa")).shape == (120, 8)
    train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    assert "gnnssl_bbb_pc03" in train.columns


def test_a_variant_block_never_sees_the_left_out_columns(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=120)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["feature_variants"]["x01"] = ["own__f01"]
    raw["feature_sets"]["t5_x01"] = ["own_x01", "gnnssl_x01"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    base = ["--config", str(path), *SMALL, "--to", "gnn_embed", "--set"]
    assert main([*base, "feature_set=m5_own_gnnssl"]) == 0
    assert main([*base, "feature_set=t5_x01"]) == 0
    p = get_paths({**config, "feature_sets": raw["feature_sets"],
                   "feature_set": "t5_x01"})  # fmt: skip
    meta = json.loads((p.embed_dir("gnnssl_x01") / "meta.json").read_text())
    assert meta["own_columns_left_out"] == ["own__f01"]
    assert (
        "own__f01" not in meta["own_columns_used"] and len(meta["own_columns_used"]) == 4
    )
    block = pd.read_parquet(p.block_file("gnnssl_x01"))
    assert list(block.columns)[1] == "gnnssl_x01__aaa_pc01"
    assert p.raw_embedding("aaa", "gnnssl_x01").parent.name == "gnn_embed_x01"
    # The all-features embeddings are separate and differ.
    full = np.load(p.raw_embedding("aaa"))
    assert not np.array_equal(full, np.load(p.raw_embedding("aaa", "gnnssl_x01")))


def test_gnn_embed_sizes_set_the_pca_size_or_keep_the_raw_embedding(tmp_path):
    from fraud_detect.config import (  # noqa: PLC0415
        DEFAULT_CONFIG,
        ConfigError,
        load_config,
        with_dataset,
    )
    from fraud_detect.pipeline import Context, gnn_embed_fingerprint  # noqa: PLC0415

    path, config = fake_dataset(tmp_path, n_nodes=120)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for v in ("x01", "x01k2", "x01raw"):
        raw["feature_variants"][v] = ["own__f01"]
    raw["gnn_embed_sizes"] = {"x01k2": 2, "x01raw": "raw"}
    raw["feature_sets"]["sizes"] = ["own_x01", "gnnssl_x01", "gnnssl_x01k2",
                                    "gnnrand_x01raw"]  # fmt: skip
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    args = ["--config", str(path), *SMALL, "--to", "df_analyze_input"]
    assert main([*args, "--set", "feature_set=sizes"]) == 0
    c = {**config, "feature_sets": raw["feature_sets"], "feature_set": "sizes",
         "feature_variants": raw["feature_variants"]}  # fmt: skip
    p = get_paths(c)
    default = pd.read_parquet(p.block_file("gnnssl_x01"))
    k2 = pd.read_parquet(p.block_file("gnnssl_x01k2"))
    rawb = pd.read_parquet(p.block_file("gnnrand_x01raw"))
    assert default.shape == (120, 1 + 2 * 3) and k2.shape == (120, 1 + 2 * 2)
    assert rawb.shape == (120, 1 + 2 * 8)  # out_dim 8 on both relations, no PCA
    assert list(rawb.columns)[1:3] == [
        "gnnrand_x01raw__aaa_z01",
        "gnnrand_x01raw__aaa_z02",
    ]
    z = np.load(p.raw_embedding("aaa", "gnnrand_x01raw"))
    assert np.array_equal(rawb.iloc[:, 1:9].to_numpy(), z.astype(np.float64))
    # Same input and seed: size 2 is the first 2 components of size 3's PCA.
    first_two = default.iloc[:, 1:3].to_numpy()
    np.testing.assert_allclose(k2.iloc[:, 1:3].to_numpy(), first_two, atol=1e-8)
    meta = json.loads((p.embed_dir("gnnssl_x01k2") / "meta.json").read_text())
    assert meta["pca_components_used"] == 2
    meta = json.loads((p.embed_dir("gnnrand_x01raw") / "meta.json").read_text())
    assert meta["pca_components_used"] == "raw"
    train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    assert train.shape[1] == 4 + 6 + 4 + 16 + 1  # own_x01, three blocks, target

    # A size entry changes only its own block's fingerprint.
    def fingerprint(cfg: dict, block: str) -> str:
        return gnn_embed_fingerprint(
            Context(config=cfg, paths=p, run_dir=tmp_path), block
        )

    without = {k: v for k, v in c.items() if k != "gnn_embed_sizes"}
    with_sizes = {**c, "gnn_embed_sizes": raw["gnn_embed_sizes"]}
    assert fingerprint(without, "gnnssl_x01") == fingerprint(with_sizes, "gnnssl_x01")
    assert fingerprint(with_sizes, "gnnssl_x01k2") != fingerprint(
        with_sizes, "gnnssl_x01"
    )

    bad = load_config(DEFAULT_CONFIG)
    for sizes in ({"nope": 4}, {"no19d16": 0}, {"no19d16": "big"}):
        bad["gnn_embed_sizes"] = sizes
        with pytest.raises(ConfigError, match="gnn_embed_sizes"):
            with_dataset(bad, "amazon")


def test_gnn_embed_resumes_after_an_interruption(tmp_path, monkeypatch):
    path, config = fake_dataset(tmp_path, n_nodes=120)
    assert main([*stage_args(path), "--to", "gnn_embed"]) == 0
    p = get_paths({**config, "feature_set": "m5_own_gnnssl"})
    reference = pd.read_parquet(p.block_file(C.GNNSSL_BLOCK))

    calls: list[str] = []
    real = ssl.embed_relation

    def interrupted_once(x, edges, loss_mask, cfg, relation):
        calls.append(relation)
        if calls == ["aaa", "bbb"]:
            raise KeyboardInterrupt  # stopped while training the second relation
        return real(x, edges, loss_mask, cfg, relation)

    monkeypatch.setattr(ssl, "embed_relation", interrupted_once)
    assert main([*stage_args(path), "--only", "gnn_embed", "--force"]) == 130
    assert main([*stage_args(path), "--resume"]) == 0
    assert calls == ["aaa", "bbb", "bbb"]  # aaa was kept, not trained again
    pd.testing.assert_frame_equal(
        pd.read_parquet(p.block_file(C.GNNSSL_BLOCK)), reference
    )
