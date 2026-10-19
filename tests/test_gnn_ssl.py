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
