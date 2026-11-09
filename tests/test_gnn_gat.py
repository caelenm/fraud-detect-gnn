"""The GAT rungs (graph/supervised.py, `model: gat`; docs/GAT_DEEP_PLAN.md).

Synthetic data only, on the CPU. Checks the attention layer against PyG's
GATConv, the sparse aggregation's gradients against plain autograd,
isolated nodes, determinism, the config, and E10 and N10 end to end through
gnn_classify. The leakage rules are those of every GNN rung (cross_fit,
tests/test_gnn_supervised.py). Skipped without the optional gnn extra.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from helpers import fake_dataset
from synthetic import fake_edges

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from fraud_detect.cli import main  # noqa: E402
from fraud_detect.config import (  # noqa: E402
    DEFAULT_CONFIG,
    ConfigError,
    check_feature_sets,
    get_paths,
    load_config,
)
from fraud_detect.graph import supervised as sup  # noqa: E402
from fraud_detect.pipeline import gnn_model_label  # noqa: E402

pytestmark = pytest.mark.gnn

SMALL = [
    "--set", "gnn_classify.hidden=[8]", "--set", "gnn_classify.dropout=[0.0]",
    "--set", "gnn_classify.lr=[0.02, 0.05]", "--set", "gnn_classify.max_epochs=8",
    "--set", "gnn_classify.patience=3", "--set", "gnn_classify.inner_folds=3",
    "--set", "gnn_classify.seeds=2",
]  # fmt: skip
GAT_CFG = sup.TrainConfig(hidden=(8,), dropout=(0.0,), lr=(0.02,), max_epochs=10,
                          patience=4, inner_folds=3, seeds=1, model="gat",
                          heads=2)  # fmt: skip
SETTING = sup.Setting(hidden=8, dropout=0.0, lr=0.02)


def toy_problem(n: int = 120, seed: int = 0):
    """Invented features and two relations; the first 3/4 of the nodes are
    labelled training rows in 5 folds (groups of 2), the rest neither."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    edges = {"aaa": fake_edges(n, 3 * n, seed), "bbb": fake_edges(n, 2 * n, seed + 1)}
    train_nodes = np.arange(3 * n // 4)
    noise = rng.normal(scale=0.5, size=len(train_nodes))
    y = (x[train_nodes, 0] + noise > 0.6).astype(int)
    groups = train_nodes // 2
    return x, edges, train_nodes, y, groups % 5, groups


# --------------------------------------------------------------------------
# The attention layer
# --------------------------------------------------------------------------
def test_gat_layer_is_pyg_gatconv_per_relation_plus_the_root_term():
    from torch_geometric.nn import GATConv  # noqa: PLC0415

    x, edges, *_ = toy_problem(n=40)
    g = sup.graph_input(x, edges, "mean", model="gat")
    torch.manual_seed(0)
    layer = sup.RelGAT(4, 8, n_relations=2, heads=2).eval()
    ours = layer(g.x, g.edge_index).detach()
    expected = layer.root(g.x).detach()
    for j, r in enumerate(edges):
        e = torch.as_tensor(np.asarray(edges[r], dtype=np.int64))
        index = torch.cat([e, e.flip(0)], dim=1)  # source row 0, target row 1
        conv = GATConv(4, 4, heads=2, concat=True, add_self_loops=False, bias=False)
        with torch.no_grad():
            conv.lin.weight.copy_(layer.rel[j].weight)
            conv.att_src.copy_(layer.att_src[j].unsqueeze(0))
            conv.att_dst.copy_(layer.att_dst[j].unsqueeze(0))
            expected = expected + conv(g.x, index)
    assert torch.allclose(ours, expected, atol=1e-5)


def test_sparse_aggregation_gradients_match_plain_autograd():
    torch.manual_seed(1)
    n, heads, channels = 12, 3, 2
    pairs = torch.randperm(n * n)[:50]
    edges = sup.RelationEdges.build((pairs % n).numpy(), (pairs // n).numpy(), n)
    alpha = torch.rand(50, heads, dtype=torch.float64, requires_grad=True)
    v = torch.randn(n, heads, channels, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: sup.attention_aggregate(a, b, edges), (alpha, v)
    )
    # The same values and gradients as the materialised (E, H, C) messages.
    weights = torch.randn(n, heads, channels, dtype=torch.float64)
    out = sup.attention_aggregate(alpha, v, edges)
    (out * weights).sum().backward()
    a2, v2 = alpha.detach().clone().requires_grad_(), v.detach().clone().requires_grad_()
    plain = torch.zeros(n, heads, channels, dtype=torch.float64).index_add(
        0, edges.dst, a2.unsqueeze(-1) * v2[edges.src]
    )
    (plain * weights).sum().backward()
    assert torch.allclose(out, plain)
    assert torch.allclose(alpha.grad, a2.grad) and torch.allclose(v.grad, v2.grad)


def test_isolated_nodes_get_only_the_root_term():
    x, edges, *_ = toy_problem(n=40)
    for r in edges:  # node 0 has no neighbour in any relation
        edges[r] = edges[r][:, edges[r].min(axis=0) > 0]
    g = sup.graph_input(x, edges, "mean", model="gat")
    torch.manual_seed(0)
    layer = sup.RelGAT(4, 8, n_relations=2, heads=4).eval()
    out = layer(g.x, g.edge_index).detach()
    assert torch.allclose(out[0], layer.root(g.x[:1]).detach()[0])
    assert not torch.allclose(out[1:], layer.root(g.x[1:]).detach())


def test_attention_weights_sum_to_one_over_incoming_edges():
    from torch_geometric.utils import softmax  # noqa: PLC0415

    undirected = fake_edges(30, 60, 2)
    src, dst = sup.directed_edges(undirected)
    n_dir = 2 * undirected.shape[1]
    assert (np.diff(dst) >= 0).all() and len(src) == n_dir  # both directions, by dst
    assert set(zip(src, dst, strict=True)) == {
        *zip(undirected[0], undirected[1], strict=True),
        *zip(undirected[1], undirected[0], strict=True),
    }
    alpha = softmax(torch.randn(n_dir, 2), torch.as_tensor(dst), num_nodes=30)
    sums = torch.zeros(30, 2).index_add(0, torch.as_tensor(dst), alpha)
    has = torch.as_tensor(np.isin(np.arange(30), dst))
    assert torch.allclose(sums[has], torch.ones(int(has.sum()), 2), atol=1e-6)


# --------------------------------------------------------------------------
# Fits and options
# --------------------------------------------------------------------------
def test_gat_fit_is_deterministic_on_the_cpu_and_seeded():
    x, edges, train_nodes, y, folds, groups = toy_problem()
    g = sup.graph_input(x, edges, "mean", model="gat")
    assert g.adjs == [] and len(g.edge_index) == 2  # no A_hat built for GAT
    first = sup.cross_fit(g, train_nodes, y, folds, groups, SETTING, GAT_CFG, seed=3)
    again = sup.cross_fit(g, train_nodes, y, folds, groups, SETTING, GAT_CFG, seed=3)
    other = sup.cross_fit(g, train_nodes, y, folds, groups, SETTING, GAT_CFG, seed=4)
    assert np.array_equal(first.oof, again.oof)
    assert np.array_equal(first.all_nodes, again.all_nodes)
    assert not np.array_equal(first.oof, other.oof)
    assert not np.isnan(first.all_nodes).any()


def test_graph_input_and_network_must_agree_on_the_model():
    x, edges, train_nodes, y, folds, groups = toy_problem()
    sage_input = sup.graph_input(x, edges, "mean")
    with pytest.raises(ValueError, match="graph input built for 'sage'"):
        sup.cross_fit(sage_input, train_nodes, y, folds, groups, SETTING, GAT_CFG, 0)
    gat_input = sup.graph_input(x, edges, "mean", model="gat")
    sage_cfg = sup.TrainConfig(hidden=(8,), dropout=(0.0,), lr=(0.02,), max_epochs=2)
    with pytest.raises(ValueError, match="graph input built for 'gat'"):
        sup.cross_fit(gat_input, train_nodes, y, folds, groups, SETTING, sage_cfg, 0)


def test_gat_takes_the_shared_size_options():
    x, edges, *_ = toy_problem(n=40)
    g = sup.graph_input(x, edges, "mean", model="gat")
    cfg = sup.TrainConfig(hidden=(8,), layers=3, model="gat", heads=2, residual=True,
                          layer_norm=True, jk=True)  # fmt: skip
    torch.manual_seed(0)
    model = sup.build_model(g, sup.Setting(8, 0.0, 0.01), cfg).eval()
    assert all(isinstance(c, sup.RelGAT) for c in model.convs) and len(model.convs) == 3
    assert model.head.in_features == 24 and model(g).shape == (40,)


def test_train_config_validates_heads():
    with pytest.raises(ValueError, match="model"):
        sup.TrainConfig.from_config({"model": "gcn"})
    with pytest.raises(ValueError, match="heads"):
        sup.TrainConfig.from_config({"model": "gat", "heads": 3, "hidden": [64]})
    with pytest.raises(ValueError, match="heads"):
        sup.TrainConfig.from_config({"model": "gat", "heads": 0})
    cfg = sup.TrainConfig.from_config({"model": "gat", "heads": 4, "hidden": [64, 128]})
    assert cfg.heads == 4 and cfg.hidden == (64, 128)
    # heads matters only for GAT: other models accept any value.
    assert sup.TrainConfig.from_config({"heads": 3, "hidden": [64]}).model == "sage"


# --------------------------------------------------------------------------
# Config, labels and the stage
# --------------------------------------------------------------------------
def test_gat_rungs_are_extra_and_not_gnn_oof_candidates():
    config = load_config(DEFAULT_CONFIG)
    models = config["gnn_models"]
    new = {"pilot_g10_gat", "pilot_g10n3_gat_rw"}
    assert new <= set(models) and new.isdisjoint(config["gnn_oof"]["candidates"])
    for old in ("pilot_g5_sage", "pilot_g5g_rgcn", "pilot_g6_mlp", "pilot_g5n3_sage_rw"):
        assert "train" not in models[old]
    assert models["pilot_g10_gat"]["train"] == {"model": "gat", "heads": 4}
    assert models["pilot_g10n3_gat_rw"]["train"] == models["pilot_g10_gat"]["train"]
    assert gnn_model_label(models["pilot_g10_gat"]) == "GAT"
    assert gnn_model_label(models["pilot_g10n3_gat_rw"]) == "GAT (rewired graph)"
    assert gnn_model_label(models["pilot_g5_sage"]) == "GraphSAGE"


def test_config_rejects_an_unknown_train_key():
    config = load_config(DEFAULT_CONFIG)
    config["gnn_models"]["pilot_g10_gat"] = {
        "graph": "real", "train": {"model": "gat", "attention": 4}
    }  # fmt: skip
    with pytest.raises(ConfigError):
        check_feature_sets(config)


def run_rung(path, feature_set: str, *extra: str) -> int:
    return main(["--config", str(path), "--set", f"feature_set={feature_set}", *SMALL,
                 *extra])  # fmt: skip


def test_gat_and_rewired_gat_rungs_run_end_to_end(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=300)
    assert run_rung(path, "pilot_g10n3_gat_rw") == 2  # needs E10's setting first
    assert run_rung(path, "pilot_g10_gat", "--force") == 0
    assert run_rung(path, "pilot_g10n3_gat_rw", "--force") == 0
    read = {}
    for fs in ("pilot_g10_gat", "pilot_g10n3_gat_rw"):
        reports = get_paths({**config, "feature_set": fs}).model_reports_dir
        read[fs] = json.loads((reports / "model_a.json").read_text("utf-8"))
        assert 0.0 <= read[fs]["model_a"]["pr_auc"] <= 1.0
    gat, rw = read["pilot_g10_gat"], read["pilot_g10n3_gat_rw"]
    assert gat["model_a"]["model"] == "GAT" and gat["gnn"]["config"]["model"] == "gat"
    assert gat["gnn"]["config"]["heads"] == 4
    assert gat["gnn"]["relations"] == ["aaa", "bbb"]
    assert rw["model_a"]["model"] == "GAT (rewired graph)"
    assert rw["gnn"]["setting"] == gat["gnn"]["setting"]
    assert rw["gnn"]["rewired_graph"].endswith("seed555")
