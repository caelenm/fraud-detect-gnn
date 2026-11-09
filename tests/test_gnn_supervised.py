"""Supervised GNNs (graph/supervised.py): the GNN classifier rungs
(gnn_classify, Models 3-4) and Model 6's gnnsup block (gnn_oof).

Synthetic data only, on the CPU. Checks the layer against PyG, the sparse
product's gradient, determinism, and the leakage rules of AGENTS.md
invariant 1 and its documented exception: fold k's labels never reach fold
k's out-of-fold scores or any nested score for outer fold k, test labels
never reach a model, and the shared CV fits on nested scores. Skipped when
the optional gnn extra (torch, torch_geometric) is missing.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from helpers import fake_dataset
from synthetic import fake_edges

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from fraud_detect import columns as C  # noqa: E402
from fraud_detect.cli import main  # noqa: E402
from fraud_detect.config import REPO_ROOT, get_paths  # noqa: E402
from fraud_detect.graph import supervised as sup  # noqa: E402
from fraud_detect.models import gnn_outputs  # noqa: E402
from fraud_detect.pipeline import Context, nested_scores_file  # noqa: E402

pytestmark = pytest.mark.gnn

CFG = sup.TrainConfig(hidden=(8,), dropout=(0.0,), lr=(0.02,), max_epochs=12, patience=4,
                      inner_folds=3, seeds=2)  # fmt: skip
SETTING = sup.Setting(hidden=8, dropout=0.0, lr=0.02)
SMALL = [
    "--set", "gnn_classify.hidden=[8]", "--set", "gnn_classify.dropout=[0.0]",
    "--set", "gnn_classify.lr=[0.02, 0.05]", "--set", "gnn_classify.max_epochs=10",
    "--set", "gnn_classify.patience=3", "--set", "gnn_classify.inner_folds=3",
    "--set", "gnn_classify.seeds=2",
]  # fmt: skip


def toy_problem(n: int = 150, seed: int = 0):
    """Invented features, two relations and labelled training rows in 5
    folds (groups of 2), plus 30 nodes that are neither (test-like)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    edges = {"aaa": fake_edges(n, 3 * n, seed), "bbb": fake_edges(n, 2 * n, seed + 1)}
    train_nodes = np.arange(n - 30)
    y = (x[train_nodes, 0] + rng.normal(scale=0.5, size=len(train_nodes)) > 0.6).astype(
        int
    )
    groups = train_nodes // 2
    folds = groups % 5
    return x, edges, train_nodes, y, folds, groups


# --------------------------------------------------------------------------
# The layer
# --------------------------------------------------------------------------
def _edge_index(edges: dict) -> dict:
    """Both directions of each undirected edge list, as PyG edge_index."""
    out = {}
    for r, e in edges.items():
        e = torch.as_tensor(np.asarray(e, dtype=np.int64))
        out[r] = torch.cat([e, e.flip(0)], dim=1)
    return out


def test_mean_layer_is_pyg_heteroconv_sageconv_and_rgcnconv():
    from torch_geometric.nn import HeteroConv, RGCNConv, SAGEConv  # noqa: PLC0415

    x, edges, *_ = toy_problem(n=40)
    g = sup.graph_input(x, edges, "mean")
    torch.manual_seed(0)
    layer = sup.RelSAGE(4, 6, n_relations=2)
    ours = layer(g.x, g.adjs).detach()
    index = _edge_index(edges)

    # HeteroConv over SAGEConv (mean, root weight), summed over relations.
    convs = {("n", r, "n"): SAGEConv(4, 6, aggr="mean") for r in edges}
    hetero = HeteroConv(convs, aggr="sum")
    with torch.no_grad():
        root = torch.zeros(6, 4)
        bias = torch.zeros(6)
        for j, r in enumerate(edges):
            conv = convs[("n", r, "n")]
            conv.lin_l.weight.copy_(layer.rel[j].weight)
            conv.lin_l.bias.zero_()
            conv.lin_r.weight.copy_(layer.root.weight / 2)
            root += conv.lin_r.weight
        bias += layer.root.bias
        out = hetero({"n": g.x}, {("n", r, "n"): index[r] for r in edges})["n"] + bias
    assert torch.allclose(root, layer.root.weight, atol=1e-6)
    assert torch.allclose(out, ours, atol=1e-5)

    # RGCNConv with mean aggregation and a root weight.
    rgcn = RGCNConv(4, 6, num_relations=2, aggr="mean", root_weight=True, bias=True)
    with torch.no_grad():
        for j in range(2):
            rgcn.weight[j].copy_(layer.rel[j].weight.T)
        rgcn.root.copy_(layer.root.weight.T)
        rgcn.bias.copy_(layer.root.bias)
        all_edges = torch.cat([index["aaa"], index["bbb"]], dim=1)
        sizes = (index["aaa"].shape[1], index["bbb"].shape[1])
        types = torch.cat(
            [torch.full((n,), j, dtype=torch.long) for j, n in enumerate(sizes)]
        )
        out = rgcn(g.x, all_edges, types)
    assert torch.allclose(out, ours, atol=1e-5)


def test_sym_layer_matches_dense_arithmetic_and_isolated_nodes_get_zero():
    x, edges, *_ = toy_problem(n=40)
    edges["aaa"] = edges["aaa"][:, edges["aaa"].min(axis=0) > 0]  # node 0 isolated in aaa
    for norm in sup.NORMS:
        a_hat = sup.normalised_adjacency(edges["aaa"], 40, norm).toarray()
        a = (sup.adjacency(edges["aaa"], 40).toarray() > 0).astype(float)
        d = a.sum(axis=1)
        with np.errstate(divide="ignore"):
            inv = np.where(d > 0, d ** (-1.0 if norm == "mean" else -0.5), 0.0)
        expected = inv[:, None] * a if norm == "mean" else inv[:, None] * a * inv[None, :]
        assert np.allclose(a_hat, expected) and not a_hat[0].any()


def test_sparse_product_gradient_is_the_dense_gradient():
    x, edges, *_ = toy_problem(n=30)
    a_hat = sup.normalised_adjacency(edges["aaa"], 30, "mean")
    a, a_t = sup._torch_csr(a_hat, "cpu"), sup._torch_csr(a_hat.T, "cpu")
    h = torch.randn(30, 3, dtype=torch.float32, requires_grad=True)
    sup._SparseProduct.apply(a, a_t, h).pow(2).sum().backward()
    dense = torch.as_tensor(a_hat.toarray(), dtype=torch.float32)
    h2 = h.detach().clone().requires_grad_(True)
    (dense @ h2).pow(2).sum().backward()
    assert torch.allclose(h.grad, h2.grad, atol=1e-5)


def test_without_relations_the_network_ignores_edges():
    x, edges, train_nodes, y, folds, groups = toy_problem()
    none = sup.graph_input(x, {}, "mean")
    first = sup.cross_fit(none, train_nodes, y, folds, groups, SETTING, CFG, seed=1)
    again = sup.cross_fit(sup.graph_input(x, {}, "sym"), train_nodes, y, folds, groups,
                          SETTING, CFG, seed=1)  # fmt: skip
    assert np.array_equal(first.oof, again.oof)
    real = sup.cross_fit(sup.graph_input(x, edges, "mean"), train_nodes, y, folds, groups,
                         SETTING, CFG, seed=1)  # fmt: skip
    assert not np.allclose(first.oof, real.oof)


# --------------------------------------------------------------------------
# Cross-fitting and the leakage rules
# --------------------------------------------------------------------------
def test_cross_fit_is_deterministic_on_the_cpu_and_seeded():
    x, edges, train_nodes, y, folds, groups = toy_problem()
    g = sup.graph_input(x, edges, "mean")
    first = sup.cross_fit(g, train_nodes, y, folds, groups, SETTING, CFG, seed=3)
    second = sup.cross_fit(g, train_nodes, y, folds, groups, SETTING, CFG, seed=3)
    other = sup.cross_fit(g, train_nodes, y, folds, groups, SETTING, CFG, seed=4)
    assert np.array_equal(first.oof, second.oof)
    assert np.array_equal(first.all_nodes, second.all_nodes)
    assert not np.array_equal(first.oof, other.oof)
    assert not np.isnan(first.oof).any() and first.all_nodes.shape == (len(x),)
    assert len(first.logs) == 5 and all(log["n_val"] > 0 for log in first.logs)


def test_fold_labels_never_reach_their_own_out_of_fold_or_nested_scores():
    x, edges, train_nodes, y, folds, groups = toy_problem()
    g = sup.graph_input(x, edges, "mean")
    outer, nested = sup.nested_cross_fit(
        g, train_nodes, y, folds, groups, SETTING, CFG, 2
    )
    rng = np.random.default_rng(0)
    for k in (0, 3):
        changed = y.copy()
        held = folds == k
        changed[held] = rng.permutation(changed[held])
        changed[np.flatnonzero(held)[:2]] = 1 - changed[np.flatnonzero(held)[:2]]
        outer2, nested2 = sup.nested_cross_fit(g, train_nodes, changed, folds, groups,
                                               SETTING, CFG, 2)  # fmt: skip
        assert np.array_equal(outer.oof[held], outer2.oof[held])
        assert np.array_equal(nested[k], nested2[k], equal_nan=True)
        # The other folds' scores do use fold k's labels (they are training rows).
        assert not np.array_equal(outer.oof[~held], outer2.oof[~held])


def test_nested_scores_cover_exactly_the_other_folds():
    x, edges, train_nodes, y, folds, groups = toy_problem()
    g = sup.graph_input(x, edges, "sym")
    _, nested = sup.nested_cross_fit(g, train_nodes, y, folds, groups, SETTING, CFG, 5)
    assert sorted(nested) == [0, 1, 2, 3, 4]
    for k, scores in nested.items():
        assert np.isnan(scores[folds == k]).all()
        assert not np.isnan(scores[folds != k]).any()
        assert ((scores[folds != k] > 0) & (scores[folds != k] < 1)).all()


def test_train_config_and_grid():
    cfg = sup.TrainConfig.from_config({"hidden": [64, 128], "dropout": 0.3, "lr": [1e-3]})
    assert [s.key for s in cfg.grid()] == ["h64_d0.3_lr0.001", "h128_d0.3_lr0.001"]
    with pytest.raises(ValueError, match="Unknown"):
        sup.TrainConfig.from_config({"width": 3})
    assert sup.fit_seed(555, 2) != sup.fit_seed(555, 2, 1) != sup.fit_seed(556, 2)
    assert np.allclose(sup.sigmoid(sup.logit(np.array([0.1, 0.5, 0.9]))), [0.1, 0.5, 0.9])


def _cv_select_module():
    spec = importlib.util.spec_from_file_location(
        "cv_select", REPO_ROOT / "scripts" / "dfa" / "cv_select.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cv_select_fits_on_nested_scores_and_keeps_the_held_out_rows(tmp_path):
    cv = _cv_select_module()
    fold_of_row = np.array([0, 0, 1, 1, 2, 2])
    rows = [(k, r) for k in range(3) for r in range(6) if fold_of_row[r] != k]
    nested = pd.DataFrame(
        {
            "row": [r for _, r in rows],
            "outer_fold": [k for k, _ in rows],
            "column": "gnnsup_score",
            "value": [100 * k + r for k, r in rows],
        }
    )
    path = tmp_path / "nested.parquet"
    nested.to_parquet(path)
    checked = cv.read_nested(path, fold_of_row)
    X = pd.DataFrame({"gnnsup_score": np.arange(6) * -1.0, "other": np.arange(6)})
    fit = cv.with_nested(X.iloc[np.flatnonzero(fold_of_row != 1)], checked, 1)
    assert fit["gnnsup_score"].tolist() == [100, 101, 104, 105]
    assert fit["other"].tolist() == [0, 1, 4, 5] and X["gnnsup_score"].iloc[2] == -2
    assert cv.with_nested(X, None, 1) is X
    nested[~((nested["outer_fold"] == 0) & (nested["row"] == 2))].to_parquet(path)
    with pytest.raises(SystemExit, match="cover exactly"):
        cv.read_nested(path, fold_of_row)


# --------------------------------------------------------------------------
# The stages
# --------------------------------------------------------------------------
def run_rung(path, feature_set: str, *extra: str) -> int:
    return main(["--config", str(path), "--set", f"feature_set={feature_set}", *SMALL,
                 *extra])  # fmt: skip


def test_gnn_classify_scores_a_rung_like_model_a(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=300)
    assert run_rung(path, "pilot_g5_sage") == 0
    p = get_paths({**config, "feature_set": "pilot_g5_sage"})
    assert p.is_gnn and not p.df_analyze_output_dir.exists()  # df-analyze never ran
    summary = json.loads((p.model_reports_dir / "model_a.json").read_text("utf-8"))
    assert gnn_outputs.is_gnn_summary(summary)
    assert (
        summary["gnn"]["seeds"] == [555, 556] and len(summary["gnn"]["fold_pr_auc"]) == 5
    )
    assert summary["model_a"]["model"] == "GraphSAGE"
    grid = pd.read_csv(p.model_reports_dir / gnn_outputs.GNN_GRID)
    assert len(grid) == 2 and summary["gnn"]["setting"]["lr"] in (0.02, 0.05)
    best = grid.loc[grid["cv_pr_auc"].idxmax(), "setting"]
    assert sup.Setting(**summary["gnn"]["setting"]).key == best
    test_ids = pd.read_csv(p.test_ids)
    prob = gnn_outputs.test_probabilities(p.model_reports_dir, test_ids)
    assert np.isclose(summary["model_a"]["pr_auc"], summary["intervals"][0]["estimate"])
    oof = gnn_outputs.oof_rows(p.model_reports_dir)
    assert len(oof) == summary["n_train"] and len(prob) == summary["n_test"]
    metrics = pd.read_csv(p.model_reports_dir / "model_a_metrics.csv")
    assert len(metrics) == 1 and {C.LABEL, C.NODE_ID}.isdisjoint(metrics.columns)

    # Test labels are read only to score: changing them leaves every
    # probability as it was.
    flipped = test_ids.copy()
    flipped[C.LABEL] = 1 - flipped[C.LABEL]
    flipped.to_csv(p.test_ids, index=False)
    assert run_rung(path, "pilot_g5_sage", "--only", "gnn_classify", "--force") == 0
    assert np.array_equal(
        gnn_outputs.test_probabilities(p.model_reports_dir, flipped), prob
    )
    assert np.array_equal(gnn_outputs.oof_rows(p.model_reports_dir)["prob"], oof["prob"])


def test_mlp_and_rewired_rungs_and_setting_from(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=300)
    assert run_rung(path, "pilot_g5n3_sage_rw") == 2  # needs E5's setting first
    for fs in ("pilot_g5_sage", "pilot_g6_mlp", "pilot_g5n3_sage_rw"):
        assert run_rung(path, fs, "--force") == 0
    read = {}
    for fs in ("pilot_g5_sage", "pilot_g6_mlp", "pilot_g5n3_sage_rw"):
        reports = get_paths({**config, "feature_set": fs}).model_reports_dir
        read[fs] = json.loads((reports / "model_a.json").read_text("utf-8"))
    assert read["pilot_g6_mlp"]["gnn"]["relations"] == []
    assert (
        read["pilot_g5n3_sage_rw"]["gnn"]["setting"]
        == read["pilot_g5_sage"]["gnn"]["setting"]
    )
    assert read["pilot_g5n3_sage_rw"]["gnn"]["rewired_graph"].endswith("seed555")
    p = get_paths({**config, "feature_set": "pilot_g5n3_sage_rw"})
    assert (p.rewired_dir(555) / "manifest.json").is_file()

    # A GNN side goes through the comparison page, the split-repeat summary
    # and the results database like a df-analyze Model A.
    from fraud_detect import results_db  # noqa: PLC0415
    from fraud_detect.report import compare  # noqa: PLC0415

    comparison = compare.build(config, "pilot_g6_mlp", "pilot_g5_sage", n_resamples=50,
                               n_random_cards=5, n_gallery=3)  # fmt: skip
    delta = comparison.metrics["delta_pr_auc"]["estimate"]
    pr = {fs: read[fs]["model_a"]["pr_auc"] for fs in ("pilot_g6_mlp", "pilot_g5_sage")}
    assert np.isclose(delta, pr["pilot_g5_sage"] - pr["pilot_g6_mlp"])
    spec = importlib.util.spec_from_file_location(
        "repeat_summary", REPO_ROOT / "scripts" / "repeat_summary.py"
    )
    summary_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary_script)
    _, folds = summary_script.side(config, "pilot_g5_sage")
    assert np.allclose(folds.to_numpy(), read["pilot_g5_sage"]["gnn"]["fold_pr_auc"])
    fields, table = results_db.from_pipeline_run(
        get_paths({**config, "feature_set": "pilot_g5_sage"}).feature_set_dir
    )
    results_db.check_results_table(table)
    assert fields["headline"]["model_a"] == "GraphSAGE (gnn)"


def test_gnn_classify_resumes_after_an_interruption(tmp_path, monkeypatch):
    path, config = fake_dataset(tmp_path, n_nodes=300)
    assert run_rung(path, "pilot_g6_mlp") == 0
    p = get_paths({**config, "feature_set": "pilot_g6_mlp"})
    reference = gnn_outputs.oof_rows(p.model_reports_dir)
    calls: list[int] = []
    real = sup.cross_fit

    def interrupted_once(*args, **kwargs):
        calls.append(args[-1])
        if len(calls) == 2:
            raise KeyboardInterrupt  # stopped during the second unit
        return real(*args, **kwargs)

    monkeypatch.setattr(sup, "cross_fit", interrupted_once)
    assert run_rung(path, "pilot_g6_mlp", "--only", "gnn_classify", "--force") == 130
    assert run_rung(path, "pilot_g6_mlp", "--resume") == 0
    assert len(calls) == 4  # 2 grid settings + 1 more seed; the first unit was kept
    pd.testing.assert_frame_equal(gnn_outputs.oof_rows(p.model_reports_dir), reference)


def test_gnn_oof_builds_gnnsup_with_nested_scores_for_the_shared_cv(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=300)
    for fs in ("pilot_g5_sage", "pilot_g5g_rgcn"):
        assert run_rung(path, fs) == 0
    fs = "pilot_m8s_own_gnnsup"
    assert run_rung(path, fs, "--to", "df_analyze_input") == 0
    p = get_paths({**config, "feature_set": fs})
    block = pd.read_parquet(p.block_file(C.GNNSUP_BLOCK))
    nested = pd.read_parquet(p.nested_scores(C.GNNSUP_BLOCK))
    meta = json.loads((p.gnn_oof_dir / "meta.json").read_text("utf-8"))
    assert list(block.columns) == [C.NODE_ID, "gnnsup__score"] and len(block) == 300
    assert meta["choice"]["feature_set"] in ("pilot_g5_sage", "pilot_g5g_rgcn")
    folds = pd.read_csv(p.cv_folds)
    for k, frame in nested.groupby("outer_fold"):
        assert set(frame[C.NODE_ID]) == set(folds.loc[folds["fold"] != k, C.NODE_ID])
    # Training rows carry their out-of-fold score (a seed-averaged logit),
    # identical to the outer fits of the chosen rung's own cross-fitting.
    train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    by_id = block.set_index(C.NODE_ID)["gnnsup__score"]
    train_ids = pd.read_csv(p.train_ids)
    assert np.allclose(train["gnnsup_score"], by_id.loc[train_ids[C.NODE_ID]].to_numpy())

    # select_model's nested file: df-analyze's row order and column name.
    export = tmp_path / "export"
    export.mkdir()
    train.drop(columns=C.TARGET).to_csv(export / "X_train_00.csv", index=False)
    cfg = {**config, "feature_set": fs}
    ctx = Context(config=cfg, paths=p, run_dir=tmp_path)
    args = nested_scores_file(ctx, tmp_path, export)
    table = pd.read_parquet(args[1])
    assert set(table["column"]) == {"gnnsup_score"}
    row_folds = folds.set_index(C.NODE_ID).loc[train_ids[C.NODE_ID], "fold"].to_numpy()
    cv = _cv_select_module()
    cv.read_nested(args[1], row_folds)  # covers the other folds exactly
    shifted = train.drop(columns=C.TARGET)
    shifted["gnnsup_score"] += 1.0
    shifted.to_csv(export / "X_train_00.csv", index=False)
    with pytest.raises(Exception, match="changed column"):
        nested_scores_file(ctx, tmp_path, export)


def test_best_gnn_rule(tmp_path, monkeypatch):
    from fraud_detect import pipeline  # noqa: PLC0415

    path, config = fake_dataset(tmp_path, n_nodes=60)
    p = get_paths({**config, "feature_set": "pilot_m8s_own_gnnsup"})
    ctx = Context(config={**config, "feature_set": "pilot_m8s_own_gnnsup"}, paths=p,
                  run_dir=tmp_path)  # fmt: skip
    folds = {"pilot_g5_sage": [0.5, 0.6, 0.55, 0.5, 0.6]}

    def summary(_ctx, fs):
        setting = {"hidden": 8, "dropout": 0.0, "lr": 0.02}
        return {
            "gnn": {"fold_pr_auc": folds[fs], "setting": setting},
            "model_a": {"cv_pr_auc": float(np.mean(folds[fs]))},
        }

    monkeypatch.setattr(pipeline, "gnn_rung_summary", summary)
    folds["pilot_g5g_rgcn"] = [0.51, 0.59, 0.57, 0.5, 0.61]  # +0.006 mean, SE ~0.005
    assert pipeline.best_gnn(ctx)["feature_set"] == "pilot_g5g_rgcn"
    folds["pilot_g5g_rgcn"] = [0.6, 0.5, 0.55, 0.52, 0.6]  # +0.004 mean, SE ~0.03
    assert pipeline.best_gnn(ctx)["feature_set"] == "pilot_g5_sage"


# --------------------------------------------------------------------------
# Larger rungs (docs/BIGGER_GNN_PLAN.md): options 1 (a larger GraphSAGE) and
# 2 (SIGN)
# --------------------------------------------------------------------------
def _sage_size(d: int, h: int, r: int, layers: int, norm: bool, jk: bool) -> int:
    """Parameters of GNNClassifier: per layer a root weight + bias and one
    weight per relation; LayerNorm 2h per layer; the head."""
    dims = [d] + [h] * layers
    convs = sum((r + 1) * a * b + b for a, b in zip(dims, dims[1:], strict=False))
    return convs + (2 * h * layers if norm else 0) + (h * layers if jk else h) + 1


def test_default_network_is_the_2026_10_31_network():
    x, edges, *_ = toy_problem(n=40)
    g = sup.graph_input(x, edges, "mean")
    torch.manual_seed(0)
    old = sup.GNNClassifier(4, 8, 2, 2, 0.0)
    torch.manual_seed(0)
    new = sup.build_model(g, SETTING, CFG)
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    assert not any("norm" in k for k in new.state_dict())
    assert torch.equal(old.eval()(g), new.eval()(g))
    assert sup.n_parameters(new) == _sage_size(4, 8, 2, 2, norm=False, jk=False)


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("jk", [False, True])
def test_larger_sage_options_change_only_what_they_say(residual, layer_norm, jk):
    x, edges, *_ = toy_problem(n=40)
    g = sup.graph_input(x, edges, "mean")
    torch.manual_seed(0)
    model = sup.GNNClassifier(4, 16, 2, 3, 0.0, residual=residual,
                              layer_norm=layer_norm, jk=jk).eval()  # fmt: skip
    out = model(g)
    assert out.shape == (40,) and torch.isfinite(out).all()
    assert sup.n_parameters(model) == _sage_size(4, 16, 2, 3, layer_norm, jk)
    # Residual: the same weights without it give another output.
    if residual:
        model.residual = False
        assert not torch.allclose(model(g), out)


def test_amazon_sized_parameter_counts_in_the_plan():
    # docs/BIGGER_GNN_PLAN.md quotes these for 25 features and 3 relations.
    assert _sage_size(25, 128, 3, 2, norm=False, jk=False) == 78_721  # E5 today
    assert _sage_size(25, 512, 3, 3, norm=True, jk=True) == 2_154_497  # E5w at 512


@pytest.mark.parametrize("norm", sup.NORMS)
def test_sign_inputs_are_powers_of_each_normalised_adjacency(norm):
    x, edges, *_ = toy_problem(n=40)
    g = sup.graph_input(x, edges, norm)
    inputs = sup.hop_features(g, 3)
    assert len(inputs) == 1 + 2 * 3
    expected = [x]
    for e in edges.values():
        a = sup.normalised_adjacency(e, 40, norm).toarray()
        expected += [np.linalg.matrix_power(a, k) @ x for k in (1, 2, 3)]
    for got, want in zip(inputs, expected, strict=True):
        assert np.allclose(got.numpy(), want, atol=1e-5)
    assert sup.hop_features(g, 3) is inputs  # computed once per graph
    assert len(sup.hop_features(sup.graph_input(x, {}, norm), 3)) == 1  # no edges: [x]


def test_sign_size_cross_fit_and_determinism():
    x, edges, train_nodes, y, folds, groups = toy_problem()
    cfg = sup.TrainConfig(**{**CFG.__dict__, "model": "sign", "hops": 2, "layers": 3,
                             "layer_norm": True})  # fmt: skip
    g = sup.graph_input(x, edges, "mean")
    model = sup.build_model(g, SETTING, cfg)
    n_inputs, h = 1 + 2 * 2, 8
    branches = n_inputs * (4 * h + h) + n_inputs * 2 * h  # linear + LayerNorm
    mlp = (n_inputs * h * h + h) + (h * h + h) + 2 * 2 * h  # 2 layers + LayerNorm
    expected = branches + mlp + h + 1  # + the head
    assert isinstance(model, sup.SIGNClassifier) and sup.n_parameters(model) == expected
    first = sup.cross_fit(g, train_nodes, y, folds, groups, SETTING, cfg, seed=3)
    again = sup.cross_fit(g, train_nodes, y, folds, groups, SETTING, cfg, seed=3)
    assert np.array_equal(first.oof, again.oof) and not np.isnan(first.oof).any()
    assert all(log["n_parameters"] == expected for log in first.logs)
    edge_free = sup.cross_fit(sup.graph_input(x, {}, "mean"), train_nodes, y, folds,
                              groups, SETTING, cfg, seed=3)  # fmt: skip
    assert not np.allclose(first.oof, edge_free.oof)


def test_sign_fold_labels_never_reach_their_own_out_of_fold_scores():
    x, edges, train_nodes, y, folds, groups = toy_problem()
    cfg = sup.TrainConfig(**{**CFG.__dict__, "model": "sign", "hops": 2})
    g = sup.graph_input(x, edges, "mean")
    first = sup.cross_fit(g, train_nodes, y, folds, groups, SETTING, cfg, seed=2)
    changed = y.copy()
    held = folds == 1
    changed[held] = 1 - changed[held]
    second = sup.cross_fit(g, train_nodes, changed, folds, groups, SETTING, cfg, seed=2)
    assert np.array_equal(first.oof[held], second.oof[held])


def test_rung_config_puts_the_rung_on_top_and_checks_it():
    section = {"hidden": [64, 128], "dropout": [0.0, 0.3], "lr": [1e-3], "seeds": 5}
    plain = sup.rung_config(section, {"graph": "real"})
    assert plain == sup.TrainConfig.from_config(section)
    big = sup.rung_config(section, {"train": {"layers": 3, "hidden": [256, 512],
                                              "residual": True, "jk": True}})  # fmt: skip
    assert (big.layers, big.hidden, big.seeds) == (3, (256, 512), 5)
    assert big.dropout == (0.0, 0.3)  # from the section: the rung did not set it
    assert big.residual and big.jk and not big.layer_norm and big.model == "sage"
    for bad, match in [
        ({"model": "gcn"}, "model"),
        ({"model": "sign", "jk": True}, "sage model"),
        ({"model": "sign", "hops": 0}, "hops"),
        ({"residual": "yes"}, "true or false"),
    ]:
        with pytest.raises(ValueError, match=match):
            sup.rung_config(section, {"train": bad})


LARGER = ("pilot_g5w_sage_big", "pilot_g6w_mlp_big", "pilot_g9_sign",
          "pilot_g9n3_sign_rw")  # fmt: skip
# Shrink the larger rungs for a quick test. Through the YAML anchors this also
# shrinks their twin (E6w) and control (N9): the same network by construction.
SHRINK = [
    "--set", "gnn_models.pilot_g5w_sage_big.train.hidden=[8]",
    "--set", "gnn_models.pilot_g5w_sage_big.train.dropout=[0.0]",
    "--set", "gnn_models.pilot_g9_sign.train.hidden=[8]",
    "--set", "gnn_models.pilot_g9_sign.train.dropout=[0.0]",
]  # fmt: skip


def test_larger_rungs_run_end_to_end_and_their_controls_share_the_network(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=300)
    for fs in LARGER:
        assert run_rung(path, fs, *SHRINK) == 0, fs
    read = {}
    for fs in LARGER:
        reports = get_paths({**config, "feature_set": fs}).model_reports_dir
        read[fs] = json.loads((reports / "model_a.json").read_text("utf-8"))
    big, twin, sign, sign_rw = (read[fs] for fs in LARGER)
    assert big["model_a"]["model"] == "GraphSAGE, larger"
    assert twin["model_a"]["model"] == "MLP (GNN without edges), larger"
    assert sign["model_a"]["model"] == "SIGN"
    assert sign_rw["model_a"]["model"] == "SIGN (rewired graph)"
    for key in ("layers", "residual", "layer_norm", "jk", "hidden"):
        assert big["gnn"]["config"][key] == twin["gnn"]["config"][key]
    assert big["gnn"]["config"]["layers"] == 3 and big["gnn"]["config"]["jk"] is True
    assert twin["gnn"]["relations"] == []
    assert sign["gnn"]["config"]["model"] == "sign" and sign["gnn"]["config"]["hops"] == 3
    assert sign_rw["gnn"]["setting"] == sign["gnn"]["setting"]
    assert sign_rw["gnn"]["rewired_graph"].endswith("seed555")
