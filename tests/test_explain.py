"""Explanation logic on synthetic data built in memory; no model library needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from synthetic import toy_explain_graph, toy_explain_table

from fraud_detect import columns as C
from fraud_detect.features.graph import adjacency, build_nbr_block
from fraud_detect.models import explain as X
from fraud_detect.models import explain_eval as E

RENAMES = {
    "own_f00": "own__f00",
    "own_f01": "own__f01",
    "own_f22": "own__f22",
    "nbr_upu_mean_f00": "nbr__upu_mean_f00",
    "nbr_upu_mean_f01": "nbr__upu_mean_f01",
    "nbr_upu_log1p_degree": "nbr__upu_log1p_degree",
    "nbr_no19_usu_mean_f00": "nbr_no19__usu_mean_f00",
    "gnnssl_rur_pc01": "gnnssl__rur_pc01",
}


def _adjacency(edges: dict[str, np.ndarray], n: int) -> dict:
    return {r: adjacency(e, n) for r, e in edges.items()}


def test_columns_map_to_blocks_and_binary_indicators_map_back() -> None:
    names = ["own_f00", "own_f22_1.0", "nbr_upu_mean_f00", "nbr_upu_log1p_degree",
             "nbr_no19_usu_mean_f00", "gnnssl_rur_pc01"]  # fmt: skip
    infos = X.parse_columns(names, RENAMES)
    assert [c.player for c in infos] == [
        "own", "own", "nbr_upu", "degree", "nbr_usu", "gnnssl_rur",
    ]  # fmt: skip
    assert infos[1].block_column == "own__f22" and infos[1].base_feature == "f22"
    assert infos[2].base_feature == "f00" and infos[2].stat == "mean"
    assert infos[3].relation == "upu" and infos[3].stat == "log1p_degree"
    assert infos[5].stat == "pc"
    assert list(X.players_of(infos)) == [
        "own",
        "nbr_upu",
        "nbr_usu",
        "gnnssl_rur",
        "degree",
    ]
    with pytest.raises(X.ExplainError):
        X.parse_columns(["mystery"], RENAMES)


def test_correlated_group_is_own_alone_and_with_neighbour_means() -> None:
    infos = X.parse_columns(["own_f00", "own_f01", "nbr_upu_mean_f00"], RENAMES)
    groups = X.correlated_group_columns(infos, ["own__f00"])
    assert groups == {
        "#36 group (own)": [0],
        "#36 group (own + neighbour means)": [0, 2],
    }


def test_implementation_check_finds_wrong_outputs() -> None:
    values = np.array([[0.5, -0.25], [1.0, 0.0]])
    margin = np.array([1.25, 2.0])
    bad, check = X.implementation_check(values, 1.0, margin, np.array([0.9, 1.1]))
    assert not bad.any() and check["n_rows_excluded"] == 0
    with pytest.raises(X.ExplainError, match="misses"):
        X.implementation_check(values, 0.0, margin)
    with pytest.raises(X.ExplainError, match="background"):
        X.implementation_check(values, 1.0, margin, np.array([3.0, 3.0]))


def test_near_threshold_finds_values_equal_in_single_precision() -> None:
    threshold = 45.333333333333336
    rows = np.array([[threshold, 1.0], [np.nextafter(threshold, 0.0), 1.0], [45.4, 1.0]])
    near = X.near_threshold(rows, {0: np.array([threshold])})
    assert near.tolist() == [True, True, False]
    assert not X.near_threshold(rows, {1: np.array([2.0])}).any()


def test_implementation_check_excludes_a_few_rows_but_not_many() -> None:
    values = np.zeros((1000, 2))
    margin = np.zeros(1000)
    margin[7] = 0.5  # one row misses by a leaf's worth
    bad, check = X.implementation_check(values, 0.0, margin)
    assert bad.nonzero()[0].tolist() == [7]
    assert check["n_rows_excluded"] == 1 and check["max_error_excluded"] == 0.5
    assert check["max_additivity_error"] == 0.0
    margin[:20] = 0.5  # 2% of rows: not a rare cause any more
    with pytest.raises(X.ExplainError, match="misses"):
        X.implementation_check(values, 0.0, margin)


def test_importance_interval_resamples_whole_groups() -> None:
    values = np.array([[1.0, 0.1], [1.0, 0.1], [3.0, 0.1], [3.0, 0.1]])
    # Two groups of two rows: a resample holds 0, 1 or 2 copies of each group, so
    # the first column's resampled mean can only be 1, 2 or 3.
    table = X.importance_table(values, ["a", "b"], np.array([0, 0, 1, 1]), seed=1)
    row = table.set_index("name").loc["a"]
    assert row["mean_abs_shap"] == pytest.approx(2.0)
    assert row["low"] == pytest.approx(1.0) and row["high"] == pytest.approx(3.0)
    assert table.iloc[0]["name"] == "a" and table.iloc[0]["rank"] == 1


def test_unstable_flag_marks_a_wide_rank_interval() -> None:
    rng = np.random.default_rng(0)
    # 12 near-identical columns in 40 groups: their ranks swap in every resample.
    values = rng.normal(1.0, 0.5, size=(40, 12))
    table = X.importance_table(
        values, [f"c{i}" for i in range(12)], np.arange(40), seed=0
    )
    assert table["unstable"].any()
    steady = np.tile(np.arange(1.0, 13.0), (40, 1)) + rng.normal(0, 1e-3, size=(40, 12))
    table = X.importance_table(
        steady, [f"c{i}" for i in range(12)], np.arange(40), seed=0
    )
    assert not table["unstable"].any()


def test_strata_use_the_given_threshold() -> None:
    y = np.array([1, 1, 0, 0])
    masks = X.strata_masks(y, np.array([0.9, 0.1, 0.8, 0.2]), 0.5)
    assert masks["true_positive"].tolist() == [True, False, False, False]
    assert masks["false_negative"].tolist() == [False, True, False, False]
    assert masks["false_positive"].tolist() == [False, False, True, False]
    assert masks["flagged"].sum() == 2


def test_rank_agreement() -> None:
    a = pd.Series({"p": 3.0, "q": 2.0, "r": 1.0})
    same = X.rank_agreement(a, a * 2, k=2)
    assert same["kendall_tau"] == pytest.approx(1.0) and same["top2_jaccard"] == 1.0
    flipped = X.rank_agreement(a, 4 - a, k=1)
    assert (
        flipped["kendall_tau"] == pytest.approx(-1.0) and flipped["top1_jaccard"] == 0.0
    )
    assert flipped["top1_max_rank_move"] == 2


# ---- Neighbour removal ----------------------------------------------------
def _model_table() -> tuple:
    own_table, edges = toy_explain_graph()
    n = len(own_table)
    nbr = build_nbr_block(own_table, edges)
    own = own_table[["own__f00", "own__f01"]].to_numpy(dtype=float)
    columns = [c for c in nbr.columns if c != C.NODE_ID]
    prepared = [c.replace("__", "_") for c in columns]
    renames = dict(zip(prepared, columns, strict=True))
    infos = X.parse_columns(prepared, renames)
    layouts = X.neighbour_layouts(infos, ["own__f00", "own__f01"])
    return (
        own_table,
        edges,
        n,
        own,
        nbr[columns].to_numpy(dtype=float),
        prepared,
        infos,
        layouts,
    )


def test_a_removal_matches_the_block_built_on_the_reduced_graph() -> None:
    own_table, edges, n, own, x, _, _, layouts = _model_table()
    relation, target = "upu", 0
    a = adjacency(edges[relation], n)
    neighbours = a.indices[a.indptr[target] : a.indptr[target + 1]]
    assert len(neighbours) >= 2
    mean_only, whole = X.rows_without_each_neighbour(
        x[target], neighbours, own, layouts[relation]
    )
    for i, j in enumerate(neighbours):
        e = edges[relation]
        pair = (e[0] == min(target, j)) & (e[1] == max(target, j))
        reduced = {**edges, relation: e[:, ~pair]}
        block = build_nbr_block(own_table, reduced).drop(columns=C.NODE_ID)
        np.testing.assert_allclose(whole[i], block.to_numpy(dtype=float)[target])
        # Mean-only keeps the original degree and changes nothing else.
        deg = layouts[relation].degree_position
        assert mean_only[i, deg] == x[target, deg]
        assert whole[i, deg] == pytest.approx(np.log1p(len(neighbours) - 1))


def test_a_lone_neighbour_leaves_zero_means_as_the_block_does() -> None:
    own_table, edges, n, own, x, _, _, layouts = _model_table()
    a = adjacency(edges["upu"], n)
    degree = np.asarray(a.sum(axis=1)).ravel()
    target = int(np.flatnonzero(degree == 1)[0])
    _, whole = X.rows_without_each_neighbour(
        x[target], a.indices[a.indptr[target] : a.indptr[target + 1]], own, layouts["upu"]
    )
    layout = layouts["upu"]
    assert (whole[0, layout.mean_positions] == 0).all()
    assert whole[0, layout.degree_position] == 0


def test_misaligned_rows_are_refused() -> None:
    _, edges, n, own, x, _, _, layouts = _model_table()
    a = adjacency(edges["upu"], n)
    targets = np.arange(n)
    X.check_layout(x, targets, own, a, layouts["upu"])
    with pytest.raises(X.ExplainError, match="lined up|degree"):
        X.check_layout(x[::-1], targets, own, a, layouts["upu"])


def test_planted_neighbour_ranks_first_and_effects_split() -> None:
    own_table, edges, n, own, x, prepared, _, layouts = _model_table()
    a = adjacency(edges["upu"], n)
    target = 0
    neighbours = a.indices[a.indptr[target] : a.indptr[target + 1]]
    planted = int(neighbours[0])
    own = own.copy()
    own[planted, 0] = 100.0  # one neighbour with a huge f00
    x = (
        build_nbr_block(
            pd.DataFrame(
                {C.NODE_ID: np.arange(n), "own__f00": own[:, 0], "own__f01": own[:, 1]}
            ),
            edges,
        )
        .drop(columns=C.NODE_ID)
        .to_numpy(dtype=float)
    )
    mean_col = prepared.index("nbr_upu_mean_f00")
    deg_col = prepared.index("nbr_upu_log1p_degree")

    def margin(rows: np.ndarray) -> np.ndarray:
        return 0.1 * rows[:, mean_col] + 0.5 * rows[:, deg_col]

    table = pd.concat(
        X.neighbour_influence(
            margin, x[[target]], np.array([target]), own,
            {"upu": a}, {"upu": layouts["upu"]}, X.training_ranges(x),
        )
    )  # fmt: skip
    assert (
        table.sort_values("delta_margin", ascending=False)["neighbour"].iloc[0] == planted
    )
    assert (table["target"] == target).all() and (
        table["degree"] == len(neighbours)
    ).all()
    # Mean-only and degree parts sum to the whole removal, in probability units.
    degree_part = table["delta_prob"] - table["delta_prob_mean"]
    assert np.abs(degree_part).max() > 0
    np.testing.assert_allclose(
        table["delta_prob_mean"] + degree_part, table["delta_prob"], atol=1e-6
    )


def test_out_of_range_flag() -> None:
    low, high = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    rows = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, -0.5]])
    assert X.out_of_range(rows, np.array([0, 1]), low, high).tolist() == [
        False,
        True,
        True,
    ]
    assert X.out_of_range(rows, np.array([1]), low, high).tolist() == [False, False, True]


def test_neighbour_shapley_sums_to_all_minus_none_and_splits_redundant_credit() -> None:
    _, _, _, own, x, prepared, _, layouts = _model_table()
    layout = layouts["upu"]
    own = np.zeros_like(own)
    # Two neighbours carry the same signal; a saturating model needs only one.
    own[1, 0] = own[2, 0] = 1.0
    own[3, 0] = 0.0
    neighbours = np.array([1, 2, 3])
    mean_col = prepared.index("nbr_upu_mean_f00")

    def margin(rows: np.ndarray) -> np.ndarray:
        return (rows[:, mean_col] > 0).astype(float)

    phi = X.neighbour_shapley(margin, x[0], neighbours, own, layout, 200, 0)
    assert phi.sum() == pytest.approx(1.0)
    assert phi[0] == pytest.approx(phi[1], abs=0.1) and phi[0] > 0.3
    # Leave-one-out sees neither: removing one redundant neighbour changes nothing.
    _, whole = X.rows_without_each_neighbour(x[0], neighbours, own, layout)
    assert (margin(whole)[:2] == 1).all()


def test_aggregate_influence_keeps_hubs_from_winning_on_count_alone() -> None:
    frame = pd.DataFrame(
        {
            "target": [1, 2, 3, 4],
            "neighbour": [9, 9, 9, 8],
            "relation": "upu",
            "degree": [10, 10, 10, 2],
            "delta_prob": [0.1, -0.1, 0.1, 0.25],
        }
    )
    table = X.aggregate_influence(frame, flagged_targets=np.array([4])).set_index(
        "neighbour"
    )
    assert table.loc[9, "sum_abs"] == pytest.approx(0.3)
    assert table.loc[8, "mean_abs"] == pytest.approx(0.25)
    assert table.loc[9, "mean_abs"] == pytest.approx(0.1)
    assert table.loc[9, "mean_abs_x_degree"] == pytest.approx(1.0)
    assert table.loc[9, "towards_fraud"] == pytest.approx(0.2)
    assert table.loc[9, "away_from_fraud"] == pytest.approx(0.1)
    assert table.loc[8, "sum_abs_flagged"] == pytest.approx(0.25)
    assert table.loc[9, "sum_abs_flagged"] == 0


def test_stratified_sample_keeps_shares_and_is_seeded() -> None:
    y = np.array([1] * 100 + [0] * 900)
    a = X.stratified_sample(y, 100, 3)
    assert y[a].sum() == 10 and len(a) == 100
    assert a.tolist() == X.stratified_sample(y, 100, 3).tolist()
    assert len(X.stratified_sample(y, 5000, 3)) == len(y)


# ---- Group Shapley --------------------------------------------------------
def _toy_players() -> dict[str, list[int]]:
    return {"a": [0, 1], "b": [2], "c": [3]}


def test_group_shapley_equals_summed_shap_on_an_additive_model() -> None:
    rng = np.random.default_rng(0)
    x, bg = rng.normal(size=(6, 4)), rng.normal(size=(20, 4))
    w = np.array([1.0, -2.0, 0.5, 3.0])

    def margin(rows: np.ndarray) -> np.ndarray:
        return rows @ w

    phi, names, base = E.group_shapley(margin, x, bg, _toy_players())
    # On a linear model each column's interventional SHAP is w_j (x_j - mean b_j).
    shap_values = (x - bg.mean(axis=0)) * w
    summed, _ = X.summed_by_group(shap_values, _toy_players())
    assert names == ["a", "b", "c"]
    np.testing.assert_allclose(phi, summed)
    np.testing.assert_allclose(phi.sum(axis=1) + base, margin(x))


def test_group_shapley_shares_a_planted_interaction_between_blocks() -> None:
    bg = np.zeros((1, 4))
    x = np.array([[1.0, 0.0, 1.0, 0.0]])

    def margin(rows: np.ndarray) -> np.ndarray:
        return rows[:, 0] * rows[:, 2]

    phi, _, _ = E.group_shapley(margin, x, bg, _toy_players())
    assert phi[0, 0] == pytest.approx(0.5) and phi[0, 1] == pytest.approx(0.5)
    assert phi[0, 2] == pytest.approx(0.0)


def test_group_shapley_differs_from_summed_shap_within_a_block() -> None:
    # A three-way product: the per-column game gives each column 1/3, so summing
    # gives block a 2/3, while the two-player block game splits it evenly.
    bg = np.zeros((1, 3))
    x = np.array([[1.0, 1.0, 1.0]])

    def margin(rows: np.ndarray) -> np.ndarray:
        return rows[:, 0] * rows[:, 1] * rows[:, 2]

    players = {"a": [0, 1], "b": [2]}
    phi, _, _ = E.group_shapley(margin, x, bg, players)
    assert phi[0, 0] == pytest.approx(0.5)
    assert 2 / 3 != pytest.approx(phi[0, 0])


def test_group_shapley_chunks_give_the_same_values() -> None:
    rng = np.random.default_rng(1)
    x, bg = rng.normal(size=(5, 4)), rng.normal(size=(7, 4))

    def margin(rows: np.ndarray) -> np.ndarray:
        return rows[:, 0] * rows[:, 2] + rows[:, 3]

    one, _, _ = E.group_shapley(margin, x, bg, _toy_players(), chunk_rows=2)
    two, _, _ = E.group_shapley(margin, x, bg, _toy_players(), chunk_rows=100)
    np.testing.assert_allclose(one, two)


# ---- Retrain after removal and the randomisation check ---------------------
def _fit(x: pd.DataFrame, y: np.ndarray, seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=500, random_state=seed).fit(x.to_numpy(), y)


def _prob(model: LogisticRegression, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x.to_numpy())[:, 1]


def _importance(model: LogisticRegression, x: pd.DataFrame) -> pd.Series:
    centred = x.to_numpy() - x.to_numpy().mean(axis=0)
    return pd.Series(np.abs(centred * model.coef_[0]).mean(axis=0), index=x.columns)


def test_roar_drops_only_when_the_informative_feature_goes() -> None:
    x, y, fold = toy_explain_table()
    table = E.roar_lite(
        x, y, fold, _fit, _prob, _importance,
        players={"signal": ["signal"], "noise": ["noise_0", "noise_1"]},
        top_ks=(1, 2), n_random=4, seed=0,
    )  # fmt: skip
    summary = E.roar_summary(table).set_index("name")
    assert summary.loc["signal", "mean_drop"] > 0.2
    assert abs(summary.loc["noise", "mean_drop"]) < 0.05
    assert summary.loc["top 1", "mean_drop"] > summary.loc["random 1", "mean_drop"]
    verdict = E.shap_beats_random(E.roar_summary(table)).set_index("k")
    assert verdict.loc[1, "top_k_hurts_more"]


def test_roar_ranking_never_sees_the_held_out_fold() -> None:
    x, y, fold = toy_explain_table()
    seen: list[int] = []

    def importance(model: LogisticRegression, rows: pd.DataFrame) -> pd.Series:
        seen.extend(rows.index.tolist())
        return _importance(model, rows)

    E.roar_lite(
        x, y, fold, _fit, _prob, importance, players={}, top_ks=(1,), n_random=1, seed=0
    )
    # Each row is ranked on in every fold except the one that holds it out.
    counts = pd.Series(seen).value_counts()
    assert counts.max() == len(np.unique(fold)) - 1


def test_randomisation_check_loses_structure() -> None:
    x, y, fold = toy_explain_table()
    table = E.randomisation_check(x, y, fold, _fit, _prob, _importance, n_permutations=3)
    true = table[table["run"] == "true labels"].iloc[0]
    permuted = table[table["run"] != "true labels"]
    assert true["pr_auc"] > permuted["pr_auc"].max()
    assert permuted["total_mean_abs"].mean() < 0.5 * true["total_mean_abs"]
