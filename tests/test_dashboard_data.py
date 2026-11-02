"""The dashboard's data layer, on small invented databases in tmp_path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import (
    fake_compare_entry,
    fake_contrast_row,
    fake_entry,
    fake_ladder_db,
    fake_run_entry,
)

from fraud_detect.dashboard import data
from fraud_detect.dashboard import registry as reg


def load(db) -> tuple[list, pd.DataFrame]:
    entries = data.load_entries(db)
    return entries, data.runs_table(entries)


def contrasts_of(entries) -> list[dict]:
    return data.current_contrasts(data.all_contrasts(entries))


def test_mean_and_sd_are_over_splits(tmp_path):
    for k, v in enumerate((0.7, 0.8, 0.9)):
        fake_run_entry(tmp_path, "amazon", "pilot_m1_own", k, test=v, cv=v - 0.1)
    _, runs = load(tmp_path)
    (row,) = data.setup_summary(runs, "amazon", reg.PILOT, reg.ALL)
    assert row["n_splits"] == 3 and row["splits"] == [0, 1, 2]
    assert row["test_pr_auc"] == pytest.approx(0.8)
    assert row["test_pr_auc_sd"] == pytest.approx(np.std([0.7, 0.8, 0.9], ddof=1))
    assert row["cv_pr_auc"] == pytest.approx(0.7)
    assert row["test_per_split"] == pytest.approx([0.7, 0.8, 0.9])
    assert row["model_a"] == ["lgbm (none)"] * 3 and len(row["run_ids"]) == 3


def test_one_split_has_no_sd_rather_than_zero(tmp_path):
    fake_run_entry(tmp_path, "amazon", "m1_own")
    _, runs = load(tmp_path)
    (row,) = data.setup_summary(runs, "amazon", reg.FULL, reg.ALL)
    assert row["n_splits"] == 1 and row["test_pr_auc_sd"] is None


def test_pilot_and_full_never_share_a_row(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", test=0.5)
    fake_run_entry(tmp_path, "amazon", "m1_own", test=0.9)
    _, runs = load(tmp_path)
    (pilot,) = data.setup_summary(runs, "amazon", reg.PILOT, reg.ALL)
    (full,) = data.setup_summary(runs, "amazon", reg.FULL, reg.ALL)
    assert pilot["test_pr_auc"] == pytest.approx(0.5) and pilot["budget"] == reg.PILOT
    assert full["test_pr_auc"] == pytest.approx(0.9) and full["budget"] == reg.FULL
    with pytest.raises(data.DashboardDataError):
        data.setup_summary(runs, "amazon", "t100", reg.ALL)


def test_a_variant_or_dataset_filter_keeps_others_out(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", test=0.9)
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own_no19", test=0.5)
    fake_run_entry(tmp_path, "yelpchi", "pilot_m1_own", test=0.7)
    _, runs = load(tmp_path)
    (row,) = data.setup_summary(runs, "amazon", reg.PILOT, reg.NO19)
    assert row["test_pr_auc"] == pytest.approx(0.5)
    (row,) = data.setup_summary(runs, "yelpchi", reg.PILOT, reg.ALL)
    assert row["test_pr_auc"] == pytest.approx(0.7)
    assert data.setup_summary(runs, "yelpchi", reg.PILOT, reg.NO19) == []


def test_rows_follow_registry_order_and_embedding_sizes_stay_apart(tmp_path):
    for name in (
        "pilot_g5_sage",
        "pilot_m5_own_gnnssl_no19raw",
        "pilot_m5_own_gnnssl_no19",
        "pilot_m5_own_gnnssl_no19d2",
        "pilot_m1_own_no19",
    ):
        fake_run_entry(tmp_path, "amazon", name)
    _, runs = load(tmp_path)
    rows = data.setup_summary(runs, "amazon", reg.PILOT, reg.NO19)
    assert [r["display_id"] for r in rows] == ["B", "E2 (2)", "E2", "E2 (raw 64)"]
    assert rows[1]["embed_size"] == "2" and rows[2]["embed_size"] == "8"


def test_a_superseded_entry_is_hidden_but_kept(tmp_path):
    old = fake_run_entry(tmp_path, "amazon", "pilot_m1_own", test=0.5, tag_id="old")
    fake_run_entry(
        tmp_path, "amazon", "pilot_m1_own", test=0.9, supersedes=old, tag_id="new"
    )
    entries, runs = load(tmp_path)
    assert len(runs) == 2 and runs["superseded"].sum() == 1
    assert sum(e.superseded for e in entries) == 1
    (row,) = data.setup_summary(runs, "amazon", reg.PILOT, reg.ALL)
    assert row["test_pr_auc"] == pytest.approx(0.9) and row["n_splits"] == 1


def test_a_failed_run_is_left_out_of_means_but_shown_as_failed(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", 0, test=0.8)
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", 1, test=0.1, status="failed")
    entries, runs = load(tmp_path)
    (row,) = data.setup_summary(runs, "amazon", reg.PILOT, reg.ALL)
    assert row["n_splits"] == 1 and row["test_pr_auc"] == pytest.approx(0.8)
    failed = data.failed_entries(entries)
    assert len(failed) == 1 and failed[0]["status"] == "failed"


def test_a_run_recorded_twice_stops_the_summary(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", 0, test=0.8, tag_id="a")
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", 0, test=0.9, tag_id="b")
    _, runs = load(tmp_path)
    assert data.duplicate_runs(runs) == ["amazon/pilot_m1_own/split 0"]
    with pytest.raises(data.DashboardDataError, match="more than once"):
        data.setup_summary(runs, "amazon", reg.PILOT, reg.ALL)


def test_normalised_skill_formula():
    assert data.normalised_skill(0.1, 0.1) == pytest.approx(0.0)
    assert data.normalised_skill(1.0, 0.1) == pytest.approx(1.0)
    assert data.normalised_skill(0.55, 0.1) == pytest.approx(0.5)
    assert data.normalised_skill(0.05, 0.1) < 0  # worse than no skill stays visible


def test_normalised_skill_is_averaged_per_split(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", 0, test=0.6, no_skill=0.1)
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", 1, test=0.6, no_skill=0.2)
    _, runs = load(tmp_path)
    (row,) = data.setup_summary(runs, "amazon", reg.PILOT, reg.ALL)
    want = np.mean([(0.6 - 0.1) / 0.9, (0.6 - 0.2) / 0.8])
    assert row["normalised_skill"] == pytest.approx(want)


# ---- The best setup -------------------------------------------------------
def test_best_setup_goes_by_cv_and_ignores_test(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", test=0.99, cv=0.50)
    fake_run_entry(tmp_path, "amazon", "pilot_m2_own_nbr", test=0.20, cv=0.90)
    _, runs = load(tmp_path)
    best = data.best_setup(runs, "amazon", reg.ALL)
    assert best["setup_id"] == "E1" and best["cv_pr_auc"] == pytest.approx(0.90)
    assert "test set played no part" in best["rule"]


def test_best_setup_is_the_first_in_registry_order_on_a_tie(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m2_own_nbr", cv=0.8)
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", cv=0.8)
    _, runs = load(tmp_path)
    assert data.best_setup(runs, "amazon", reg.ALL)["setup_id"] == "B"


def test_best_setup_needs_a_split_0_run(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", 0, cv=0.5)
    fake_run_entry(tmp_path, "amazon", "pilot_m2_own_nbr", 1, cv=0.9)
    _, runs = load(tmp_path)
    assert data.best_setup(runs, "amazon", reg.ALL)["setup_id"] == "B"


def test_best_setup_uses_the_largest_budget_with_every_setup(tmp_path):
    # The full budget has only B, so it cannot be the comparison's budget.
    fake_run_entry(tmp_path, "amazon", "m1_own", cv=0.99)
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", cv=0.6)
    fake_run_entry(tmp_path, "amazon", "pilot_m2_own_nbr", cv=0.7)
    _, runs = load(tmp_path)
    best = data.best_setup(runs, "amazon", reg.ALL)
    assert best["budget"] == reg.PILOT and best["setup_id"] == "E1"
    # When the full budget has every setup, it is the one used.
    fake_run_entry(tmp_path, "amazon", "m2_own_nbr", cv=0.5)
    _, runs = load(tmp_path)
    best = data.best_setup(runs, "amazon", reg.ALL)
    assert best["budget"] == reg.FULL and best["setup_id"] == "B"


def test_best_setup_lists_close_setups_with_any_verdict(tmp_path):
    for k, shift in enumerate((-0.02, 0.0, 0.02)):
        fake_run_entry(tmp_path, "amazon", "pilot_m1_own", k, cv=0.80 + shift)
        fake_run_entry(tmp_path, "amazon", "pilot_m2_own_nbr", k, cv=0.81 + shift)
        fake_run_entry(tmp_path, "amazon", "pilot_m5_own_gnnssl", k, cv=0.60 + shift)
    fake_compare_entry(
        tmp_path,
        "amazon",
        [fake_contrast_row("pilot_m1_own", "pilot_m2_own_nbr", [0.01, 0.0, 0.0], p=0.5)],
        "c",
    )
    # The same pair on another dataset must not be taken for this dataset's verdict.
    fake_compare_entry(
        tmp_path,
        "yelpchi",
        [fake_contrast_row("pilot_m1_own", "pilot_m2_own_nbr", [0.5, 0.5, 0.5], p=0.001)],
        "d",
    )
    entries, runs = load(tmp_path)
    best = data.best_setup(runs, "amazon", reg.ALL, contrasts_of(entries))
    assert best["setup_id"] == "E1"
    assert [c["display_id"] for c in best["close"]] == ["B"]
    assert best["close"][0]["verdict"] == "E1 − B: not distinguishable from zero"


def test_best_setup_of_nothing_is_none(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own")
    _, runs = load(tmp_path)
    assert data.best_setup(runs, "yelpchi", reg.ALL) is None
    assert data.best_setup(runs, "amazon", reg.NO19) is None


# ---- Contrasts ------------------------------------------------------------
def test_holm_adjusts_in_order_and_stays_monotonic():
    assert data.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert data.holm([0.9, 0.9]) == pytest.approx([1.0, 1.0])
    assert data.holm([0.05]) == pytest.approx([0.05])


@pytest.mark.parametrize(
    ("verdict", "p", "n", "kind"),
    [
        ("supported", 0.01, 3, data.SUPPORTED),
        ("not distinguishable from zero", 0.4, 3, data.NOT_DISTINGUISHABLE),
        ("worse", 0.01, 3, data.WORSE_SIGNIFICANT),
        ("worse", 0.5, 3, data.WORSE_MEAN),
        ("supported", 0.01, 1, data.DESCRIPTIVE),
        ("worse", 0.01, 1, data.DESCRIPTIVE),
    ],
)
def test_verdict_kind(verdict, p, n, kind):
    assert data.verdict_kind(verdict, p, n) == kind


def test_an_unknown_verdict_is_refused():
    with pytest.raises(data.DashboardDataError):
        data.verdict_kind("probably fine", 0.01, 3)


def test_a_verdict_on_more_splits_replaces_one_on_fewer(tmp_path):
    a, b = "pilot_m1_own", "pilot_m2_own_nbr"
    fake_compare_entry(
        tmp_path,
        "amazon",
        [fake_contrast_row(a, b, [0.01, 0.02, 0.03], p=0.2)],
        "three",
        started="2026-01-01T00:00:00Z",
    )
    fake_compare_entry(
        tmp_path,
        "amazon",
        [fake_contrast_row(a, b, [0.03, 0.03, 0.03, 0.03, 0.03], p=0.001)],
        "five",
        started="2026-01-02T00:00:00Z",
    )
    # A later summary on fewer splits does not take over.
    fake_compare_entry(
        tmp_path,
        "amazon",
        [fake_contrast_row(a, b, [-0.1, -0.1, -0.1], p=0.2)],
        "later",
        started="2026-01-03T00:00:00Z",
    )
    entries, _ = load(tmp_path)
    assert len(data.all_contrasts(entries)) == 3
    (row,) = contrasts_of(entries)
    assert row["n_splits"] == 5 and row["verdict_kind"] == data.SUPPORTED
    assert sorted(r["n_splits"] for r in row["replaces"]) == [3, 3]


def test_equal_splits_take_the_latest(tmp_path):
    a, b = "pilot_m1_own", "pilot_m2_own_nbr"
    fake_compare_entry(
        tmp_path,
        "amazon",
        [fake_contrast_row(a, b, [0.1, 0.1, 0.1], p=0.01)],
        "old",
        started="2026-01-01T00:00:00Z",
    )
    fake_compare_entry(
        tmp_path,
        "amazon",
        [fake_contrast_row(a, b, [0.2, 0.2, 0.2], p=0.01)],
        "new",
        started="2026-01-02T00:00:00Z",
    )
    entries, _ = load(tmp_path)
    (row,) = contrasts_of(entries)
    assert row["mean"] == pytest.approx(0.2) and len(row["replaces"]) == 1


def test_a_replacement_only_applies_within_a_dataset(tmp_path):
    a, b = "pilot_m1_own", "pilot_m2_own_nbr"
    fake_compare_entry(
        tmp_path, "amazon", [fake_contrast_row(a, b, [0.1] * 5, p=0.01)], "a5"
    )
    fake_compare_entry(
        tmp_path, "yelpchi", [fake_contrast_row(a, b, [0.2] * 3, p=0.01)], "y3"
    )
    entries, _ = load(tmp_path)
    assert {r["dataset"]: r["n_splits"] for r in contrasts_of(entries)} == {
        "amazon": 5,
        "yelpchi": 3,
    }


def test_holm_applies_only_inside_a_primary_family(tmp_path):
    a, b = "pilot_m1_own", "pilot_m2_own_nbr"
    c = "pilot_m5_own_gnnssl"
    fake_compare_entry(
        tmp_path,
        "amazon",
        [
            fake_contrast_row(a, b, [0.1, 0.1], p=0.01),
            fake_contrast_row(a, c, [0.1, 0.1], p=0.04),
        ],
        "p",
        description="GNN plan: primary family E1-B, E2-B (pilot)",
    )
    fake_compare_entry(
        tmp_path,
        "amazon",
        [fake_contrast_row(b, c, [0.1, 0.1], p=0.04)],
        "s",
        description="Secondary contrasts",
    )
    entries, _ = load(tmp_path)
    by = {r["b_id"] + r["a_id"]: r for r in contrasts_of(entries)}
    assert by["E1B"]["family"] == data.PRIMARY
    assert by["E1B"]["holm_p"] == pytest.approx(0.02)
    assert by["E2B"]["holm_p"] == pytest.approx(0.04)
    assert by["E2E1"]["family"] == data.OTHER and by["E2E1"]["holm_p"] is None


def test_one_split_or_cross_budget_contrasts_read_as_descriptive(tmp_path):
    fake_compare_entry(
        tmp_path,
        "amazon",
        [
            fake_contrast_row("m1_own", "m2_own_nbr", [0.1], p=0.001),
            fake_contrast_row("pilot_m1_own", "m1_own", [0.1, 0.1], p=0.001),
        ],
        "full",
    )
    entries, _ = load(tmp_path)
    families = data.contrasts_for(contrasts_of(entries), "amazon", reg.ALL)
    assert len(families[data.DESCRIPTIVE]) == 2
    assert families[data.PRIMARY] == [] and families[data.OTHER] == []
    one = next(r for r in families[data.DESCRIPTIVE] if r["n_splits"] == 1)
    assert one["verdict_kind"] == data.DESCRIPTIVE and one["sd"] is None


def test_contrasts_for_filters_dataset_and_variant(tmp_path):
    fake_compare_entry(
        tmp_path,
        "amazon",
        [
            fake_contrast_row(
                "pilot_m1_own_no19", "pilot_m2_own_nbr_no19", [0.1, 0.1], p=0.01
            ),
            fake_contrast_row("pilot_m1_own", "pilot_m1_own_no19", [-0.3, -0.3], p=0.01),
        ],
        "x",
    )
    entries, _ = load(tmp_path)
    rows = contrasts_of(entries)
    assert len(data.contrasts_for(rows, "amazon", reg.NO19)[data.OTHER]) == 2
    assert len(data.contrasts_for(rows, "amazon", reg.ALL)[data.OTHER]) == 1
    assert data.contrasts_for(rows, "yelpchi", reg.ALL)[data.OTHER] == []


def test_key_questions_skip_what_was_not_run_and_other_budgets(tmp_path):
    fake_compare_entry(
        tmp_path,
        "amazon",
        [
            fake_contrast_row("pilot_m1_own", "pilot_m2_own_nbr", [0.1, 0.1], p=0.01),
            fake_contrast_row("m1_own", "m5_own_gnnssl", [0.2], p=0.01),
            fake_contrast_row(
                "pilot_m1_own_no19", "pilot_m5_own_gnnssl_no19d16", [0.2, 0.2], p=0.01
            ),
        ],
        "x",
    )
    entries, _ = load(tmp_path)
    questions = data.key_questions(contrasts_of(entries), "amazon", reg.ALL)
    assert [(q["b_id"], q["a_id"]) for q in questions] == [("E1", "B")]
    # E2 at size 16 is not the default-size question "E2 − B".
    assert data.key_questions(contrasts_of(entries), "amazon", reg.NO19) == []


def test_a_superseded_or_failed_compare_entry_gives_no_verdict(tmp_path):
    a, b = "pilot_m1_own", "pilot_m2_own_nbr"
    old = fake_compare_entry(
        tmp_path, "amazon", [fake_contrast_row(a, b, [0.1, 0.1], p=0.01)], "o"
    )
    fake_compare_entry(
        tmp_path,
        "amazon",
        [fake_contrast_row(a, b, [0.2, 0.2], p=0.01)],
        "n",
        started="2026-01-03T00:00:00Z",
        supersedes=old,
    )
    fake_compare_entry(
        tmp_path,
        "amazon",
        [fake_contrast_row(a, b, [0.9, 0.9], p=0.01)],
        "f",
        started="2026-01-04T00:00:00Z",
        status="failed",
    )
    entries, _ = load(tmp_path)
    (row,) = contrasts_of(entries)
    assert row["mean"] == pytest.approx(0.2)


# ---- Across datasets ------------------------------------------------------
def test_cross_dataset_only_has_setups_on_both_at_the_pilot_budget(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", test=0.8)
    fake_run_entry(tmp_path, "yelpchi", "pilot_m1_own", test=0.7, no_skill=0.15)
    fake_run_entry(tmp_path, "amazon", "pilot_m5r_own_gnnrand", test=0.9)  # Amazon only
    fake_run_entry(tmp_path, "yelpchi", "pilot_m2_own_nbr", test=0.9, no_skill=0.15)
    fake_run_entry(tmp_path, "amazon", "m2_own_nbr", test=0.9)  # another budget
    entries, runs = load(tmp_path)
    cross = data.cross_dataset(runs, contrasts_of(entries))
    assert [r["setup"] for r in cross["setups"]] == ["B"]
    row = cross["setups"][0]
    assert row["amazon"]["normalised_skill"] == pytest.approx((0.8 - 0.1) / 0.9)
    assert row["yelpchi"]["normalised_skill"] == pytest.approx((0.7 - 0.15) / 0.85)
    assert row["amazon_no19"] is None and cross["gains"] == []


def test_graph_gain_uses_only_the_splits_both_ran(tmp_path):
    for k, v in enumerate((0.8, 0.7, 0.6)):
        fake_run_entry(tmp_path, "amazon", "pilot_m1_own", k, test=v)
    fake_run_entry(tmp_path, "amazon", "pilot_m2_own_nbr", 1, test=0.75)
    fake_run_entry(tmp_path, "amazon", "pilot_m2_own_nbr", 2, test=0.70)
    _, runs = load(tmp_path)
    gain = data.graph_gain(runs, "amazon", reg.PILOT, reg.ALL, "E1")
    assert gain["splits"] == [1, 2] and gain["n_splits"] == 2
    assert gain["deltas"] == pytest.approx([0.05, 0.10])
    assert gain["mean"] == pytest.approx(0.075)
    assert data.graph_gain(runs, "amazon", reg.PILOT, reg.ALL, "E2") is None


def test_verdicts_agree_only_if_they_are_the_same_kind(tmp_path):
    a, b = "pilot_m1_own", "pilot_m2_own_nbr"
    c = "pilot_m5_own_gnnssl"
    fake_run_entry(tmp_path, "amazon", a)
    fake_run_entry(tmp_path, "yelpchi", a)
    fake_compare_entry(
        tmp_path,
        "amazon",
        [
            fake_contrast_row(a, b, [-0.001, -0.001], p=0.9),  # a tiny lower mean
            fake_contrast_row(a, c, [0.1, 0.1], p=0.01),
        ],
        "a",
    )
    fake_compare_entry(
        tmp_path,
        "yelpchi",
        [
            fake_contrast_row(a, b, [-0.1, -0.1], p=0.001),  # a significant drop
            fake_contrast_row(a, c, [0.2, 0.2], p=0.01),
        ],
        "y",
    )
    entries, runs = load(tmp_path)
    table = {
        r["contrast"]: r
        for r in data.cross_dataset(runs, contrasts_of(entries))["agreement"]
    }
    assert not table["E1 − B"]["agrees"]
    assert table["E1 − B"]["amazon"]["verdict_kind"] == data.WORSE_MEAN
    assert table["E1 − B"]["yelpchi"]["verdict_kind"] == data.WORSE_SIGNIFICANT
    assert table["E2 − B"]["agrees"]


def test_agreement_leaves_out_contrasts_on_one_dataset(tmp_path):
    a, b = "pilot_m1_own", "pilot_m2_own_nbr"
    fake_run_entry(tmp_path, "amazon", a)
    fake_run_entry(tmp_path, "yelpchi", a)
    fake_compare_entry(
        tmp_path, "amazon", [fake_contrast_row(a, b, [0.1, 0.1], p=0.01)], "a"
    )
    entries, runs = load(tmp_path)
    assert data.cross_dataset(runs, contrasts_of(entries))["agreement"] == []


# ---- Overview, coverage, sensitivity --------------------------------------
def test_overview_counts_and_coverage(tmp_path):
    fake_ladder_db(tmp_path)
    entries, runs = load(tmp_path)
    view = data.overview(entries, runs)
    usable = data.usable_runs(runs)
    assert (
        view["n_runs"]
        == len(usable)
        == view["budgets"][reg.PILOT] + view["budgets"][reg.FULL]
    )
    assert view["budgets"][reg.FULL] == 1 and view["n_backfilled"] == 1
    assert view["devices"] == ["fake-laptop"] and view["seeds"] == [555]
    assert view["full_budget_where"] == ["amazon split 0"]
    names = [
        "pilot_m1_own",
        "pilot_m5r_own_gnnrand",
        "pilot_m2_own_nbr_no19",
        "m2_own_nbr",
    ]
    cov = data.coverage(runs, "amazon", reg.ALL, names)
    assert cov["most_splits"] == 3 and cov["fewer_splits"] == []
    assert {"display_id": "E3", "budget": reg.PILOT} in cov["not_run"]
    assert {"display_id": "E1", "budget": reg.FULL} in cov["not_run"]
    # A variant's missing setups are not listed under another variant.
    assert all(r["display_id"] != "E1" or r["budget"] == reg.FULL for r in cov["not_run"])


def test_coverage_flags_fewer_splits(tmp_path):
    for k in range(5):
        fake_run_entry(tmp_path, "amazon", "pilot_m1_own", k)
    fake_run_entry(tmp_path, "amazon", "pilot_m2_own_nbr", 0)
    _, runs = load(tmp_path)
    cov = data.coverage(runs, "amazon", reg.ALL, [])
    assert cov["most_splits"] == 5
    assert cov["fewer_splits"] == [{"display_id": "E1", "n_splits": 1}]


def test_default_settings_check_reads_both_entries(tmp_path):
    scores = pd.DataFrame(
        {
            "setup": ["B", "B", "E1", "E1"],
            "split_repeat": [0, 1, 0, 1],
            "cv_pr_auc_default": [0.7, 0.8, 0.8, 0.9],
            "test_pr_auc_default": [0.6, 0.8, 0.9, 0.9],
            "test_pr_auc_model_a": [0.7, 0.7, 0.8, 0.8],
        }
    )
    table = pd.DataFrame(
        [
            {
                "contrast": "E1 - B",
                "test_delta_split0": 0.3,
                "test_delta_split1": 0.1,
                "test_delta_mean": 0.2,
                "test_delta_sd": 0.14,
                "p": 0.04,
                "verdict": "supported",
            }
        ]
    )
    fake_entry(tmp_path, "a", scores, kind="check", dataset="yelpchi")
    fake_entry(tmp_path, "b", table, kind="check", dataset="yelpchi")
    entries, _ = load(tmp_path)
    check = data.default_settings_check(entries, "yelpchi")
    assert [r["setup"] for r in check["scores"]] == ["B", "E1"]
    assert check["scores"][0]["test_default"] == pytest.approx(0.7)
    assert check["scores"][1]["test_tuned"] == pytest.approx(0.8)
    assert check["contrasts"][0]["contrast"] == "E1 − B"
    assert check["contrasts"][0]["verdict_kind"] == data.SUPPORTED
    assert data.default_settings_check(entries, "amazon") is None


def test_tuned_vs_default_cv(tmp_path):
    frame = pd.DataFrame(
        {
            "setup": ["B"] * 4,
            "split_repeat": [0, 0, 1, 1],
            "settings": ["tuned", "default"] * 2,
            "cv_pr_auc": [0.8, 0.7, 0.9, 0.6],
        }
    )
    fake_entry(tmp_path, "a", frame, kind="check", dataset="amazon")
    entries, _ = load(tmp_path)
    (row,) = data.tuned_vs_default_cv(entries, "amazon")
    assert row["tuned"] == pytest.approx(0.85) and row["default"] == pytest.approx(0.65)
    assert data.tuned_vs_default_cv(entries, "yelpchi") == []
