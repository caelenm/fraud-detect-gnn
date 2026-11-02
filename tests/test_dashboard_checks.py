"""Each sanity check must fail on the fault it exists to catch.

A check that cannot fail is decoration, so every test here breaks one thing in a
small invented database and asserts that the right check goes red, and only
that one where the fault is isolated.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from synthetic import (
    fake_compare_entry,
    fake_contrast_row,
    fake_entry,
    fake_ladder_db,
    fake_run_entry,
)

from fraud_detect.dashboard import checks, data


def run_all(db) -> dict[str, checks.Check]:
    entries = data.load_entries(db)
    runs = data.runs_table(entries)
    rows = data.all_contrasts(entries)
    results = checks.run_all(db, entries, runs, rows, data.current_contrasts(rows))
    return {c.name: c for c in results}


NAMES = {
    "db": "Results database is valid",
    "registry": "Every run maps to a setup",
    "supersedes": "Corrections point at real entries",
    "duplicates": "No run is counted twice",
    "ranges": "Metrics are in range",
    "frozen": "Each split has one test set",
    "budget": "Budget labels match the logged settings",
    "model_a": "Model A was chosen by shared CV, not by test",
    "verdicts": "Verdicts follow the pre-registered rule",
    "arithmetic": "Contrast arithmetic adds up",
    "match": "Contrasts agree with the runs",
    "replacement": "No verdict was replaced by one on fewer splits",
}


def failed(db) -> set[str]:
    return {name for name, c in run_all(db).items() if not c.ok}


def test_a_consistent_database_passes_every_check(tmp_path):
    fake_ladder_db(tmp_path)
    results = run_all(tmp_path)
    assert set(results) == set(NAMES.values())
    assert [c.name for c in results.values() if not c.ok] == []


def test_a_row_level_column_fails_the_database_check(tmp_path):
    fake_ladder_db(tmp_path)
    entry = sorted(p for p in tmp_path.iterdir() if p.is_dir())[0]
    pd.read_csv(entry / "results.csv").assign(node_id=1).to_csv(
        entry / "results.csv", index=False
    )
    assert NAMES["db"] in failed(tmp_path)


def test_an_unknown_feature_set_fails_the_registry_check(tmp_path):
    fake_ladder_db(tmp_path)
    fake_entry(
        tmp_path,
        "x",
        pd.DataFrame({"a": [1]}),
        feature_set="t100_m1_own",
        dataset="amazon",
    )
    results = run_all(tmp_path)
    assert not results[NAMES["registry"]].ok
    assert "t100_m1_own" in results[NAMES["registry"]].problems[0]


def test_a_supersedes_pointing_nowhere_fails(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", supersedes="no_such_entry")
    assert NAMES["supersedes"] in failed(tmp_path)


def test_a_duplicate_run_fails(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", tag_id="a")
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", tag_id="b")
    assert NAMES["duplicates"] in failed(tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "problem"),
    [
        ({"test": 1.5}, "test_pr_auc"),
        ({"cv": -0.1}, "cv_pr_auc"),
        ({"no_skill": 1.2}, "no_skill"),
        ({"test": 0.05, "no_skill": 0.1}, "not above no skill"),
    ],
)
def test_out_of_range_numbers_fail(tmp_path, kwargs, problem):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", **kwargs)
    result = run_all(tmp_path)[NAMES["ranges"]]
    assert not result.ok and any(problem in p for p in result.problems)


def test_a_score_outside_its_interval_fails(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", test=0.8)
    entry = next(p for p in tmp_path.iterdir() if p.is_dir())
    meta = json.loads((entry / "meta.json").read_text("utf-8"))
    meta["headline"]["test_pr_auc_low"] = 0.9
    meta["headline"]["test_pr_auc_high"] = 0.95
    (entry / "meta.json").write_text(json.dumps(meta), "utf-8")
    result = run_all(tmp_path)[NAMES["ranges"]]
    assert not result.ok and "outside" in result.problems[0]


@pytest.mark.parametrize(
    "kwargs", [{"no_skill": 0.2}, {"n_test": 99}], ids=["base rate", "test size"]
)
def test_setups_with_different_test_sets_on_one_split_fail(tmp_path, kwargs):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", 0)
    fake_run_entry(tmp_path, "amazon", "pilot_m2_own_nbr", 0, **kwargs)
    assert NAMES["frozen"] in failed(tmp_path)


def test_different_splits_or_datasets_may_differ(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", 0, no_skill=0.1)
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", 1, no_skill=0.11)
    fake_run_entry(tmp_path, "yelpchi", "pilot_m1_own", 0, no_skill=0.15)
    assert NAMES["frozen"] not in failed(tmp_path)


@pytest.mark.parametrize(
    ("feature_set", "settings"),
    [
        (
            "pilot_m1_own",
            {"df_analyze.htune_trials": 100, "df_analyze.classifiers": list("abc")},
        ),
        (
            "pilot_m1_own",
            {"df_analyze.htune_trials": 10, "df_analyze.classifiers": list("abcdef")},
        ),
        (
            "m1_own",
            {"df_analyze.htune_trials": 10, "df_analyze.classifiers": list("abc")},
        ),
    ],
)
def test_a_budget_label_that_disagrees_with_the_settings_fails(
    tmp_path, feature_set, settings
):
    fake_run_entry(tmp_path, "amazon", feature_set, settings=settings)
    assert NAMES["budget"] in failed(tmp_path)


def test_gnn_rungs_are_not_held_to_df_analyze_settings(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_g5_sage", settings={})
    assert NAMES["budget"] not in failed(tmp_path)


def test_a_model_a_that_is_not_the_cv_best_fails(tmp_path):
    table = pd.DataFrame(
        {
            "model": ["lgbm", "catboost"],
            "selection": ["none", "none"],
            "cv_pr_auc": [0.7, 0.9],
            "pr_auc": [0.9, 0.7],
        }
    )
    # Chosen by its test score: exactly what invariant 8 forbids.
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", model="lgbm", table=table)
    result = run_all(tmp_path)[NAMES["model_a"]]
    assert not result.ok and "lgbm (none)" in result.problems[0]


def test_a_tie_for_the_best_cv_score_is_fine(tmp_path):
    table = pd.DataFrame(
        {
            "model": ["lgbm", "catboost"],
            "selection": ["none", "none"],
            "cv_pr_auc": [0.9, 0.9],
            "pr_auc": [0.9, 0.7],
        }
    )
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", model="catboost", table=table)
    assert NAMES["model_a"] not in failed(tmp_path)


def test_a_verdict_the_rule_does_not_give_fails(tmp_path):
    row = fake_contrast_row(
        "pilot_m1_own", "pilot_m2_own_nbr", [0.1, -0.1, 0.1], p=0.5, verdict="supported"
    )
    fake_compare_entry(tmp_path, "amazon", [row], "x")
    result = run_all(tmp_path)[NAMES["verdicts"]]
    assert not result.ok and "rule gives" in result.problems[0]


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"test_delta_mean": 0.5}, "mean"),
        ({"test_delta_sd": 0.5}, "SD"),
        ({"test_pr_auc_b_mean": 0.1}, "two sides"),
        ({"p": 1.5}, "p ="),
        ({"cv_folds": 7}, "CV folds"),
    ],
)
def test_contrast_arithmetic_faults_fail(tmp_path, overrides, problem):
    row = fake_contrast_row(
        "pilot_m1_own", "pilot_m2_own_nbr", [0.1, 0.2, 0.3], **{"p": 0.01, **overrides}
    )
    fake_compare_entry(tmp_path, "amazon", [row], "x")
    result = run_all(tmp_path)[NAMES["arithmetic"]]
    assert not result.ok and any(problem in p for p in result.problems)


def test_a_contrast_that_disagrees_with_its_runs_fails(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", 0, test=0.8)
    fake_run_entry(tmp_path, "amazon", "pilot_m2_own_nbr", 0, test=0.9)
    row = fake_contrast_row("pilot_m1_own", "pilot_m2_own_nbr", [0.05, 0.05], p=0.01)
    fake_compare_entry(tmp_path, "amazon", [row], "x")
    result = run_all(tmp_path)[NAMES["match"]]
    assert not result.ok and "split 0" in result.problems[0]
    assert "could not be checked" not in result.detail


def test_a_contrast_whose_runs_are_missing_says_so_rather_than_passing_silently(tmp_path):
    row = fake_contrast_row("pilot_m1_own", "pilot_m2_own_nbr", [0.05, 0.05], p=0.01)
    fake_compare_entry(tmp_path, "amazon", [row], "x")
    result = run_all(tmp_path)[NAMES["match"]]
    assert result.ok and "2 could not be checked" in result.detail


def test_a_verdict_replaced_by_fewer_splits_fails():
    current = [{"b_id": "E1", "a_id": "B", "n_splits": 3, "replaces": [{"n_splits": 5}]}]
    result = checks.check_replacement(current)
    assert not result.ok and "5 splits replaced by 3" in result.problems[0]


def test_check_as_dict_caps_the_problems_it_shows():
    check = checks.Check("x", False, "many", tuple(f"p{i}" for i in range(30)))
    shown = check.as_dict()
    assert len(shown["problems"]) == checks.SHOWN_PROBLEMS and shown["n_problems"] == 30
