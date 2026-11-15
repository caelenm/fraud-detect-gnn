"""The dashboard's sentences, on small invented databases: each must follow the
database it is built from, so it cannot go stale as runs are added."""

from __future__ import annotations

from synthetic import fake_run_entry

from fraud_detect.dashboard import data, narrative
from fraud_detect.dashboard import registry as reg


def load(db):
    entries = data.load_entries(db)
    return entries, data.runs_table(entries)


def test_split_list():
    assert narrative.split_list([0, 1, 2]) == "0–2"
    assert narrative.split_list([0, 2]) == "0 and 2"
    assert narrative.split_list([0, 2, 4]) == "0, 2 and 4"
    assert narrative.split_list([3]) == "3"


def test_one_machine_is_said_only_when_there_is_one(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own")
    entries, runs = load(tmp_path)
    one = narrative.machines_note(data.overview(entries, runs))
    assert one.startswith("Every run used one machine (fake-laptop)")
    fake_run_entry(tmp_path, "amazon", "m1_own", device="fake-desktop")
    entries, runs = load(tmp_path)
    two = narrative.machines_note(data.overview(entries, runs))
    assert "one machine" not in two
    assert "2 machines" in two and "fake-desktop (1 full)" in two
    assert "fake-laptop (1 pilot)" in two and "not give bit-identical" in two


def test_budget_note_names_the_full_budget_splits(tmp_path):
    for k in range(3):
        fake_run_entry(tmp_path, "amazon", "m1_own", k)
    fake_run_entry(tmp_path, "yelpchi", "m1_own", 0, no_skill=0.15)
    fake_run_entry(tmp_path, "yelpchi", "pilot_m1_own", 0, no_skill=0.15)
    entries, runs = load(tmp_path)
    note = narrative.budget_note(data.overview(entries, runs))
    assert "1 of the 5 recorded runs use the pilot budget" in note
    assert "4 use the full budget" in note
    assert "Amazon splits 0–2 and YelpChi split 0" in note


def test_not_run_is_derived_from_the_database(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own")
    fake_run_entry(tmp_path, "amazon", "pilot_m2_own_nbr")
    fake_run_entry(tmp_path, "amazon", "m1_own")
    fake_run_entry(tmp_path, "yelpchi", "pilot_m1_own", no_skill=0.15)
    _, runs = load(tmp_path)
    items = narrative.not_run(runs)
    assert "No full-budget run, only the pilot: Amazon, all 25 features: E1." in items
    assert "Pilot budget only, for every setup: YelpChi, all 32 features." in items
    assert all(s in items for s in narrative.NOT_RUN_STATIC)
    # Once the gap is closed, the sentence goes.
    fake_run_entry(tmp_path, "amazon", "m2_own_nbr")
    fake_run_entry(tmp_path, "yelpchi", "m1_own", no_skill=0.15)
    _, runs = load(tmp_path)
    assert not any(
        "only the pilot" in s or "every setup" in s for s in narrative.not_run(runs)
    )


def test_the_no19_note_follows_the_column_map():
    (note,) = narrative.dataset_caveats("amazon", reg.NO19, [], [], None)
    assert "no column built from votes is left" in note
    assert "may still be vote-derived" not in note


def _b(test, n=3):
    return {"display_id": "B", "test_pr_auc": test, "n_splits": n}


def test_the_f19_warning_says_its_budget_and_view():
    note = narrative.amazon_f19([_b(0.916)], [_b(0.546)], reg.FULL, reg.ALL)
    assert "0.916" in note and "0.546" in note and "full budget" in note
    assert "and so does this view" in note
    top4 = narrative.amazon_f19([_b(0.9)], [_b(0.5)], reg.PILOT, reg.TOP4)
    assert "so does this view" not in top4 and "keeps f19 too" in top4


def test_the_default_settings_note_adds_the_full_budget_order():
    check = {
        "scores": [
            {"setup": "B", "test_tuned": 0.8, "test_default": 0.7, "n_splits": 3},
            {"setup": "E1", "test_tuned": 0.9, "test_default": 0.6, "n_splits": 3},
        ]
    }
    full = [
        {"display_id": "B", "test_pr_auc": 0.85, "n_splits": 3},
        {"display_id": "E1", "test_pr_auc": 0.84, "n_splits": 2},
    ]
    (note,) = narrative.dataset_caveats("amazon", reg.NO19, [], [], check, full_all=full)[
        1:
    ]
    assert "E1 > B" in note and "B > E1" in note
    assert (
        "At the full budget (6 classifiers, 100 trials): B > E1 (3 splits; E1 on 2)"
        in note
    )
    # Without a full-budget run of every setup, the sentence is left out.
    (short,) = narrative.dataset_caveats(
        "amazon", reg.NO19, [], [], check, full_all=full[:1]
    )[1:]
    assert "full budget" not in short
