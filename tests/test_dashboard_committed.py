"""The dashboard against the committed results database (aggregate numbers only).

The acceptance criterion of the data layer: its means reproduce the published
results pages to 3 decimals, from a different code path (the database, not the
pipeline's outputs). If a later commit records a run that moves a mean, these
fail until the pages are regenerated, which is the point.
"""

from __future__ import annotations

import importlib.util
import re
import sys

import numpy as np
import pytest

from fraud_detect.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from fraud_detect.dashboard import checks, data, narrative
from fraud_detect.dashboard import registry as reg

DB = REPO_ROOT / "results" / "runs"


@pytest.fixture(scope="module")
def db():
    entries = data.load_entries(DB)
    runs = data.runs_table(entries)
    rows = data.all_contrasts(entries)
    return entries, runs, rows, data.current_contrasts(rows)


def test_every_sanity_check_passes_on_the_committed_database(db):
    entries, runs, rows, current = db
    results = checks.run_all(DB, entries, runs, rows, current)
    assert [(c.name, c.problems[:3]) for c in results if not c.ok] == []
    match = next(c for c in results if c.name == "Contrasts agree with the runs")
    assert "could not be checked" not in match.detail  # the backfill closed the gaps


def table_rows(text: str, heading: str) -> list[list[str]]:
    """The cells of the first Markdown table after `heading`."""
    body = text[text.index(heading) :]
    lines = []
    for line in body.splitlines()[1:]:
        if line.startswith("|"):
            lines.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif lines:
            break
    return lines[2:]  # past the header and the rule


SCORE = re.compile(r"([\d.]+)(?: ± ([\d.]+))?")


@pytest.mark.parametrize(
    ("dataset", "heading"),
    [("amazon", "## Amazon (main"), ("yelpchi", "## YelpChi (validation")],
)
def test_means_reproduce_gnn_results(db, dataset, heading):
    _, runs, _, _ = db
    text = (REPO_ROOT / "docs" / "GNN_RESULTS.md").read_text("utf-8")
    summary = {
        r["display_id"]: r for r in data.setup_summary(runs, dataset, reg.PILOT, reg.ALL)
    }
    rows = table_rows(text, heading)
    assert len(rows) >= 8
    for setup, what, splits, test, cv, no_skill, _model in rows:
        row = summary[setup]
        assert row["label"] == what
        assert row["n_splits"] == int(splits), setup
        mean, sd = SCORE.fullmatch(test).groups()
        assert f"{row['test_pr_auc']:.3f}" == mean, setup
        assert f"{row['test_pr_auc_sd']:.3f}" == sd, setup
        assert f"{row['cv_pr_auc']:.3f}" == cv, setup
        assert f"{row['no_skill']:.3f}" == no_skill, setup


@pytest.mark.parametrize(
    ("dataset", "heading", "n_splits"),
    [("amazon", "## Amazon (main", 5), ("yelpchi", "## YelpChi (validation", 3)],
)
def test_primary_verdicts_and_holm_reproduce_gnn_results(db, dataset, heading, n_splits):
    _, _, _, current = db
    text = (REPO_ROOT / "docs" / "GNN_RESULTS.md").read_text("utf-8")
    section = text[text.index(heading) :]
    rows = table_rows(section, "### Primary contrasts")
    families = data.contrasts_for(current, dataset, reg.ALL)
    primary = {f"{r['b_id']} − {r['a_id']}": r for r in families[data.PRIMARY]}
    assert len(primary) == len(rows) > 0
    for cells in rows:
        contrast, *deltas = cells[: n_splits + 1]
        mean_sd, _headroom, p, holm_p, verdict = cells[n_splits + 1 :]
        row = primary[contrast]
        assert [f"{d:+.3f}" for d in row["deltas"]] == [
            d.replace("−", "-") for d in deltas
        ]
        assert f"{row['mean']:+.3f} ± {row['sd']:.3f}" == mean_sd
        assert f"{row['p']:.2g}" == p
        assert f"{row['holm_p']:.2g}" == holm_p
        assert row["verdict"] == verdict


def test_means_reproduce_rq_results(db):
    """RQ_RESULTS.md was written over 3 splits unless a cell says otherwise, so
    each cell is compared over its own splits."""
    _, runs, _, _ = db
    text = (REPO_ROOT / "docs" / "RQ_RESULTS.md").read_text("utf-8")
    rows = table_rows(text, "## 1. Every model setup")
    columns = {
        "all 25": reg.ALL,
        "without #36": reg.NO36,
        "without f19": reg.NO19,
        "top 4 without f19": reg.TOP4NO19,
        "#36 four only": reg.ONLY36,
        "top 4 (with f19)": reg.TOP4,
    }
    order = list(columns)
    summary = {
        v: {r["display_id"]: r for r in data.setup_summary(runs, "amazon", reg.PILOT, v)}
        for v in columns.values()
    }
    compared = 0
    cell_pattern = re.compile(r"([\d.]+) ± ([\d.]+)(?: \((\d) splits\))?")
    for setup, *cells in rows:
        for column, cell in zip(order, cells, strict=True):
            if not cell:
                continue
            mean, sd, n = cell_pattern.fullmatch(cell).groups()
            n = int(n or 3)
            row = summary[columns[column]][setup]
            values = dict(zip(row["splits"], row["test_per_split"], strict=True))
            subset = [values[k] for k in range(n)]
            assert f"{np.mean(subset):.3f}" == mean, (setup, column)
            assert f"{np.std(subset, ddof=1):.3f}" == sd, (setup, column)
            compared += 1
    assert compared >= 30


def test_means_reproduce_the_readme_table(db):
    _, runs, _, _ = db
    text = (REPO_ROOT / "README.md").read_text("utf-8")
    rows = table_rows(text, "## Results (GNN round")
    assert len(rows) == 7
    amazon = {
        r["display_id"]: r for r in data.setup_summary(runs, "amazon", reg.PILOT, reg.ALL)
    }
    yelp = {
        r["display_id"]: r
        for r in data.setup_summary(runs, "yelpchi", reg.PILOT, reg.ALL)
    }
    for setup, what, am, ye in rows:
        assert amazon[setup]["label"] == what
        assert (
            f"{amazon[setup]['test_pr_auc']:.3f} ± {amazon[setup]['test_pr_auc_sd']:.3f}"
            == am
        )
        assert (
            f"{yelp[setup]['test_pr_auc']:.3f} ± {yelp[setup]['test_pr_auc_sd']:.3f}"
            == ye
        )


def test_the_best_setups_by_the_agreed_rule(db):
    _, runs, _, current = db
    amazon = data.best_setup(runs, "amazon", reg.ALL, current)
    assert amazon["setup_id"] == "E7r" and amazon["budget"] == reg.PILOT
    assert {c["display_id"] for c in amazon["close"]} >= {"E3", "E7"}
    assert data.best_setup(runs, "amazon", reg.NO19, current)["setup_id"] == "E1"
    yelp = data.best_setup(runs, "yelpchi", reg.ALL, current)
    assert yelp["setup_id"] == "E2" and yelp["model_a_split0"] == "lgbm (assoc)"
    assert yelp["n_splits"] == 3


def test_no_setup_is_chosen_by_test_where_cv_disagrees(db):
    # On Amazon E7 has the best mean test score, and it is still not chosen.
    _, runs, _, _ = db
    rows = data.setup_summary(runs, "amazon", reg.PILOT, reg.ALL)
    best_test = max(rows, key=lambda r: r["test_pr_auc"])["display_id"]
    assert best_test == "E7"
    assert data.best_setup(runs, "amazon", reg.ALL)["display_id"] != best_test


def test_every_early_amazon_run_is_now_in_the_database(db):
    _, runs, _, _ = db
    usable = data.usable_runs(runs)
    for name in ("pilot_m1_own", "pilot_m2_own_nbr", "pilot_m5_own_gnnssl"):
        got = usable[(usable["feature_set"] == name) & (usable["dataset"] == "amazon")]
        assert sorted(got["split_repeat"]) == [0, 1, 2, 3, 4], name
    for name in ("m1_own", "m1_own_no36"):
        got = usable[usable["feature_set"] == name]
        assert len(got) == 1 and got.iloc[0]["backfilled"], name


def test_the_stopped_100_trial_check_is_on_record_and_counts_for_nothing(db):
    entries, runs, _, _ = db
    failed = data.failed_entries(entries)
    assert any("up to 100 trials" in f["description"] for f in failed)
    assert not runs["feature_set"].str.contains("t100").any()
    view = data.overview(entries, runs)
    notes = narrative.general_caveats(view, failed)
    assert any(n.startswith("Not finished:") and "0 of 18" in n for n in notes)
    assert not any("t100" in n for n in load_config(DEFAULT_CONFIG)["feature_sets"])


def test_the_page_says_what_the_numbers_are(db):
    entries, runs, _, _ = db
    view = data.overview(entries, runs)
    note = narrative.budget_note(view)
    assert f"{view['budgets'][reg.PILOT]} of the {view['n_runs']}" in note
    assert "amazon split 0" in note
    amazon_all = data.setup_summary(runs, "amazon", reg.PILOT, reg.ALL)
    no19 = data.setup_summary(runs, "amazon", reg.PILOT, reg.NO19)
    warning = narrative.amazon_f19(amazon_all, no19)
    assert "likely leaks" in warning and "0.922" in warning and "0.544" in warning


def test_the_default_settings_order_is_reported(db):
    entries, _, _, _ = db
    check = data.default_settings_check(entries, "yelpchi")
    notes = narrative.dataset_caveats("yelpchi", reg.ALL, [], [], check)
    joined = " ".join(notes)
    assert "E2 > E1 > E7 > B > E8" in joined and "E8 > E1 > E7 > E2 > B" in joined


def test_the_command_line_check_passes():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "dashboard_script", REPO_ROOT / "scripts" / "dashboard.py"
        )
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
    finally:
        sys.path.pop(0)
    assert script.main(["--check", "--no-model"]) == 0
