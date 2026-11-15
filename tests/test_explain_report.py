"""The explanation page and tables, from invented shap_explain.py outputs."""

from __future__ import annotations

import pandas as pd
import pytest
from synthetic import fake_explain_outputs

from fraud_detect import results_db as db
from fraud_detect.report import explain as report


def _summary(tmp_path):
    folder, feature_set_json, edges = fake_explain_outputs(tmp_path)
    return folder, report.summarise(folder, feature_set_json, ["own__f00"], edges, seed=1)


def test_block_importance_has_both_measures(tmp_path) -> None:
    _, summary = _summary(tmp_path)
    assert set(summary.groups["measure"]) == {"summed SHAP", "group Shapley"}
    assert set(summary.groups["name"]) == {"own", "nbr_upu", "degree"}
    assert list(summary.correlated["name"])[0].startswith("#36 group")


def test_refit_recurrence_counts_fits_and_block_shares(tmp_path) -> None:
    _, summary = _summary(tmp_path)
    recurrence = summary.tables["top10_recurrence"]
    # With 4 columns and a top 10, every column is in the top of both fits.
    assert (recurrence["fits_in_top10"] == 2).all() and (recurrence["n_fits"] == 2).all()
    shares = summary.tables["block_share_across_fits"].set_index("name")
    assert shares.loc["own", "min_share"] == pytest.approx(0.4)
    assert shares.loc["own", "max_share"] == pytest.approx(0.7)


def test_roar_section_only_claims_separation_beyond_the_spread(tmp_path) -> None:
    _, summary = _summary(tmp_path)
    summary.tables["roar_top_k_vs_random"]["gap"] = 0.0
    assert "did not separate clearly" in report.render_page(summary)


def test_page_is_self_contained_and_escapes_values(tmp_path) -> None:
    folder, summary = _summary(tmp_path)
    summary.importance.loc[0, "name"] = '<script>alert("x")</script>'
    summary.meta["feature_set"] = "<b>set</b>"
    html = report.render_page(summary)
    assert "<script" not in html
    assert "&lt;script&gt;" in html and "&lt;b&gt;set&lt;/b&gt;" in html
    assert "http://" not in html and "https://" not in html
    assert " src=" not in html and "<link" not in html
    assert "for the group only" in html


def test_page_wording_does_not_overclaim(tmp_path) -> None:
    _, summary = _summary(tmp_path)
    text = report.render_page(summary).lower()
    assert "faithful by construction" not in text
    assert "implementation check" in text
    assert "necessity-style" in text
    assert "no model in this ladder uses attention" in text
    assert "not the cause of fraud" in text


def test_db_summary_passes_the_results_database_rules(tmp_path) -> None:
    folder, summary = _summary(tmp_path)
    path = report.write_tables(summary, folder)
    table = pd.read_csv(path)
    assert len(table) <= db.MAX_ROWS
    assert not db.FORBIDDEN_COLUMNS & set(table.columns)
    # Node IDs stay out: the table of top neighbours is not part of it.
    assert "neighbour" not in table.columns
    assert "explain" in db.KINDS


def test_neighbour_tables_split_the_mean_and_degree_parts(tmp_path) -> None:
    _, summary = _summary(tmp_path)
    relations = summary.tables["neighbour_relations"].set_index("relation")
    assert relations.loc["upu", "mean_abs_mean_part"] > 0
    assert 0 <= relations.loc["upu", "share_out_of_range"] <= 1
    assert "neighbour_structure" in summary.tables
    assert "neighbour_redundancy_summary" in summary.tables


def test_encoder_influence_page_says_what_it_held_fixed(tmp_path) -> None:
    _, summary = _summary(tmp_path)
    summary.meta.setdefault("sections", {})["neighbours_encoder"] = {
        "max_neighbours_per_target": 30
    }
    html = report.render_page(summary)
    assert "through the DGI encoder, approximate" in html
    assert "encoder and PCA held fixed" in html
    assert "Mean part" not in html and "under mean aggregation</strong>" not in html


def test_missing_outputs_are_reported(tmp_path) -> None:
    try:
        report.read_meta(tmp_path)
    except report.ExplainReportError as e:
        assert "shap_explain" in str(e)
    else:
        raise AssertionError("expected an ExplainReportError")


def test_split_agreement_compares_each_pair() -> None:
    a = pd.Series({"p": 3.0, "q": 2.0, "r": 1.0})
    frames = {"split 0": a, "split 1": a, "split 2": a[::-1]}
    table = report.split_agreement(frames)
    assert len(table) == 3
    assert (table["kendall_tau"] == 1.0).all()
