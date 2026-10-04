"""Label shortcut audit: fraud rate by product name x year (synthetic data)."""

from __future__ import annotations

import pandas as pd
import pytest
from synthetic import labeled_sample, renamed_product_complaints
from test_runstate import tmp_config

from fraud_detect import columns as C
from fraud_detect.cli import main
from fraud_detect.config import get_paths
from fraud_detect.data import audit
from fraud_detect.pipeline import LABEL_AUDIT_REPORT, LABEL_AUDIT_TABLES, STAGE_NAMES


def _row(table: pd.DataFrame, product: str, year: int) -> pd.Series:
    match = table[(table[C.PRODUCT] == product) & (table["year"] == year)]
    assert len(match) == 1
    return match.iloc[0]


def test_product_year_rates_by_hand():
    table = audit.product_year(renamed_product_complaints())
    old_2019 = _row(table, "Synthetic product OLD", 2019)
    assert (old_2019["n_complaints"], old_2019["n_fraud"]) == (4, 2)
    assert old_2019["fraud_rate"] == pytest.approx(0.5)
    assert _row(table, "Synthetic product NEW", 2021)["fraud_rate"] == 0.0
    # The all-products row pools every product in that year.
    all_2021 = _row(table, audit.ALL_PRODUCTS, 2021)
    assert (all_2021["n_complaints"], all_2021["n_fraud"]) == (8, 1)
    assert table[table[C.PRODUCT] != audit.ALL_PRODUCTS]["n_complaints"].sum() == 24


def test_renamed_product_is_visible_in_span_and_grid():
    df = renamed_product_complaints()
    span = audit.product_span(df).set_index(C.PRODUCT)
    assert span.loc["Synthetic product OLD", "first_month"] == "2019-03"
    assert span.loc["Synthetic product OLD", "last_month"] == "2020-03"
    assert span.loc["Synthetic product NEW", "first_month"] == "2021-05"
    grid = audit.pivot_markdown(audit.product_year(df))
    old_line = next(line for line in grid.splitlines() if "product OLD" in line)
    assert old_line.endswith("| 0.500 (4) | 0.500 (4) | – |")  # absent in 2021
    assert grid.splitlines()[-1].startswith(f"| **{audit.ALL_PRODUCTS}** |")


def test_missing_sub_product_is_its_own_level():
    table = audit.sub_product_year(renamed_product_complaints())
    assert audit.NONE_LABEL in set(table[C.SUB_PRODUCT])
    assert table["n_complaints"].sum() == 24


def test_label_audit_runs_last_so_it_never_forces_long_stages_to_rerun():
    assert STAGE_NAMES[-1] == "label_audit"


def test_stage_writes_aggregates_only(tmp_path, capsys):
    path, config = tmp_config(tmp_path)
    paths = get_paths(config)
    labeled = labeled_sample(n=300, seed=1)
    for target, frame in ((paths.labeled, labeled), (paths.sample, labeled.iloc[:200])):
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)

    assert main(["--config", str(path), "--only", "label_audit"]) == 0
    report = (paths.reports_dir / LABEL_AUDIT_REPORT).read_text(encoding="utf-8")
    assert "## All labelled complaints" in report
    assert "## The sample used by the models" in report
    by_year = pd.read_csv(paths.reports_dir / LABEL_AUDIT_TABLES[0])
    assert set(by_year["dataset"]) == {"labelled", "sample"}
    labelled_rows = by_year[(by_year["dataset"] == "labelled")
                            & (by_year[C.PRODUCT] != audit.ALL_PRODUCTS)]  # fmt: skip
    assert labelled_rows["n_complaints"].sum() == 300
    # Counts and rates only: no narratives, companies or IDs in any output.
    for name in (LABEL_AUDIT_REPORT, *LABEL_AUDIT_TABLES):
        text = (paths.reports_dir / name).read_text(encoding="utf-8").lower()
        assert "narrative" not in text.replace("narratives", "")
        assert "synthetic-company" not in text
        assert "5000" not in text  # first synthetic Complaint ID
    assert not list(paths.reports_dir.glob("*.tmp"))
