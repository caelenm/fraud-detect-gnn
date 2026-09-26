"""Label rule: expected labels on hand-written examples, and the review gate."""

from __future__ import annotations

import copy

import pandas as pd
import pytest
from synthetic import CATEGORY_EXAMPLES, SYNTHETIC_CATEGORIES, complaints_with_categories

from fraud_detect import columns as C
from fraud_detect.data.label import (
    CategoryConfig,
    LabelConfigError,
    apply_labels,
    candidate_report,
    is_candidate,
    validate_categories,
)


def config(**overrides) -> CategoryConfig:
    data = copy.deepcopy(SYNTHETIC_CATEGORIES) | overrides
    return CategoryConfig.from_dict(data)


def test_labels_match_hand_written_examples():
    df = complaints_with_categories()
    cfg = config()
    validate_categories(df, cfg)
    labeled = apply_labels(df, cfg)

    expected = {
        1000 + i: lab
        for i, (product, _, _, lab) in enumerate(CATEGORY_EXAMPLES)
        if product in cfg.target_products
    }
    got = dict(zip(labeled[C.COMPLAINT_ID], labeled[C.LABEL], strict=True))
    assert got == expected


def test_apply_labels_drops_issue_columns_and_other_products():
    labeled = apply_labels(complaints_with_categories(), config())
    assert C.ISSUE not in labeled.columns
    assert C.SUB_ISSUE not in labeled.columns
    assert not any(C.is_label_derived(c) for c in labeled.columns)
    assert "Credit reporting" not in set(labeled[C.PRODUCT])


def test_unconfirmed_config_is_rejected():
    with pytest.raises(LabelConfigError, match="not confirmed"):
        validate_categories(complaints_with_categories(), config(confirmed=False))


def test_entry_matching_nothing_is_rejected():
    cfg = copy.deepcopy(SYNTHETIC_CATEGORIES)
    cfg["positive"].append({"issue": "Fraud or scma"})  # typo
    with pytest.raises(LabelConfigError, match="matches no complaints"):
        validate_categories(complaints_with_categories(), CategoryConfig.from_dict(cfg))


def test_missing_target_product_is_rejected():
    cfg = copy.deepcopy(SYNTHETIC_CATEGORIES)
    cfg["target_products"].append("Synthetic product that does not exist")
    with pytest.raises(LabelConfigError, match="target_products not found"):
        validate_categories(complaints_with_categories(), CategoryConfig.from_dict(cfg))


def test_unreviewed_keyword_candidate_is_rejected():
    cfg = copy.deepcopy(SYNTHETIC_CATEGORIES)
    cfg["positive"] = [e for e in cfg["positive"] if e["issue"] != "Fraud or scam"]
    with pytest.raises(LabelConfigError, match="not classified"):
        validate_categories(complaints_with_categories(), CategoryConfig.from_dict(cfg))


def test_overlapping_positive_and_negative_is_rejected():
    cfg = copy.deepcopy(SYNTHETIC_CATEGORIES)
    cfg["reviewed_negative"].append({"issue": "Fraud or scam"})
    with pytest.raises(LabelConfigError, match="both positive and reviewed_negative"):
        validate_categories(complaints_with_categories(), CategoryConfig.from_dict(cfg))


def test_null_sub_issue_matches_only_missing():
    df = complaints_with_categories()
    cfg = copy.deepcopy(SYNTHETIC_CATEGORIES)
    cfg["positive"].append({"issue": "Managing an account", "sub_issue": None})
    labeled = apply_labels(df, CategoryConfig.from_dict(cfg))
    managing = df.loc[df[C.ISSUE] == "Managing an account", C.COMPLAINT_ID]
    assert labeled.set_index(C.COMPLAINT_ID).loc[managing, C.LABEL].eq(1).all()


def test_candidate_flags():
    df = pd.DataFrame(
        {
            C.ISSUE: ["Fraud or scam", "Fee problem", "Managing an account"],
            C.SUB_ISSUE: [None, None, "Transaction was not authorized"],
        }
    )
    assert is_candidate(df).tolist() == [True, False, True]


def test_candidate_report_counts_every_combination():
    df = complaints_with_categories()
    report = candidate_report(df, config())
    assert report["n_complaints"].sum() == len(df)
    assert (report[C.SUB_ISSUE] == "<missing>").sum() == 2
