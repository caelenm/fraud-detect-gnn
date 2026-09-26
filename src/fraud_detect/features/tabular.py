"""Tabular (non-text) features shared by both models.

Per-complaint features: product, sub-product, state, submission channel, the
Older American / Servicemember tags, and year/month received. Categorical
levels with too few *training* complaints (or seen only outside training) are
merged into one level, using training counts only.

Company features describe each complaint's company using *training*
complaints only: how many there are, their timely-response rate, and the rate
of each company-response type. They never use labels or Issue/Sub-issue.

For a training complaint the statistics are leave-one-out: they are computed
from the company's *other* training complaints, so a complaint never sees its
own company response (an outcome recorded after filing). Test complaints use
all of the company's training complaints. Both therefore mean "the company's
training complaints other than this one", so train and test features are
built the same way. When no such complaints exist, the count is 0 and the
rates are the overall training rates.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from fraud_detect import columns as C

CATEGORICAL_FEATURES = ("product", "sub_product", "state", "submitted_via")
ORDINAL_FEATURES = ("year", "month")
BINARY_FEATURES = ("tag_older_american", "tag_servicemember")
COMPANY_PREFIX = "company_"

FEATURE_GROUPS_BASE = {
    "product": ["product", "sub_product"],
    "region": ["state"],
    "complaint_metadata": ["submitted_via", *BINARY_FEATURES, *ORDINAL_FEATURES],
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def complaint_features(sample: pd.DataFrame) -> pd.DataFrame:
    tags = sample[C.TAGS].fillna("")
    return pd.DataFrame(
        {
            C.COMPLAINT_ID: sample[C.COMPLAINT_ID].to_numpy(),
            "product": sample[C.PRODUCT].to_numpy(),
            "sub_product": sample[C.SUB_PRODUCT].to_numpy(),
            "state": sample[C.STATE].to_numpy(),
            "submitted_via": sample[C.SUBMITTED_VIA].to_numpy(),
            "tag_older_american": tags.str.contains("Older American").astype("int8"),
            "tag_servicemember": tags.str.contains("Servicemember").astype("int8"),
            "year": sample[C.DATE_RECEIVED].dt.year.astype("int16").to_numpy(),
            "month": sample[C.DATE_RECEIVED].dt.month.astype("int8").to_numpy(),
        }
    )


RARE_LEVEL = "__rare_or_unseen__"
COMPANY_COUNT = COMPANY_PREFIX + "n_train_complaints"


def collapse_rare_levels(
    features: pd.DataFrame, train_ids: pd.Series, columns: tuple[str, ...], min_count: int
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Merge levels with fewer than `min_count` training complaints, and levels
    that never occur in training, into RARE_LEVEL. Missing values stay missing.
    Returns the new frame and the merged levels per column."""
    out = features.copy()
    is_train = out[C.COMPLAINT_ID].isin(train_ids).to_numpy()
    merged: dict[str, list[str]] = {}
    for col in columns:
        counts = out.loc[is_train, col].value_counts()
        keep = set(counts[counts >= min_count].index)
        rare = out[col].notna() & ~out[col].isin(keep)
        merged[col] = sorted(out.loc[rare, col].astype(str).unique())
        out[col] = out[col].astype(object)
        out.loc[rare, col] = RARE_LEVEL
    return out, merged


def _outcomes(sample: pd.DataFrame, responses: list[str]) -> pd.DataFrame:
    """Per-complaint outcome indicators (NaN when the outcome is not recorded)."""
    timely = sample[C.TIMELY_RESPONSE].map({"Yes": 1.0, "No": 0.0})
    frame = pd.DataFrame({"timely_rate": timely.to_numpy(dtype=float)})
    response = sample[C.COMPANY_RESPONSE]
    for r in responses:
        values = np.array((response == r).astype(float), dtype=float)  # writable copy
        values[response.isna().to_numpy()] = np.nan  # rates over recorded responses
        frame[f"response_rate_{slug(r)}"] = values
    return frame


def _training_rows(sample: pd.DataFrame, train_ids: pd.Series) -> np.ndarray:
    is_train = sample[C.COMPLAINT_ID].isin(train_ids).to_numpy()
    if int(is_train.sum()) != len(train_ids):
        raise ValueError("Some training Complaint IDs are not in the sample")
    return is_train


def company_statistics(sample: pd.DataFrame, train_ids: pd.Series) -> pd.DataFrame:
    """Per-company aggregates over all training complaints (one row per company
    seen in training), plus a row with company NaN holding the overall training
    rates. This is what a company node will carry; per-complaint features use
    `company_features`, which is leave-one-out for training complaints."""
    is_train = _training_rows(sample, train_ids)
    train = sample[is_train]
    responses = sorted(train[C.COMPANY_RESPONSE].dropna().unique())
    frame = _outcomes(train, responses)
    frame.insert(0, C.COMPANY, train[C.COMPANY].to_numpy())
    rate_cols = list(frame.columns[1:])
    per_company = frame.groupby(C.COMPANY, dropna=True)[rate_cols].mean()
    overall = frame[rate_cols].mean()
    per_company = per_company.fillna(overall).add_prefix(COMPANY_PREFIX)
    per_company.insert(0, COMPANY_COUNT, frame.groupby(C.COMPANY).size())
    fallback = overall.add_prefix(COMPANY_PREFIX).to_frame().T
    fallback.insert(0, COMPANY_COUNT, 0)
    fallback.index = pd.Index([None], name=C.COMPANY)
    return pd.concat([per_company, fallback])


def company_features(sample: pd.DataFrame, train_ids: pd.Series) -> pd.DataFrame:
    """Company features for every complaint in `sample` (same row order):
    leave-one-out over training complaints for training rows, all training
    complaints for other rows, overall training rates when there are none."""
    is_train = _training_rows(sample, train_ids)
    train = sample[is_train]
    responses = sorted(train[C.COMPANY_RESPONSE].dropna().unique())
    outcomes = _outcomes(sample, responses)  # own outcome of every complaint
    rate_cols = list(outcomes.columns)

    train_frame = outcomes[is_train].assign(**{C.COMPANY: train[C.COMPANY].to_numpy()})
    grouped = train_frame.groupby(C.COMPANY, dropna=True)
    sums = grouped[rate_cols].sum(min_count=0)
    known = grouped[rate_cols].count()
    sizes = grouped.size()
    overall = outcomes.loc[is_train, rate_cols].mean()

    company = sample[C.COMPANY].to_numpy()
    in_stats = pd.Series(company).isin(sizes.index).to_numpy()
    row_sums = sums.reindex(company).fillna(0.0).to_numpy()
    row_known = known.reindex(company).fillna(0).to_numpy()
    row_sizes = sizes.reindex(company).fillna(0).to_numpy()

    # Remove each training complaint's own outcome (leave-one-out).
    own = outcomes[rate_cols].to_numpy()
    own_present = ~np.isnan(own) & is_train[:, None]
    loo_sums = row_sums - np.where(own_present, own, 0.0)
    loo_known = row_known - own_present
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(loo_known > 0, loo_sums / loo_known, overall.to_numpy())

    out = pd.DataFrame(rates, columns=[COMPANY_PREFIX + c for c in rate_cols])
    counts = np.where(in_stats, row_sizes - is_train, 0).astype("int64")
    out.insert(0, COMPANY_COUNT, counts)
    return out


def build_tabular_features(
    sample: pd.DataFrame, train_ids: pd.Series, min_level_count: int = 20
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    """Return (features with complaint_id, feature-group mapping, merged rare
    levels per categorical column)."""
    base = complaint_features(sample).reset_index(drop=True)
    base, merged = collapse_rare_levels(
        base, train_ids, CATEGORICAL_FEATURES, min_level_count
    )
    company = company_features(sample.reset_index(drop=True), train_ids)
    features = pd.concat([base, company], axis=1)
    for col in features.columns:
        if C.is_label_derived(col) or col == C.LABEL:
            raise AssertionError(f"Label-derived column in features: {col}")
    groups = {**FEATURE_GROUPS_BASE, "company": list(company.columns)}
    return features, groups, merged
