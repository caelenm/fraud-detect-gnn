"""Tabular (non-text) features shared by both models.

Per-complaint features: product, sub-product, state, submission channel, the
Older American / Servicemember tags, and year/month received.

Company features are aggregates over *training* complaints only: number of
complaints, timely-response rate, and the rate of each company-response type.
They never use labels, and a complaint's own Issue/Sub-issue is never used.
Companies with no training complaints get a count of 0 and the overall
training rates.
"""

from __future__ import annotations

import re

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


def company_statistics(sample: pd.DataFrame, train_ids: pd.Series) -> pd.DataFrame:
    """One row per company seen in training, plus a row with company NaN holding
    the overall training rates used for unseen companies."""
    train = sample[sample[C.COMPLAINT_ID].isin(train_ids)]
    if len(train) != len(train_ids):
        raise ValueError("Some training Complaint IDs are not in the sample")

    responses = sorted(train[C.COMPANY_RESPONSE].dropna().unique())
    timely = train[C.TIMELY_RESPONSE].map({"Yes": 1.0, "No": 0.0})
    frame = pd.DataFrame({C.COMPANY: train[C.COMPANY].to_numpy()})
    frame["timely"] = timely.to_numpy()
    response = train[C.COMPANY_RESPONSE]
    for r in responses:
        frame[f"resp_{slug(r)}"] = (response == r).astype(float).to_numpy()
    # Rates are over complaints with a recorded response.
    frame.loc[response.isna().to_numpy(), frame.columns[2:]] = float("nan")

    rate_cols = ["timely", *[f"resp_{slug(r)}" for r in responses]]
    per_company = frame.groupby(C.COMPANY, dropna=True)[rate_cols].mean()
    overall = frame[rate_cols].mean()
    per_company = per_company.fillna(overall)
    per_company.insert(0, "n_train", frame.groupby(C.COMPANY).size())

    names = {"n_train": "n_train_complaints", "timely": "timely_rate"}
    names |= {f"resp_{slug(r)}": f"response_rate_{slug(r)}" for r in responses}
    per_company = per_company.rename(columns=names).add_prefix(COMPANY_PREFIX)
    fallback = overall.rename(lambda k: COMPANY_PREFIX + names[k]).to_frame().T
    fallback.insert(0, COMPANY_PREFIX + "n_train_complaints", 0)
    fallback.index = pd.Index([None], name=C.COMPANY)
    return pd.concat([per_company, fallback])


def join_company_statistics(sample: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Company features for each complaint (fallback row for unseen companies)."""
    known = stats[stats.index.notna()]
    fallback = stats[stats.index.isna()].iloc[0]
    joined = known.reindex(sample[C.COMPANY].to_numpy()).reset_index(drop=True)
    unseen = joined.iloc[:, 0].isna().to_numpy()
    joined.loc[unseen, :] = fallback.to_numpy()
    count_col = COMPANY_PREFIX + "n_train_complaints"
    joined[count_col] = joined[count_col].astype("int64")
    return joined


def build_tabular_features(
    sample: pd.DataFrame, train_ids: pd.Series
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (features with complaint_id, feature-group mapping)."""
    base = complaint_features(sample)
    stats = company_statistics(sample, train_ids)
    company = join_company_statistics(sample, stats)
    features = pd.concat([base.reset_index(drop=True), company], axis=1)
    for col in features.columns:
        if C.is_label_derived(col) or col == C.LABEL:
            raise AssertionError(f"Label-derived column in features: {col}")
    groups = {**FEATURE_GROUPS_BASE, "company": list(company.columns)}
    return features, groups
