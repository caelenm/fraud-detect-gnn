"""Leakage invariants for features: no Issue/Sub-issue, company statistics from
training rows only, PCA fit on training rows only, no label-derived feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import labeled_sample

from fraud_detect import columns as C
from fraud_detect.data.sample import ids_for, stratified_split
from fraud_detect.features.tabular import build_tabular_features, company_statistics
from fraud_detect.features.text import attach_ids_to_embeddings, fit_pca
from fraud_detect.models.df_analyze import build_tables


def sample_and_split(n=300, seed=0):
    df = labeled_sample(n=n, seed=seed)
    split = stratified_split(df, 0.4, seed=seed)
    return df, ids_for(split, "train"), ids_for(split, "test")


def fake_embeddings(df, width=16, seed=0):
    rng = np.random.default_rng(seed)
    emb = pd.DataFrame(
        rng.normal(size=(len(df), width)), columns=[f"embed{i:04d}" for i in range(width)]
    )
    emb.insert(0, C.COMPLAINT_ID, df[C.COMPLAINT_ID].to_numpy())
    return emb


def test_features_have_no_issue_or_label_columns():
    df, train_ids, test_ids = sample_and_split()
    df = df.assign(**{C.ISSUE: "Fraud or scam", C.SUB_ISSUE: "Synthetic"})
    features, groups, _ = build_tabular_features(df, train_ids)
    pcs, _ = fit_pca(fake_embeddings(df), train_ids, n_components=4, seed=0)
    train, test = build_tables(features, pcs, df, train_ids, test_ids)
    for table in (features, train, test):
        assert not any(C.is_label_derived(c) for c in table.columns)
        assert C.COMPANY not in table.columns
        assert C.COMPLAINT_ID not in table.columns or table is features
        assert C.LABEL not in table.columns
    grouped = {c for cols in groups.values() for c in cols}
    assert grouped == set(features.columns) - {C.COMPLAINT_ID}


def test_company_stats_ignore_test_rows():
    df, train_ids, test_ids = sample_and_split()
    before = company_statistics(df, train_ids)

    changed = df.copy()
    is_test = changed[C.COMPLAINT_ID].isin(test_ids)
    changed.loc[is_test, C.COMPANY_RESPONSE] = "Closed with monetary relief"
    changed.loc[is_test, C.TIMELY_RESPONSE] = "No"
    changed.loc[is_test, C.COMPANY] = "SYNTHETIC-COMPANY-A"
    pd.testing.assert_frame_equal(before, company_statistics(changed, train_ids))
    pd.testing.assert_frame_equal(before, company_statistics(df[~is_test], train_ids))


def test_company_stats_do_not_depend_on_labels():
    df, train_ids, _ = sample_and_split()
    flipped = df.assign(**{C.LABEL: 1 - df[C.LABEL]})
    pd.testing.assert_frame_equal(
        company_statistics(df, train_ids), company_statistics(flipped, train_ids)
    )


def test_unseen_company_gets_training_fallback():
    df, train_ids, test_ids = sample_and_split()
    df = df.copy()
    first_test = df[C.COMPLAINT_ID].isin(test_ids).idxmax()
    df.loc[first_test, C.COMPANY] = "SYNTHETIC-COMPANY-ONLY-IN-TEST"
    features, _, _ = build_tabular_features(df, train_ids)
    row = features.loc[first_test]
    assert row["company_n_train_complaints"] == 0
    train = df[df[C.COMPLAINT_ID].isin(train_ids)]
    expected = (train[C.TIMELY_RESPONSE] == "Yes").mean()
    assert row["company_timely_rate"] == pytest.approx(expected)


def test_pca_ignores_test_embeddings():
    df, train_ids, _ = sample_and_split()
    emb = fake_embeddings(df)
    a, _ = fit_pca(emb, train_ids, n_components=4, seed=0)
    changed = emb.copy()
    is_test = ~changed[C.COMPLAINT_ID].isin(train_ids)
    changed.loc[is_test, changed.columns[1:]] = 1000.0
    b, _ = fit_pca(changed, train_ids, n_components=4, seed=0)
    train_rows = a[C.COMPLAINT_ID].isin(train_ids)
    pd.testing.assert_frame_equal(a[train_rows], b[train_rows])


def test_no_feature_perfectly_predicts_label():
    """Leakage smoke test: on random synthetic data no single feature column
    should separate the classes perfectly."""
    df, train_ids, test_ids = sample_and_split(n=400)
    features, _, _ = build_tabular_features(df, train_ids)
    pcs, _ = fit_pca(fake_embeddings(df), train_ids, n_components=4, seed=0)
    train, _ = build_tables(features, pcs, df, train_ids, test_ids)
    y = train["target"]
    for col in train.columns.drop("target"):
        codes = pd.factorize(train[col])[0]
        # a column "perfectly predicts" if every value maps to one class
        purity = pd.Series(y.to_numpy()).groupby(codes).nunique()
        assert not (purity == 1).all() or train[col].nunique() == len(train), col
        if pd.api.types.is_numeric_dtype(train[col]):
            assert abs(np.corrcoef(train[col].astype(float), y)[0, 1]) < 0.99, col


def test_embeddings_join_checks_order():
    df, _, _ = sample_and_split(n=20)
    embedded = fake_embeddings(df).drop(columns=C.COMPLAINT_ID)
    embedded["target"] = df[C.LABEL].to_numpy()
    out = attach_ids_to_embeddings(embedded, df[C.COMPLAINT_ID], df[C.LABEL])
    assert out[C.COMPLAINT_ID].tolist() == df[C.COMPLAINT_ID].tolist()
    with pytest.raises(ValueError, match="rows"):
        attach_ids_to_embeddings(embedded.iloc[:-1], df[C.COMPLAINT_ID], df[C.LABEL])
    shuffled = embedded.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="labels do not match"):
        attach_ids_to_embeddings(shuffled, df[C.COMPLAINT_ID], df[C.LABEL])


def _company_features_by_id(df, train_ids):
    from fraud_detect.features.tabular import company_features

    out = company_features(df.reset_index(drop=True), train_ids)
    return out.set_index(df[C.COMPLAINT_ID].to_numpy())


def test_training_row_never_sees_its_own_outcome():
    """Leave-one-out: changing a training complaint's own response and
    timeliness does not change that complaint's company features."""
    df, train_ids, _ = sample_and_split()
    target = int(train_ids.iloc[0])
    before = _company_features_by_id(df, train_ids).loc[target]
    changed = df.copy()
    row = changed[C.COMPLAINT_ID] == target
    changed.loc[row, C.COMPANY_RESPONSE] = "Closed with monetary relief"
    changed.loc[row, C.TIMELY_RESPONSE] = "No"
    after = _company_features_by_id(changed, train_ids).loc[target]
    pd.testing.assert_series_equal(before, after)


def test_train_and_test_rows_use_the_same_definition():
    """A test complaint's company features equal what a training complaint of the
    same company gets when that training complaint's outcome is left out."""
    df, train_ids, test_ids = sample_and_split()
    feats = _company_features_by_id(df, train_ids)
    stats = company_statistics(df, train_ids)
    test_id = int(test_ids.iloc[0])
    company = df.set_index(C.COMPLAINT_ID).loc[test_id, C.COMPANY]
    expected = stats.loc[company]
    pd.testing.assert_series_equal(
        feats.loc[test_id], expected.rename(test_id), check_dtype=False
    )
    n_train_same = df[df[C.COMPLAINT_ID].isin(train_ids) & (df[C.COMPANY] == company)]
    train_id = int(n_train_same[C.COMPLAINT_ID].iloc[0])
    assert feats.loc[train_id, "company_n_train_complaints"] == len(n_train_same) - 1


def test_singleton_training_company_gets_fallback():
    df, train_ids, _ = sample_and_split()
    df = df.copy()
    lone = int(train_ids.iloc[0])
    df.loc[df[C.COMPLAINT_ID] == lone, C.COMPANY] = "SYNTHETIC-COMPANY-ONLY-ONE"
    row = _company_features_by_id(df, train_ids).loc[lone]
    assert row["company_n_train_complaints"] == 0
    train = df[df[C.COMPLAINT_ID].isin(train_ids)]
    assert row["company_timely_rate"] == pytest.approx(
        (train[C.TIMELY_RESPONSE] == "Yes").mean()
    )


def test_company_features_of_train_rows_ignore_test_rows():
    df, train_ids, test_ids = sample_and_split()
    before = _company_features_by_id(df, train_ids).loc[train_ids.to_numpy()]
    changed = df.copy()
    is_test = changed[C.COMPLAINT_ID].isin(test_ids)
    changed.loc[is_test, C.COMPANY_RESPONSE] = "Closed with monetary relief"
    changed.loc[is_test, C.COMPANY] = "SYNTHETIC-COMPANY-A"
    after = _company_features_by_id(changed, train_ids).loc[train_ids.to_numpy()]
    pd.testing.assert_frame_equal(before, after)


def test_rare_levels_use_training_counts_only():
    from fraud_detect.features.tabular import RARE_LEVEL, collapse_rare_levels

    ids = pd.Series(range(10))
    frame = pd.DataFrame(
        {
            C.COMPLAINT_ID: range(10),
            "state": ["ZZ"] * 5 + ["YY"] + [None] + ["YY", "YY", "XX"],
        }
    )
    train_ids = ids[:7]  # ZZ x5, YY x1, None x1; test: YY, YY, XX
    out, merged = collapse_rare_levels(frame, train_ids, ("state",), min_count=3)
    assert out["state"].isna().tolist() == [False] * 6 + [True] + [False] * 3
    present = out["state"].drop(index=6).tolist()
    assert present == ["ZZ"] * 5 + [RARE_LEVEL] + [RARE_LEVEL] * 3
    assert merged == {"state": ["XX", "YY"]}
    # Test rows cannot rescue a level: YY has 3 rows overall but 1 in training.
    kept = out.loc[out[C.COMPLAINT_ID].isin(train_ids), "state"]
    assert kept.value_counts().drop(RARE_LEVEL, errors="ignore").min() >= 3
