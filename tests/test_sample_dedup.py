"""Near-duplicate removal, stratified sampling, and the train/test split."""

from __future__ import annotations

import pandas as pd
from synthetic import fake_narrative, labeled_sample

from fraud_detect import columns as C
from fraud_detect.data.dedup import normalize_text, remove_near_duplicates
from fraud_detect.data.sample import ids_for, stratified_sample, stratified_split

DEDUP = {"shingle_words": 5, "threshold": 0.85, "num_perm": 128, "seed": 1}


def test_normalize_text_collapses_redactions():
    assert normalize_text("On XX/XX/XXXX I paid {$50.00}!") == "on x x x i paid 50 00"


def test_near_duplicates_keep_earliest():
    base = fake_narrative(1)
    df = pd.DataFrame(
        {
            C.COMPLAINT_ID: [1, 2, 3, 4],
            C.DATE_RECEIVED: pd.to_datetime(
                ["2020-01-05", "2020-01-01", "2020-01-03", "2020-01-02"]
            ),
            C.NARRATIVE: [
                base,
                base.upper() + " XXXX",  # same text after normalisation
                base + " one extra synthetic word",  # near-duplicate
                fake_narrative(2),  # unrelated
            ],
            C.LABEL: [1, 0, 1, 0],
        }
    )
    result = remove_near_duplicates(df, **DEDUP)
    assert sorted(result.kept[C.COMPLAINT_ID]) == [2, 4]
    assert result.n_removed == 2
    assert result.n_groups_with_duplicates == 1
    assert result.n_groups_with_mixed_labels == 1


def test_distinct_texts_are_all_kept():
    df = labeled_sample(n=60)
    assert remove_near_duplicates(df, **DEDUP).n_removed == 0


def test_stratified_sample_preserves_rate():
    df = labeled_sample(n=1000, positive_rate=0.1)
    s = stratified_sample(df, 500, seed=3)
    assert len(s) == 500
    assert abs(s[C.LABEL].mean() - df[C.LABEL].mean()) < 0.005
    assert s[C.COMPLAINT_ID].is_monotonic_increasing


def test_split_is_disjoint_complete_and_stratified():
    df = labeled_sample(n=1000, positive_rate=0.1)
    split = stratified_split(df, test_size=0.4, seed=3)
    train, test = ids_for(split, "train"), ids_for(split, "test")
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(df[C.COMPLAINT_ID])
    assert len(test) == 400
    rate = df.set_index(C.COMPLAINT_ID)[C.LABEL]
    assert abs(rate[test].mean() - rate[train].mean()) < 0.01


def test_split_is_reproducible():
    df = labeled_sample(n=300)
    pd.testing.assert_frame_equal(
        stratified_split(df, 0.4, seed=7), stratified_split(df, 0.4, seed=7)
    )
