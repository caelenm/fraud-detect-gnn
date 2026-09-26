"""Stratified sampling and the train/test split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detect import columns as C


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw `n` rows preserving the label proportions of `df`."""
    if n > len(df):
        raise ValueError(f"Requested {n} rows but only {len(df)} are available")
    if n == len(df):
        return df.sort_values(C.COMPLAINT_ID, ignore_index=True)
    sampled, _ = train_test_split(
        df, train_size=n, stratify=df[C.LABEL], random_state=seed
    )
    return sampled.sort_values(C.COMPLAINT_ID, ignore_index=True)


def stratified_split(df: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    """Return a frame of Complaint ID and split ('train' or 'test'), stratified by
    label and sorted by Complaint ID."""
    train, test = train_test_split(
        df[[C.COMPLAINT_ID, C.LABEL]],
        test_size=test_size,
        stratify=df[C.LABEL],
        random_state=seed,
    )
    split = pd.concat(
        [
            train[[C.COMPLAINT_ID]].assign(split="train"),
            test[[C.COMPLAINT_ID]].assign(split="test"),
        ]
    )
    return split.sort_values(C.COMPLAINT_ID, ignore_index=True)


def ids_for(split: pd.DataFrame, name: str) -> pd.Series:
    ids = split.loc[split["split"] == name, C.COMPLAINT_ID]
    return ids.sort_values(ignore_index=True)
