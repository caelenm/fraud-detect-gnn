"""The saved test node IDs must match df-analyze's exported test set.

These tests fake a df-analyze export directory in tmp_path, mimicking how
df-analyze writes it (clip + min-max normalised continuous columns, binary
columns passed through, no identifiers), and check that verification accepts
an aligned export and rejects reordered, truncated or relabelled ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import fake_model_tables

from fraud_detect.models.df_analyze import TARGET as T
from fraud_detect.models.df_analyze import (
    SplitVerificationError,
    df_analyze_args,
    find_export_dir,
    read_export,
    verify_export,
)

CONTINUOUS = ["own__f00", "own__f01"]


def our_tables():
    train, test = fake_model_tables()
    return train, test, CONTINUOUS


def like_df_analyze(table: pd.DataFrame, cont_cols: list[str]) -> pd.DataFrame:
    """Clip to a robust range, then min-max scale; pass a binary column through."""
    out = table[["own__f03"]].astype(float)
    for col in cont_cols:
        x = table[col]
        lo, hi = x.quantile(0.05), x.quantile(0.95)
        width = hi - lo
        clipped = x.clip(lo - 0.5 * width, hi + 0.5 * width)
        out[col] = (clipped - clipped.min()) / (clipped.max() - clipped.min())
    return out


def write_export(root, train, test, pca_cols, transform_test=None):
    d = root / "train" / "abc123hash" / "results" / "test00"
    d.mkdir(parents=True)
    X_test = like_df_analyze(test, pca_cols)
    y_test = test[T]
    if transform_test:
        X_test, y_test = transform_test(X_test, y_test)
    like_df_analyze(train, pca_cols).to_csv(d / "X_train_00.csv", index=False)
    X_test.to_csv(d / "X_test_00.csv", index=False)
    train[T].to_frame().to_csv(d / "y_train_00.csv", index=True)
    y_test.to_frame().to_csv(d / "y_test_00.csv", index=True)
    return root


def verify(root, train, test, pca_cols):
    export = read_export(find_export_dir(root))
    return verify_export(export, train, test, pca_cols)


def test_aligned_export_passes(tmp_path):
    train, test, pca_cols = our_tables()
    report = verify(write_export(tmp_path, train, test, pca_cols), train, test, pca_cols)
    assert report["n_test"] == len(test)
    assert report["continuous_columns_checked"] == {"train": 2, "test": 2}


def test_reordered_test_rows_fail(tmp_path):
    train, test, pca_cols = our_tables()

    def shuffle(X, y):
        order = np.random.default_rng(1).permutation(len(X))
        return X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)

    root = write_export(tmp_path, train, test, pca_cols, shuffle)
    with pytest.raises(SplitVerificationError, match="test"):
        verify(root, train, test, pca_cols)


def test_reordered_rows_with_identical_labels_fail(tmp_path):
    """Swapping two same-label rows keeps y identical; features must catch it."""
    train, test, pca_cols = our_tables()
    negatives = np.flatnonzero(test[T].to_numpy() == 0)[:2]

    def swap(X, y):
        order = np.arange(len(X))
        order[negatives] = order[negatives[::-1]]
        return X.iloc[order].reset_index(drop=True), y

    root = write_export(tmp_path, train, test, pca_cols, swap)
    with pytest.raises(SplitVerificationError, match="row order differs"):
        verify(root, train, test, pca_cols)


def test_dropped_test_row_fails(tmp_path):
    train, test, pca_cols = our_tables()
    root = write_export(
        tmp_path, train, test, pca_cols, lambda X, y: (X.iloc[1:], y.iloc[1:])
    )
    with pytest.raises(SplitVerificationError, match="rows passed in"):
        verify(root, train, test, pca_cols)


def test_changed_labels_fail(tmp_path):
    train, test, pca_cols = our_tables()
    root = write_export(tmp_path, train, test, pca_cols, lambda X, y: (X, 1 - y))
    with pytest.raises(SplitVerificationError, match="labels differ"):
        verify(root, train, test, pca_cols)


def test_missing_or_ambiguous_export_dir_fails(tmp_path):
    with pytest.raises(SplitVerificationError, match="exactly one"):
        find_export_dir(tmp_path)


def test_df_analyze_args_use_predefined_split(tmp_path):
    cfg = {"classifiers": ["lgbm", "lr"], "htune_trials": 5, "htune_cls_metric": "acc"}
    args = df_analyze_args(cfg, tmp_path / "tr.parquet", tmp_path / "te.parquet",
                           tmp_path / "out", seed=1)  # fmt: skip
    joined = " ".join(args)
    assert "--df-train" in args and "--df-tests" in args
    assert "--df-tests-method" not in args  # upstream bug; default is `list`
    assert "--classifiers lgbm lr" in joined
    assert "--wrapper-select" not in args  # wrapper selection stays off


def test_df_analyze_args_pass_explicit_types_and_omit_empty_lists(tmp_path):
    cfg = {"classifiers": ["lgbm"], "htune_trials": 5, "htune_cls_metric": "bal-acc"}
    paths = (tmp_path / "tr.parquet", tmp_path / "te.parquet", tmp_path / "out")
    args = df_analyze_args(cfg, *paths, seed=1, ordinals=["own__f02", "own__f05"])
    i = args.index("--ordinals")
    assert args[i + 1] == "own__f02,own__f05"
    # No categoricals: the flag is left out (df-analyze's default is []), never
    # passed as "", which a different parser could read as one column named "".
    assert "--categoricals" not in args and "" not in args
    none = df_analyze_args(cfg, *paths, seed=1)
    assert "--ordinals" not in none and "--categoricals" not in none
    with pytest.raises(ValueError, match="cannot parse"):
        df_analyze_args(cfg, *paths, seed=1, ordinals=["a,b"])
