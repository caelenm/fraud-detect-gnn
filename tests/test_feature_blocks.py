"""Feature blocks, feature sets and the df-analyze input tables (synthetic).

Blocks are written to pytest's tmp_path; values are invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import fake_dataset
from synthetic import fake_block

from fraud_detect import columns as C
from fraud_detect.cli import main
from fraud_detect.config import get_paths
from fraud_detect.features import blocks


def write(tmp_path, name: str, frame: pd.DataFrame) -> None:
    frame.to_parquet(tmp_path / f"{name}.parquet", index=False)


def ids(node_ids, labels) -> pd.DataFrame:
    return pd.DataFrame({C.NODE_ID: node_ids, C.LABEL: labels, C.GROUP_ID: node_ids})


def test_joining_an_extra_block(tmp_path):
    write(tmp_path, "own", fake_block("own", n=10, width=3))
    # A later block (e.g. neighbour aggregates), stored in another node order.
    extra = fake_block("nbr_aaa", n=10, width=2, seed=1).iloc[::-1]
    write(tmp_path, "nbr_aaa", extra)
    joined = blocks.load_feature_set(tmp_path, ["own", "nbr_aaa"])
    assert list(joined.columns) == [C.NODE_ID, "own__x0", "own__x1", "own__x2",
                                    "nbr_aaa__x0", "nbr_aaa__x1"]  # fmt: skip
    row = joined.set_index(C.NODE_ID).loc[7]
    assert row["nbr_aaa__x1"] == extra.set_index(C.NODE_ID).loc[7, "nbr_aaa__x1"]


@pytest.mark.parametrize(
    "column, message",
    [
        ("label", "forbidden"),
        ("own__label", "forbidden"),
        ("group_id", "forbidden"),
        ("own__split", "forbidden"),
        ("target", "forbidden"),
        ("degree", "prefix"),
        ("nbr_aaa__x9", "prefix"),
    ],
)
def test_forbidden_or_unprefixed_columns_fail(tmp_path, column, message):
    block = fake_block("own", n=5, width=2)
    block[column] = 1.0
    with pytest.raises(blocks.BlockError, match=message):
        blocks.validate_block("own", block)


def test_missing_values_missing_ids_and_missing_blocks_fail(tmp_path):
    block = fake_block("own", n=5, width=2)
    block.loc[2, "own__x1"] = np.nan
    with pytest.raises(blocks.BlockError, match="missing values"):
        blocks.validate_block("own", block)
    with pytest.raises(blocks.BlockError, match="node_id"):
        blocks.validate_block("own", block.drop(columns=C.NODE_ID))
    with pytest.raises(blocks.BlockError, match="duplicate"):
        blocks.validate_block("own", pd.concat([block.dropna(), block.dropna()]))
    with pytest.raises(blocks.BlockError, match="is missing"):
        blocks.load_feature_set(tmp_path, ["own"])


def test_rows_follow_the_split_files_and_hold_no_identifiers():
    features = fake_block("own", n=12, width=2)
    train = ids([9, 2, 5, 0], [1, 0, 0, 1])
    test = ids([11, 3], [0, 1])
    tr, te = blocks.model_tables(features, train, test)
    by_id = features.set_index(C.NODE_ID)
    assert tr["own__x0"].tolist() == by_id.loc[[9, 2, 5, 0], "own__x0"].tolist()
    assert tr[C.TARGET].tolist() == [1, 0, 0, 1] and te[C.TARGET].tolist() == [0, 1]
    assert list(tr.columns) == ["own__x0", "own__x1", C.TARGET]
    with pytest.raises(blocks.BlockError, match="both train and test"):
        blocks.model_tables(features, train, ids([2], [0]))
    with pytest.raises(blocks.BlockError, match="no feature row"):
        blocks.model_tables(features, train, ids([99], [0]))


def test_df_analyze_input_stage_writes_only_own_columns_and_target(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=120)
    assert main(["--config", str(path), "--to", "df_analyze_input"]) == 0
    p = get_paths(config)
    train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    test = pd.read_parquet(p.df_analyze_input_dir / "test.parquet")
    # Exactly the own block's columns (under df-analyze's names) plus target.
    expected = [f"own_f{j:02d}" for j in range(5)] + [C.TARGET]
    assert list(train.columns) == expected and list(test.columns) == expected
    saved = pd.read_csv(p.test_ids)
    assert test[C.TARGET].tolist() == saved[C.LABEL].tolist()
    block = pd.read_parquet(p.block_file(C.OWN_BLOCK)).set_index(C.NODE_ID)
    assert np.allclose(test["own_f01"], block.loc[saved[C.NODE_ID], "own__f01"])
