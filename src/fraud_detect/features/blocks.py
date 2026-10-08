"""Feature blocks and feature sets (plan §6).

A feature block is one parquet file, data/processed/<dataset>/features/
<block>.parquet, keyed by `node_id`, with every other column named
`<block>__<feature>`. The `own` block holds each node's own features; later
blocks (neighbour aggregates `nbr_<relation>`, GNN embeddings
`gnnssl_<relation>`) are added the same way.

A feature set (a rung of the model ladder) is a list of blocks in the config
(`feature_sets`). The df_analyze_input stage joins a set's blocks on
`node_id` and writes one train and one test table in the order of the saved
split, with only feature columns and `target`.

Graph feature builders follow one convention, so labels cannot leak into
features (leakage invariants 1 and 5): a builder is called as
`builder(features, edges)` with the feature table and an edge list, and never
receives labels. tests/test_label_blindness.py shows the check every builder
must pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detect import columns as C


class BlockError(ValueError):
    """A feature block or feature set breaks the rules; the message says how."""


def validate_block(name: str, block: pd.DataFrame) -> None:
    """A block must have a unique `node_id`, only `<name>__*` feature columns,
    no missing values, and nothing that identifies, labels or places a node."""
    if C.NODE_ID not in block.columns:
        raise BlockError(f"Block {name!r} has no {C.NODE_ID} column")
    if block[C.NODE_ID].duplicated().any():
        raise BlockError(f"Block {name!r} has duplicate {C.NODE_ID} values")
    prefix = name + C.BLOCK_SEPARATOR
    features = [c for c in block.columns if c != C.NODE_ID]
    if not features:
        raise BlockError(f"Block {name!r} has no feature columns")
    forbidden = set(C.FORBIDDEN_FEATURE_COLUMNS)
    bad = [
        c for c in features
        if c in forbidden or c.removeprefix(prefix) in forbidden
    ]  # fmt: skip
    if bad:
        raise BlockError(f"Block {name!r} has forbidden columns: {bad}")
    unprefixed = [c for c in features if not c.startswith(prefix)]
    if unprefixed:
        raise BlockError(
            f"Block {name!r} has columns without the {prefix!r} prefix: {unprefixed}"
        )
    nulls = [c for c in features if block[c].isna().any()]
    if nulls:
        raise BlockError(f"Block {name!r} has missing values in {nulls}")
    numeric = block[features].apply(pd.api.types.is_numeric_dtype)
    if not numeric.all():
        raise BlockError(
            f"Block {name!r} has non-numeric columns: {list(numeric[~numeric].index)}"
        )


def load_feature_set(features_dir: Path, blocks: Sequence[str]) -> pd.DataFrame:
    """Join a feature set's blocks on node_id (each block validated)."""
    if not blocks:
        raise BlockError("A feature set needs at least one block")
    joined: pd.DataFrame | None = None
    for name in blocks:
        path = features_dir / f"{name}.parquet"
        if not path.is_file():
            raise BlockError(
                f"Feature block {name!r} is missing ({path}); run the stage that "
                "builds it first."
            )
        block = pd.read_parquet(path)
        validate_block(name, block)
        if joined is None:
            joined = block
            continue
        joined = joined.merge(block, on=C.NODE_ID, how="inner", validate="1:1")
    assert joined is not None
    return joined


def model_tables(
    features: pd.DataFrame, train_ids: pd.DataFrame, test_ids: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test tables: the feature columns plus `target`, with rows in
    the order of the saved ID files and no identifier column."""
    overlap = set(train_ids[C.NODE_ID]) & set(test_ids[C.NODE_ID])
    if overlap:
        raise BlockError(f"{len(overlap)} node IDs are in both train and test")
    table = features.set_index(C.NODE_ID)

    def rows(ids: pd.DataFrame) -> pd.DataFrame:
        missing = set(ids[C.NODE_ID]) - set(table.index)
        if missing:
            raise BlockError(f"{len(missing)} split nodes have no feature row")
        out = table.loc[ids[C.NODE_ID].to_numpy()].reset_index(drop=True)
        out[C.TARGET] = ids[C.LABEL].to_numpy(dtype=np.int64)
        return out

    return rows(train_ids), rows(test_ids)
