"""The `own` feature block and explicit column types (plan §6-7).

The own block is each node's own features from the .mat, in column order,
named own__f00, own__f01, ... It covers every node in the graph, including
unlabelled ones (later graph features use them as context); the model input
stages select the labelled rows they need.

Column types are decided here, on TRAINING nodes only (leakage invariant 6),
and passed to df-analyze explicitly rather than letting it guess:
- binary: exactly 2 distinct values;
- ordinal: integer-valued with more than 2 distinct values;
- continuous: everything else.
A column with a single value on the training nodes carries no information
and is reported instead of typed; dropping it is a decision for the group.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraud_detect import columns as C

BINARY, ORDINAL, CONTINUOUS, CONSTANT = "binary", "ordinal", "continuous", "constant"


def own_block(features: np.ndarray) -> pd.DataFrame:
    """node_id plus own__fXX columns for every node (node_id = row index)."""
    n, d = features.shape
    names = [f"{C.OWN_BLOCK}{C.BLOCK_SEPARATOR}{C.feature_name(j, d)}" for j in range(d)]
    block = pd.DataFrame(features, columns=names)
    block.insert(0, C.NODE_ID, np.arange(n, dtype=np.int64))
    return block


def column_type(values: np.ndarray) -> str:
    """Type of one column from its training values (see module docstring)."""
    values = values[~np.isnan(values)]
    n_unique = len(np.unique(values))
    if n_unique <= 1:
        return CONSTANT
    if n_unique == 2:
        return BINARY
    if np.array_equal(values, np.round(values)):
        return ORDINAL
    return CONTINUOUS


def column_spec(block: pd.DataFrame, train_ids: np.ndarray) -> dict[str, Any]:
    """Types of every feature column, decided on the training nodes only."""
    train = block.set_index(C.NODE_ID).loc[train_ids]
    columns = {}
    for name in train.columns:
        values = train[name].to_numpy(dtype=float)
        columns[name] = {
            "type": column_type(values),
            "n_unique_train": int(len(np.unique(values[~np.isnan(values)]))),
        }
    by_type = {
        kind: [c for c, info in columns.items() if info["type"] == kind]
        for kind in (BINARY, ORDINAL, CONTINUOUS, CONSTANT)
    }
    return {
        "decided_on": f"{len(train)} training nodes",
        "rules": {
            BINARY: "exactly 2 distinct values",
            ORDINAL: "integer-valued with more than 2 distinct values",
            CONTINUOUS: "everything else",
        },
        "columns": columns,
        **by_type,
    }
