"""Files a GNN classifier rung writes in place of a df-analyze run.

The gnn_classify stage (Models 3-4, E5/E5g/E6/N3) writes the same Model A
summary and metrics table as df_analyze_report, marked `"source": "gnn"`,
plus its out-of-fold and test probabilities. The comparison page and the
split-repeat summary read a GNN side through these helpers, so both sides of
any contrast are scored by the same procedure (invariant 9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detect import columns as C

GNN_SOURCE = "gnn"
GNN_OOF = "gnn_oof.parquet"  # row (train_ids.csv order), fold, prob
GNN_TEST = "gnn_test.parquet"  # node_id (test_ids.csv order), prob
GNN_GRID = "gnn_grid.csv"
GNN_SEEDS = "gnn_seeds.csv"


def is_gnn_summary(summary: dict[str, Any]) -> bool:
    return summary.get("source") == GNN_SOURCE


def oof_rows(reports_dir: Path) -> pd.DataFrame:
    """A GNN rung's out-of-fold predictions (row, fold, prob), in row order."""
    rows = pd.read_parquet(reports_dir / GNN_OOF).sort_values("row")
    if rows["row"].duplicated().any() or rows["prob"].isna().any():
        raise ValueError(f"{reports_dir / GNN_OOF} must have one prediction per row")
    return rows[["row", "fold", "prob"]].reset_index(drop=True)


def test_probabilities(reports_dir: Path, test_ids: pd.DataFrame) -> np.ndarray:
    """A GNN rung's test P(fraud), in test_ids.csv order (checked)."""
    frame = pd.read_parquet(reports_dir / GNN_TEST)
    if not np.array_equal(frame[C.NODE_ID].to_numpy(), test_ids[C.NODE_ID].to_numpy()):
        raise ValueError(f"{reports_dir / GNN_TEST} is not in test_ids.csv order")
    return frame["prob"].to_numpy(dtype=float)
