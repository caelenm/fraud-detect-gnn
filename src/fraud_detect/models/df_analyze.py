"""Model A: df-analyze command line and split verification.

We create the train/test split ourselves and pass it to df-analyze with
`--df-train` / `--df-tests` (method `list`), so the frozen test set is defined
by our saved node IDs. df-analyze re-encodes features, so after a run we
verify that its exported X/y train and test tables line up row-for-row with
ours before anything downstream relies on them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET = "target"


class SplitVerificationError(RuntimeError):
    """Raised when df-analyze's exported split does not match ours."""


def df_analyze_args(
    cfg: dict[str, Any],
    train_path: Path,
    test_path: Path,
    outdir: Path,
    seed: int,
    ordinals: Sequence[str] = (),
    categoricals: Sequence[str] = (),
) -> list[str]:
    """Command-line arguments for df-analyze.py. Paths must be absolute because
    df-analyze runs from its own directory.

    Feature types are passed explicitly (from column_spec.json) instead of
    letting df-analyze guess. An empty list is passed by leaving the flag out:
    df-analyze's default for both is [] (`column_parser` at the pinned commit
    would also turn "" into [], but omitting the flag does not rely on that).
    """
    args = [
        "--df-train", str(train_path.resolve()),
        "--df-tests", str(test_path.resolve()),
        # Not passing `--df-tests-method list`: df-analyze 4.1.0 parses the flag
        # to a plain string and then fails an enum identity check. Its default
        # is already the `list` method (tune on train, evaluate on the test file).
        "--target", TARGET,
        "--mode", "classify",
    ]  # fmt: skip
    for flag, columns in (("--categoricals", categoricals), ("--ordinals", ordinals)):
        bad = [c for c in columns if not c or "," in c or c != c.strip()]
        if bad:
            raise ValueError(f"Column names df-analyze cannot parse in {flag}: {bad}")
        if columns:
            args += [flag, ",".join(columns)]
    args += [
        "--classifiers", *cfg["classifiers"],
        "--htune-trials", str(cfg["htune_trials"]),
        "--htune-cls-metric", str(cfg["htune_cls_metric"]),
        "--seed", str(seed),
        "--outdir", str(outdir.resolve()),
    ]  # fmt: skip
    return args + [str(a) for a in cfg.get("extra_args", [])]


@dataclass(frozen=True)
class ExportedSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    directory: Path


def find_export_dir(run_outdir: Path) -> Path:
    """df-analyze writes <outdir>/<train file stem>/<options hash>/results/test00/
    when given one test file. Each of our runs uses a fresh outdir, so exactly
    one export must exist below it."""
    matches = sorted(run_outdir.rglob("results/test00/X_test_00.csv"))
    if len(matches) != 1:
        raise SplitVerificationError(
            f"Expected exactly one df-analyze export under {run_outdir}, "
            f"found {[str(m) for m in matches]}"
        )
    return matches[0].parent


def read_export(export_dir: Path) -> ExportedSplit:
    def y(name: str) -> pd.Series:
        frame = pd.read_csv(export_dir / name, index_col=0)
        if frame.shape[1] != 1:
            raise SplitVerificationError(f"{name} has columns {frame.columns.tolist()}")
        return frame.iloc[:, 0]

    return ExportedSplit(
        X_train=pd.read_csv(export_dir / "X_train_00.csv"),
        y_train=y("y_train_00.csv"),
        X_test=pd.read_csv(export_dir / "X_test_00.csv"),
        y_test=y("y_test_00.csv"),
        directory=export_dir,
    )


def _check_monotone_affine(ours: np.ndarray, theirs: np.ndarray) -> str | None:
    """df-analyze's default ('robust') normalisation clips each continuous column
    and then min-max scales it, a non-decreasing map that is affine between the
    clip points. Return an error message if `theirs` is not such a map of
    `ours`, else None."""
    if not (np.isfinite(ours).all() and np.isfinite(theirs).all()):
        return "non-finite values"
    lo, hi = float(theirs.min()), float(theirs.max())
    span = hi - lo
    if span == 0:
        return "exported column is constant"
    tol = 1e-6 * span
    order = np.argsort(ours, kind="stable")
    if (np.diff(theirs[order]) < -tol).any():
        return "row order differs (exported values are not monotone in ours)"
    interior = (theirs > lo + tol) & (theirs < hi - tol)
    if interior.sum() < 3:
        return "too few unclipped values to check"
    slope, intercept = np.polyfit(ours[interior], theirs[interior], 1)
    residual = np.abs(theirs[interior] - (slope * ours[interior] + intercept)).max()
    if slope <= 0 or residual > 1e-4 * span:
        return f"not an affine map (slope={slope:.3g}, max residual={residual:.3g})"
    return None


def verify_export(
    export: ExportedSplit,
    train: pd.DataFrame,
    test: pd.DataFrame,
    continuous_cols: list[str],
) -> dict[str, Any]:
    """Check that df-analyze's exported train/test rows are exactly our rows in
    our order. Raises SplitVerificationError on any mismatch."""
    problems: list[str] = []
    checked: dict[str, int] = {}
    for name, ours, X, y in (
        ("train", train, export.X_train, export.y_train),
        ("test", test, export.X_test, export.y_test),
    ):
        if len(X) != len(ours) or len(y) != len(ours):
            problems.append(
                f"{name}: {len(ours)} rows passed in, df-analyze exported "
                f"X={len(X)}, y={len(y)}"
            )
            continue
        if not np.array_equal(y.to_numpy(), ours[TARGET].to_numpy()):
            problems.append(f"{name}: exported labels differ from ours row by row")
        n_ok = 0
        for col in continuous_cols:
            if col not in X.columns:
                problems.append(f"{name}: column {col} missing from df-analyze export")
                continue
            msg = _check_monotone_affine(
                ours[col].to_numpy(dtype=float), X[col].to_numpy(dtype=float)
            )
            if msg:
                problems.append(f"{name}: {col}: {msg}")
            else:
                n_ok += 1
        checked[name] = n_ok
    if problems:
        raise SplitVerificationError(
            "df-analyze split does not match ours:\n- " + "\n- ".join(problems)
        )
    return {
        "export_dir": str(export.directory),
        "n_train": len(train),
        "n_test": len(test),
        "continuous_columns_checked": checked,
        "labels_identical": True,
    }
