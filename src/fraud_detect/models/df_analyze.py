"""Model A: df-analyze command line, column names, and run verification.

We create the train/test split ourselves and pass it to df-analyze with
`--df-train` / `--df-tests` (method `list`), so the frozen test set is defined
by our saved node IDs. df-analyze re-encodes features, so after a run we
verify that its exported X/y train and test tables line up row-for-row with
ours, and that it typed every column as we specified and dropped none,
before anything downstream relies on them.

Column names: df-analyze renames "trashy" feature names (`sanitize_names` at
the pinned commit), and its rules collapse the block separator `__` to `_`.
The df-analyze input therefore uses `df_analyze_name(column)`, which is what
df-analyze would rename it to, so df-analyze renames nothing and its outputs
use the same names as our input tables.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detect import columns as C

TARGET = C.TARGET


class SplitVerificationError(RuntimeError):
    """Raised when df-analyze's exported split does not match ours."""


class TypeCheckError(RuntimeError):
    """Raised when df-analyze typed or dropped a column against our spec."""


# df-analyze's sanitize_names (preprocessing/cleaning.py at the pinned commit).
_TRASH = re.compile(r"[\\\.\^\$\*\+\?\{\}\[\]\(\)\| ]")
_LGBM = re.compile(r"[,:\"]")


def df_analyze_name(name: str) -> str:
    """The name df-analyze gives a feature column (see the module docstring)."""
    renamed = _LGBM.sub("_", _TRASH.sub("_", name))
    renamed = re.sub(r"_+", "_", renamed)
    return renamed[:-1] if renamed.endswith("_") else renamed


def df_analyze_names(columns: Sequence[str]) -> dict[str, str]:
    """Our column name -> df-analyze's, refusing collisions and the target."""
    names = {c: df_analyze_name(c) for c in columns}
    seen: dict[str, str] = {}
    for original, renamed in names.items():
        if renamed == TARGET:
            raise ValueError(f"Column {original!r} would be renamed to the target")
        if renamed in seen:
            raise ValueError(
                f"Columns {seen[renamed]!r} and {original!r} both become {renamed!r}"
            )
        seen[renamed] = original
    return names


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


# df-analyze's inferred kinds (InferredKind values at the pinned commit) ->
# our column types. Every other kind (id, time, const, cat, ...) means the
# column was dropped or would be one-hot encoded: a mismatch.
DFA_KINDS = {
    "bin": "binary",
    "cont": "continuous",
    "cont-coerce": "continuous",
    "ord": "ordinal",
    "user-ord": "ordinal",
    "ord-coerce": "ordinal",
}
# Kinds df-analyze drops before modelling (its "destructive" changes).
DROPPED_KINDS = {"id", "id?", "time", "time?", "const", "nyan"}


def binary_indicator(name: str, exported: set[str], ours: set[str]) -> str | None:
    """The exported column df-analyze made from binary column `name`, if any.

    df-analyze encodes a binary column without NaNs with
    `pd.get_dummies(..., drop_first=True)` (preprocessing/cleaning.py at the
    pinned commit): one 0/1 indicator named `<name>_<second value>`, e.g.
    own_f22 -> own_f22_1.0. Exactly one such column must exist, and it must not
    be one of our own column names."""
    matches = [c for c in exported if c.startswith(name + "_") and c not in ours]
    return matches[0] if len(matches) == 1 else None


def find_inferred_types(run_outdir: Path) -> Path:
    matches = sorted(run_outdir.rglob("inspection/inferred_types.csv"))
    if len(matches) != 1:
        raise TypeCheckError(
            f"Expected exactly one inspection/inferred_types.csv under {run_outdir}, "
            f"found {[str(m) for m in matches]}"
        )
    return matches[0]


def check_inferred_types(
    inferred: pd.DataFrame,
    expected: dict[str, str],
    exported_columns: Sequence[str],
) -> dict[str, Any]:
    """Compare df-analyze's inferred_types.csv (index feature_name; columns
    user, inferred, reason) with our types, keyed by df-analyze names.

    Fails if a column's final type differs, if df-analyze typed a column we
    did not give it, or if one of our columns is missing from its inspection
    or from its exported training table (dropped). Coercions (df-analyze was
    unsure and guessed) are allowed when the guess matches ours, and reported.
    """
    found = {str(k): str(v) for k, v in inferred["inferred"].items()}
    reasons = {str(k): str(v) for k, v in inferred["reason"].items()}
    problems, rows = [], []
    for name, ours in expected.items():
        kind = found.get(name)
        theirs = DFA_KINDS.get(kind or "", f"not modelled ({kind})")
        rows.append({"feature": name, "ours": ours, "df_analyze": kind,
                     "reason": reasons.get(name, "")})  # fmt: skip
        if kind is None:
            problems.append(f"{name}: missing from df-analyze's inspection")
        elif theirs != ours:
            problems.append(f"{name}: ours {ours}, df-analyze {kind} ({reasons[name]})")
    for name in sorted(set(found) - set(expected)):
        problems.append(f"{name}: typed by df-analyze ({found[name]}) but not ours")
    exported = set(exported_columns)
    encoded = {}
    for name, ours in expected.items():
        if name in exported:
            continue
        indicator = binary_indicator(name, exported, set(expected))
        if ours == "binary" and indicator is not None:
            encoded[name] = indicator
        else:
            problems.append(f"{name}: missing from df-analyze's exported X_train")
    destructive = [
        f"{n}: {found[n]} ({reasons[n]})"
        for n in found
        if found[n] in DROPPED_KINDS or "coerce" in found[n].lower()
    ]
    if problems:
        raise TypeCheckError(
            "df-analyze's column types differ from column_spec.json, or it dropped "
            "features:\n- " + "\n- ".join(problems)
        )
    return {
        "n_features": len(expected),
        "all_types_match": True,
        "binary_encoded_as": encoded,
        "dropped_or_coerced": destructive,
        "columns": rows,
    }
