"""Predict with Model A (or any other tuned df-analyze model) without retraining.

THIS SCRIPT RUNS INSIDE DF-ANALYZE'S ENVIRONMENT. From this repository's root:

    uv run --project ../df-analyze --python '>=3.13.11,<3.14' \
        python scripts/dfa/predict.py --dataset amazon \
        --input <df-analyze run>/train/<hash>/results/test00/X_test_00.csv \
        --out predictions.csv

(`--project` uses df-analyze's environment without changing directory, so
relative paths work; set DF_ANALYZE_DIR if df-analyze is not in ../df-analyze.)

By default it loads Model A from outputs/<dataset>/<feature set>/model/
(model_a.json and the model file the export_model stage saved). With
`--model lgbm --selection pred` it loads that tuned combination instead,
straight from df-analyze's saved results (only combinations that
models_check.csv marks ok).

The input must be df-analyze's PREPARED features: its X_train_00.csv or
X_test_00.csv export, or a table with exactly the model's input columns in the
same order. df-analyze rescales continuous columns and encodes binary ones
before modelling, so our raw feature tables (data/processed/.../train.parquet)
are not valid input. Column names and order are checked; any mismatch stops
the script.

The output CSV has one row per input row: `row`, `prob_fraud`, and `label`
(1 when prob_fraud >= the saved decision threshold, which was chosen on
out-of-fold training predictions).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from model_io import (
    REPO_ROOT,
    decode_results,
    load_native,
    read_table,
    setup_paths,
    without_checkpoint_files,
)

setup_paths()

from fraud_detect.models import model_export as mx  # noqa: E402


def model_dir_for(args: argparse.Namespace) -> Path:
    if args.model_dir:
        return args.model_dir.resolve()
    if not args.dataset:
        raise SystemExit("Give --dataset (e.g. amazon) or --model-dir")
    return REPO_ROOT / "outputs" / args.dataset / args.feature_set / "model"


def repo_path(value: str) -> Path:
    """A path from model_a.json: relative to the repository root unless absolute."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def other_combination(
    model_dir: Path, meta: dict, model: str, selection: str
) -> tuple[Any, list[str], float, str]:
    """A tuned combination other than Model A, decoded from df-analyze's saved
    results: (model, input columns, threshold, description)."""
    check = pd.read_csv(model_dir / mx.MODELS_CHECK, keep_default_na=False)
    rows = check[(check["model"] == model)
                 & ((check["selection_label"] == selection)
                    | (check["selection"] == selection))]  # fmt: skip
    if len(rows) != 1:
        options = sorted(zip(check["model"], check["selection_label"], strict=True))
        raise SystemExit(
            f"No single tuned combination {model} / {selection}. Choose one of: {options}"
        )
    row = rows.iloc[0]
    if str(row["ok"]) != "True":
        raise SystemExit(
            f"{model} / {selection} did not reproduce df-analyze's test probabilities "
            f"when exported ({row['error']}), so it cannot be used here."
        )
    results = decode_results(
        repo_path(meta["results_dir"]) / "eval_htune_results_jsonpickle_00.json"
    )
    result = results[int(row.name)]  # models_check.csv keeps df-analyze's order
    if str(result.selection) != str(row["selection"]):
        raise SystemExit("models_check.csv and df-analyze's results are out of order")
    threshold = float(row["threshold"]) if row["threshold"] != "" else 0.5
    label = f"{model} ({row['selection_label']}), decoded from df-analyze's results"
    # As in the export check: never read checkpoint files outside the bundle.
    model_obj = without_checkpoint_files(result.model)
    return model_obj, list(result.selected_cols), threshold, label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="df-analyze prepared feature table (.csv or .parquet)",
    )
    parser.add_argument("--out", type=Path, required=True, help="predictions .csv")
    parser.add_argument("--dataset", help="dataset whose Model A to use, e.g. amazon")
    parser.add_argument("--feature-set", default="m1_own")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="folder with model_a.json (default: from --dataset)",
    )
    parser.add_argument("--model", help="another tuned model, e.g. lgbm (not Model A)")
    parser.add_argument("--selection", help="its feature selection, e.g. none or pred")
    parser.add_argument(
        "--threshold", type=float, help="decision threshold (default: the saved one)"
    )
    args = parser.parse_args()  # fmt: skip
    if bool(args.model) != bool(args.selection):
        raise SystemExit("--model and --selection go together")

    model_dir = model_dir_for(args)
    meta = mx.read_metadata(model_dir / mx.MODEL_A_META)
    for warning in mx.version_warnings(meta["package_versions"], mx.package_versions()):
        print(f"WARNING: {warning}", file=sys.stderr)
    if args.model:
        model, columns, threshold, label = other_combination(
            model_dir, meta, args.model, args.selection
        )
        predict = model.predict_proba
    else:
        estimator = load_native(meta["model_format"], model_dir / meta["model_file"])
        columns, threshold = list(meta["input_columns"]), float(meta["threshold"])
        label = f"Model A: {meta['model']} ({meta['selection_label']})"
        predict = estimator.predict_proba
    if args.threshold is not None:
        threshold = args.threshold

    table = read_table(args.input.resolve())
    try:
        columns = mx.check_columns(table.columns, columns, meta["prepared_columns"])
    except mx.ColumnMismatchError as e:
        raise SystemExit(f"error: {args.input}: {e}") from e
    prob = np.asarray(predict(table[columns]), dtype=float)[:, 1]
    out = pd.DataFrame(
        {"row": np.arange(len(prob)), "prob_fraud": prob,
         "label": (prob >= threshold).astype(int)}
    )  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(args.out)
    print(
        f"{label}: {len(out):,} rows, {int(out['label'].sum()):,} predicted fraud at "
        f"threshold {threshold:.4f}. Saved {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
