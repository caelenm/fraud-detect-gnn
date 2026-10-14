"""Check every tuned model df-analyze saved; save Model A in its native format.

THIS SCRIPT RUNS INSIDE DF-ANALYZE'S ENVIRONMENT, not this project's:

    uv run --directory <df-analyze> --python '>=3.13.11,<3.14' \
        python <this file> --request <staging dir>/request.json

The export_model stage writes the request (paths, Model A's identity and
threshold, seed) and moves the finished folder into place.

df-analyze keeps every tuned (model, feature selection) combination, refit on
the full training set, in eval_htune_results_jsonpickle_00.json (written
with jsonpickle by EvaluationResults.save). If jsonpickle cannot encode an
object, df-analyze silently writes str(object) instead, so this script checks
each combination: did its model decode to a fitted estimator, and does it
reproduce the test probabilities df-analyze saved in
prediction_results_00.json? The answers go to models_check.csv.

Model A (chosen by the report stage from the shared CV, never here) is then
saved in its library's own format (joblib for scikit-learn and LightGBM,
.cbm for CatBoost). If its decoded model fails the check, it is refit with
df-analyze's `refit_tuned` and its tuned parameters on the full training set
and checked again. The saved file is reloaded and checked once more. If
Model A cannot reproduce its probabilities, the script exits with code 3 and
the stage stops: ask before working around that. No other combination is
ever refit.

The test set is read only to compare probabilities with df-analyze's own
saved ones; nothing is chosen or tuned on it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from model_io import (
    count_placeholders,
    decode_results,
    load_native,
    read_table,
    read_target,
    save_native,
    setup_paths,
    without_checkpoint_files,
)

setup_paths()

from cv_select import (  # noqa: E402
    NEEDS_NUM_CLASSES,
    find_model_class,
    seed_everything,
    tuned_params,
)

from fraud_detect.models import model_export as mx  # noqa: E402

UNVERIFIED = 3  # Model A cannot reproduce its saved probabilities
UNSUPPORTED = 4  # Model A's library has no native export here


def matches(result: Any, entry: dict) -> bool:
    """Whether a decoded HtuneResult belongs to a prediction_results entry
    (when the result itself decoded; a placeholder string cannot be told)."""
    if isinstance(result, str):
        return True
    return str(getattr(result, "selection", "")) == str(entry["selection"]) and list(
        getattr(result, "selected_cols", [])
    ) == list(entry["selected_cols"])


def probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Test probabilities as anyone with the bundle gets them (no GANDALF
    checkpoint files; see model_io.without_checkpoint_files)."""
    return np.asarray(without_checkpoint_files(model).predict_proba(X), dtype=float)


def check_model(model: Any, entry: dict, X_test: pd.DataFrame, tolerance: float) -> dict:
    """A df-analyze model against the test predictions df-analyze saved for it:
    probabilities (to `tolerance`), predicted labels, and whether every part
    of it could be rebuilt (model_io.count_placeholders)."""
    cols = list(entry["selected_cols"])
    return mx.check_combination(
        model,
        lambda m: probabilities(m, X_test[cols]),
        entry["probs_test"],
        tolerance,
        placeholders=count_placeholders(getattr(model, "tuned_model", model)),
        predict_labels=lambda m: without_checkpoint_files(m).tuned_predict(X_test[cols]),
        saved_labels=entry.get("preds_test"),
    )


def check_all(
    entries: list[dict], results: list[Any], X_test: pd.DataFrame, thresholds: dict
) -> list[dict]:
    """One models_check.csv row per tuned combination."""
    if len(results) != len(entries):
        raise SystemExit(
            f"{len(results)} saved models but {len(entries)} saved predictions"
        )
    rows = []
    for entry, result in zip(entries, results, strict=True):
        if not matches(result, entry):
            raise SystemExit(
                "eval_htune_results and prediction_results are not in the same order"
            )
        name = mx.model_name(entry["model_cls"])
        cols = list(entry["selected_cols"])
        embed = entry.get("embed_select_model") or ""
        tolerance = mx.tolerance_for(name)
        model = getattr(result, "model", result)
        start = time.perf_counter()
        check = check_model(model, entry, X_test, tolerance)
        row = {
            "model": name,
            "model_cls": entry["model_cls"],
            "selection": entry["selection"],
            "embed_selector": embed,
            "selection_label": mx.selection_label(entry["selection"], embed),
            "n_features": len(cols),
            "decoded": "yes" if check["decoded"] else "no",
            "intact": "yes" if check["intact"] else "no",
            "max_abs_diff": check["max_abs_diff"],
            "tolerance": tolerance,
            "labels_match": check["labels_match"],
            "ok": check["ok"],
            "threshold": thresholds.get((name, str(entry["selection"])), np.nan),
            "error": check["error"],
        }
        rows.append(row)
        status = "ok" if row["ok"] else f"FAILED: {row['error']}"
        print(
            f"{name:<9} {row['selection_label']:<13} decoded {row['decoded']:<3} "
            f"max |diff| {row['max_abs_diff']:.3g}  "
            f"({time.perf_counter() - start:.1f} s)  {status}",
            flush=True,
        )
    return rows


def refit(entry: dict, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> Any:
    """Model A refit with df-analyze's own refit_tuned and its tuned
    parameters on the full training set, as df-analyze did after tuning."""
    cls = find_model_class(entry["model_cls"])
    num_classes = int(y_train.nunique())
    model = (
        cls(num_classes=num_classes) if entry["model_cls"] in NEEDS_NUM_CLASSES else cls()
    )
    seed_everything(seed)
    model.refit_tuned(
        X=X_train[list(entry["selected_cols"])],
        y=y_train,
        tuned_args=tuned_params(entry),
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    export_dir = Path(request["export_dir"])
    out_dir = Path(request["out_dir"])

    entries = json.loads(
        (export_dir / "prediction_results_00.json").read_text(encoding="utf-8")
    )["predictions"]
    X_test = read_table(export_dir / "X_test_00.csv")
    print(f"Decoding {len(entries)} tuned models from {export_dir}", flush=True)
    start = time.perf_counter()
    results = decode_results(export_dir / "eval_htune_results_jsonpickle_00.json")
    print(f"Decoded in {time.perf_counter() - start:.0f} s", flush=True)
    thresholds = {
        (t["model"], str(t["selection"])): float(t["threshold"])
        for t in request["thresholds"]
    }
    rows = check_all(entries, results, X_test, thresholds)
    pd.DataFrame(rows).to_csv(out_dir / mx.MODELS_CHECK, index=False)

    # ---- Model A ---------------------------------------------------------
    want = request["model_a"]
    found = [
        i
        for i, r in enumerate(rows)
        if r["model"] == want["model"]
        and str(r["selection"]) == str(want["selection"])
        and r["embed_selector"] == (want.get("embed_selector") or "")
    ]
    if len(found) != 1:
        raise SystemExit(f"Model A {want} matches {len(found)} saved combinations")
    i = found[0]
    entry, check = entries[i], rows[i]
    name, cols = check["model"], list(entry["selected_cols"])
    try:
        fmt = mx.native_format(name)
    except ValueError as e:
        print(f"ERROR: {e} Stop and ask how Model A should be saved.", file=sys.stderr)
        return UNSUPPORTED
    tolerance = check["tolerance"]
    saved = np.asarray(entry["probs_test"], dtype=float)
    verification: dict[str, Any] = {
        "n_test_rows": len(X_test),
        "tolerance": tolerance,
        "decoded_check": {k: check[k] for k in ("decoded", "intact", "max_abs_diff",
                                                "labels_match", "error")},
    }  # fmt: skip
    source = mx.model_a_source({**check, "decoded": check["decoded"] == "yes"})
    if source == "decoded":
        wrapper = results[i].model
    else:
        print(
            f"Model A ({name}, {check['selection_label']}) cannot be used as decoded "
            f"({check['error']}); refitting it with its tuned parameters on the full "
            "training set, as df-analyze did after tuning.",
            flush=True,
        )
        X_train = read_table(export_dir / "X_train_00.csv")
        y_train = read_target(export_dir / "y_train_00.csv")
        wrapper = refit(entry, X_train, y_train, int(request["seed"]))
        again = check_model(wrapper, entry, X_test, tolerance)
        keys = ("max_abs_diff", "labels_match", "error")
        verification["refit_check"] = {k: again[k] for k in keys}
        if not again["ok"]:
            print(
                f"ERROR: the refit Model A does not reproduce df-analyze's test "
                f"predictions either ({again['error']}). Stop and ask.",
                file=sys.stderr,
            )
            return UNVERIFIED

    estimator = wrapper.tuned_model
    model_file = mx.MODEL_FILES[fmt]
    save_native(estimator, fmt, out_dir / model_file)
    reloaded = load_native(fmt, out_dir / model_file)
    diff = mx.max_abs_diff(probabilities(reloaded, X_test[cols]), saved)
    labels = np.asarray(reloaded.predict(X_test[cols])).ravel()
    saved_labels = np.asarray(entry.get("preds_test")).ravel()
    labels_match = bool(
        labels.shape == saved_labels.shape and (labels == saved_labels).all()
    )
    verification["max_abs_diff_native"] = diff
    verification["labels_match_native"] = labels_match
    if diff > tolerance or not labels_match:
        print(
            f"ERROR: Model A reloaded from {model_file} does not reproduce df-analyze's "
            f"test predictions (probabilities differ by up to {diff:.3g}, tolerance "
            f"{tolerance}; labels match: {labels_match}). Stop and ask.",
            file=sys.stderr,
        )
        return UNVERIFIED

    embed = entry.get("embed_select_model") or ""
    meta = {
        "format_version": mx.FORMAT_VERSION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": request["dataset"],
        "feature_set": request["feature_set"],
        "run": request["run"],
        "results_dir": request["results_dir"],
        "model": name,
        "model_cls": entry["model_cls"],
        "selection": entry["selection"],
        "embed_selector": embed,
        "selection_label": mx.selection_label(entry["selection"], embed),
        "input_columns": cols,
        "prepared_columns": list(X_test.columns),
        "tuned_params": tuned_params(entry),
        "threshold": float(request["threshold"]),
        "threshold_rule": request["threshold_rule"],
        "seed": int(request["seed"]),
        "df_analyze_commit": request["df_analyze_commit"],
        "source": source,
        "model_file": model_file,
        "model_format": fmt,
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "verification": verification,
        "package_versions": mx.package_versions(),
    }
    mx.write_metadata(out_dir / mx.MODEL_A_META, meta)
    print(
        f"Saved Model A ({name}, {meta['selection_label']}, {source}) to "
        f"{out_dir / model_file}; reloaded test probabilities match df-analyze's to "
        f"{diff:.3g}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
