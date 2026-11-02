"""Refit chosen configurations on the whole training set and predict the test set.

THIS SCRIPT RUNS INSIDE DF-ANALYZE'S ENVIRONMENT, not this project's:

    uv run --directory <df-analyze> --python '>=3.13.11,<3.14' \
        python <this file> --export-dir <.../results/test00> --jobs jobs.json \
        --out probs.parquet

For the default-settings sensitivity check (scripts/default_settings_check.py;
docs/RESEARCH_PLAN.md §11, 2026-11-02). --jobs is a JSON list of {settings,
model_cls, selection, embed_selector}, chosen by the caller on training data
only. Writes one probability per job and test row, in the row order of
df-analyze's X_test export. The test labels are not read here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# cv_select also puts df-analyze's source tree on the path.
from cv_select import (
    NEEDS_NUM_CLASSES,
    find_model_class,
    job_key,
    seed_everything,
    tuned_params,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=555)
    args = parser.parse_args()

    X = pd.read_csv(args.export_dir / "X_train_00.csv")
    y = pd.read_csv(args.export_dir / "y_train_00.csv", index_col=0).iloc[:, 0]
    y = y.astype(int).reset_index(drop=True).rename("target")
    X_test = pd.read_csv(args.export_dir / "X_test_00.csv")
    predictions = json.loads(
        (args.export_dir / "prediction_results_00.json").read_text(encoding="utf-8")
    )["predictions"]
    entries = {job_key(e, s): e for e in predictions for s in ("tuned", "default")}
    num_classes = int(y.nunique())

    frames = []
    for job in json.loads(args.jobs.read_text(encoding="utf-8")):
        settings, name = job["settings"], job["model_cls"]
        key = (name, job["selection"], job["embed_selector"], settings)
        entry = entries[key]
        params = tuned_params(entry) if settings == "tuned" else {}
        cols = entry["selected_cols"]
        seed_everything(args.seed)
        cls = find_model_class(name)
        model = cls(num_classes=num_classes) if name in NEEDS_NUM_CLASSES else cls()
        model.refit_tuned(X=X[cols], y=y, tuned_args=params)
        prob = np.asarray(model.predict_proba(X_test[cols]))[:, 1]
        print(f"{name} · {job['selection']} · {settings}: {len(cols)} columns, "
              f"{len(prob)} test rows", flush=True)  # fmt: skip
        frames.append(
            pd.DataFrame(
                {"settings": settings, "row": np.arange(len(prob)), "prob": prob}
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.stem + ".tmp.parquet")
    pd.concat(frames, ignore_index=True).to_parquet(tmp, index=False)
    os.replace(tmp, args.out)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
