"""Test helpers: temporary configs and fake dataset files under tmp_path.

Nothing here is real data; the fake .mat contents come from synthetic.py.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import yaml
from synthetic import (
    fake_care_gnn_mat,
    fake_df_analyze_results,
    fake_oof_predictions,
    fake_shared_cv,
)

from fraud_detect.config import DEFAULT_CONFIG, load_config, with_dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tmp_config(tmp_path: Path, dataset: str = "amazon", config: dict | None = None):
    """The default config (or `config`) with every path under tmp_path and a
    dataset chosen. Returns the config file (for the CLI) and the config with
    the dataset set."""
    config = config or load_config(DEFAULT_CONFIG)
    for key in ("raw_dir", "processed_dir", "outputs_dir"):
        config["paths"][key] = str(tmp_path / key)
    config["dataset"] = dataset
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, with_dataset(config, None)


def fake_dataset(tmp_path: Path, **kwargs) -> tuple[Path, dict]:
    """Write a fake CARE-GNN zip to tmp_path/server/ and return (config file,
    config) with a dataset `fake` whose checksums and expected counts describe
    it, downloaded from that folder through a file:// URL."""
    server = tmp_path / "server"
    server.mkdir()
    mat_path = tmp_path / "Fake.mat"
    contents = fake_care_gnn_mat(**kwargs)
    scipy.io.savemat(mat_path, contents)
    with zipfile.ZipFile(server / "Fake.zip", "w") as zf:
        zf.write(mat_path, "Fake.mat")
    labels = contents["label"].ravel()
    prefix = kwargs.get("unlabelled_prefix", 8)

    def n_edges(m: sp.spmatrix) -> int:
        return int(sp.triu(m, k=1).nnz)

    config = load_config(DEFAULT_CONFIG)
    config["download"]["base_url"] = server.resolve().as_uri()
    config["datasets"]["fake"] = {
        "archive": "Fake.zip",
        "archive_sha256": sha256(server / "Fake.zip"),
        "mat": "Fake.mat",
        "mat_sha256": sha256(mat_path),
        "relations": {"aaa": "net_aaa", "bbb": "net_bbb"},
        "unlabelled_prefix": prefix,
        "groups": {"rule": "duplicate_features"},
        "expected": {
            "n_nodes": len(labels),
            "n_labelled": len(labels) - prefix,
            "n_positive": int(labels[prefix:].sum()),
            "n_features": contents["features"].shape[1],
            "n_edges": {
                "aaa": n_edges(contents["net_aaa"]),
                "bbb": n_edges(contents["net_bbb"]),
                "homo": n_edges(contents["homo"]),
            },
        },
        "sanity_band": {"pr_auc": 0.5, "auroc": 0.5},
    }
    mat_path.unlink()
    return tmp_config(tmp_path, "fake", config)


def load_edges(path: Path) -> np.ndarray:
    with np.load(path) as f:
        return f["edges"]


FAKE_RUN = "20260101T000000Z"


def fake_df_analyze_run(paths, cv_scores: dict[tuple[str, str], float]) -> None:
    """Write the files of a finished, verified df-analyze run and select_model
    stage for the feature set in `paths` (all values invented), so the report
    stages can run. Needs the df_analyze_input stage's tables."""
    train = pd.read_parquet(paths.df_analyze_input_dir / "train.parquet")
    test = pd.read_parquet(paths.df_analyze_input_dir / "test.parquet")
    run = paths.df_analyze_output_dir / FAKE_RUN
    deep = run / "train" / "hash"
    for sub in ("tuning/test00", "results/test00", "prepared"):
        (deep / sub).mkdir(parents=True)
    entries, tuned = fake_df_analyze_results(test["target"].to_numpy(), cv_scores)
    tuned.to_csv(deep / "tuning/test00/tuned_models_00.csv")
    (deep / "results/test00/prediction_results_00.json").write_text(
        json.dumps({"predictions": entries})
    )
    pd.DataFrame({"0": [0, 1]}).to_parquet(deep / "prepared/labels.parquet")
    (deep / "options.json").write_text(json.dumps({"htune_trials": 3, "seed": 555}))
    for check in ("split_check.json", "type_check.json"):
        (run / check).write_text("{}")
    fake_shared_cv(cv_scores).to_csv(run / "model_selection_cv.csv", index=False)
    folds = pd.read_csv(paths.cv_folds).set_index("node_id")["fold"]
    train_ids = pd.read_csv(paths.train_ids)["node_id"]
    fake_oof_predictions(
        list(cv_scores), train["target"].to_numpy(), folds.loc[train_ids].to_numpy()
    ).to_parquet(run / "oof_predictions.parquet", index=False)
    (paths.df_analyze_output_dir / "latest.txt").write_text(FAKE_RUN + "\n")
