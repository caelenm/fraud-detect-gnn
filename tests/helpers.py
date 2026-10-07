"""Test helpers: temporary configs and fake dataset files under tmp_path.

Nothing here is real data; the fake .mat contents come from synthetic.py.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp
import yaml
from synthetic import fake_care_gnn_mat

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
