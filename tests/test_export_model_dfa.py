"""export_model and predict.py for real, in df-analyze's environment.

A tiny model is fit with df-analyze's own classes on synthetic tables and
saved with df-analyze's own EvaluationResults.save (tests/dfa/
tiny_htune_run.py), so it goes through jsonpickle exactly as a real run's
models do. Then the export_model stage decodes it, verifies it and saves it
natively, and scripts/dfa/predict.py reproduces df-analyze's saved test
probabilities from the export. Everything is in tmp_path.

Skipped when df-analyze's clone (at the pinned commit) or uv is not
available. Each df-analyze subprocess takes several seconds to start.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from helpers import FAKE_RUN, fake_dataset, fake_df_analyze_run

from fraud_detect.cli import main
from fraud_detect.config import REPO_ROOT, get_paths
from fraud_detect.config import df_analyze_dir as find_df_analyze
from fraud_detect.external import df_analyze_commit, run_df_analyze_script
from fraud_detect.models import model_export as mx

CV = {("dummy", "none"): 0.2, ("lr", "none"): 0.6}  # shared CV makes lr Model A


def _df_analyze_ready(config: dict) -> bool:
    dfa = find_df_analyze(config)
    return (
        shutil.which("uv") is not None
        and (dfa / "df-analyze.py").is_file()
        and df_analyze_commit(dfa) == config["df_analyze"]["commit"]
    )


@pytest.fixture
def exported(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=200)
    if not _df_analyze_ready(config):
        pytest.skip("df-analyze (at the pinned commit) or uv is not available")
    assert main(["--config", str(path), "--to", "df_analyze_input"]) == 0
    p = get_paths(config)
    fake_df_analyze_run(p, CV)
    # Replace the fake results with real, jsonpickled df-analyze models.
    deep = p.df_analyze_output_dir / FAKE_RUN / "train" / "hash"
    run_df_analyze_script(
        find_df_analyze(config),
        str(REPO_ROOT / "tests" / "dfa" / "tiny_htune_run.py"),
        ["--train", str((p.df_analyze_input_dir / "train.parquet").resolve()),
         "--test", str((p.df_analyze_input_dir / "test.parquet").resolve()),
         "--out", str(deep.resolve())],
        tmp_path / "tiny_run.log",
        python=config["df_analyze"]["python"],
    )  # fmt: skip
    assert main(["--config", str(path), "--from", "df_analyze_report"]) == 0
    return config, p, deep / "results" / "test00"


def predict(config: dict, model_dir, table, out, *extra: str):
    """scripts/dfa/predict.py in df-analyze's environment, as a teammate runs it."""
    dfa = find_df_analyze(config)
    env = {**os.environ, "DF_ANALYZE_DIR": str(dfa)}
    env.pop("VIRTUAL_ENV", None)
    script = REPO_ROOT / "scripts" / "dfa" / "predict.py"
    cmd = ["uv", "run", "--project", str(dfa), "--python", config["df_analyze"]["python"],
           "python", str(script), "--model-dir", str(model_dir), "--input", str(table),
           "--out", str(out), *extra]  # fmt: skip
    return subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)


def saved_probs(results_dir, model_cls: str) -> np.ndarray:
    entries = json.loads((results_dir / "prediction_results_00.json").read_text())
    (entry,) = [e for e in entries["predictions"] if e["model_cls"] == model_cls]
    return np.asarray(entry["probs_test"])[:, 1]


def test_model_a_is_exported_verified_and_predicts_without_retraining(exported, tmp_path):
    config, p, results_dir = exported
    meta = mx.read_metadata(p.model_dir / mx.MODEL_A_META)
    assert meta["model"] == "lr" and meta["source"] == "decoded"
    assert meta["model_format"] == "joblib"
    assert meta["verification"]["max_abs_diff_native"] <= mx.CPU_TOLERANCE
    assert (p.model_dir / meta["model_file"]).is_file()
    check = pd.read_csv(p.model_dir / mx.MODELS_CHECK)
    assert set(check["model"]) == {"dummy", "lr"}
    assert check["decoded"].eq("yes").all() and check["ok"].all()

    x_test = results_dir / "X_test_00.csv"
    out = tmp_path / "pred_a.csv"
    done = predict(config, p.model_dir, x_test, out)
    assert done.returncode == 0, done.stderr
    pred = pd.read_csv(out)
    expected = saved_probs(results_dir, "LRClassifier")
    np.testing.assert_allclose(
        pred["prob_fraud"], expected, rtol=0, atol=mx.CPU_TOLERANCE
    )
    assert pred["label"].tolist() == (pred["prob_fraud"] >= meta["threshold"]).tolist()

    # Any other combination loads straight from df-analyze's saved results.
    out = tmp_path / "pred_dummy.csv"
    done = predict(config, p.model_dir, x_test, out, "--model", "dummy",
                   "--selection", "none")  # fmt: skip
    assert done.returncode == 0, done.stderr
    expected = saved_probs(results_dir, "DummyClassifier")
    got = pd.read_csv(out)["prob_fraud"]
    np.testing.assert_allclose(got, expected, rtol=0, atol=mx.CPU_TOLERANCE)

    # Columns in another order are refused, not silently mispredicted.
    shuffled = tmp_path / "shuffled.csv"
    table = pd.read_csv(x_test)
    table[table.columns[::-1]].to_csv(shuffled, index=False)
    done = predict(config, p.model_dir, shuffled, tmp_path / "never.csv")
    assert done.returncode != 0 and "do not match" in done.stderr
    assert not (tmp_path / "never.csv").exists()
