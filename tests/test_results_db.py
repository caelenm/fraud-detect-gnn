"""The results database: entries, their checks, and the committed results/ folder."""

from __future__ import annotations

import importlib.util
import json

import pandas as pd
import pytest
import yaml
from synthetic import fake_model_a_summary, fake_results_table

from fraud_detect import results_db as db
from fraud_detect.config import REPO_ROOT


def load_script():
    spec = importlib.util.spec_from_file_location(
        "results_db_script", REPO_ROOT / "scripts" / "results_db.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_meta(**fields):
    return db.build_meta(
        description="Synthetic entry", device="test-box",
        started_utc="2099-01-01T10:00:00Z", **fields,
    )  # fmt: skip


def test_committed_results_folder_passes_every_check():
    # Whatever teammates have committed under results/runs/ must be valid.
    assert db.check_db(REPO_ROOT / "results" / "runs") == []


def test_add_then_read_back(tmp_path):
    meta = fake_meta(dataset="amazon", feature_set="m1_own",
                     finished_utc="2099-01-01T11:30:00Z", tags=["pilot"])  # fmt: skip
    entry = db.add_entry(tmp_path, meta, fake_results_table())
    assert entry.name == meta["run_id"] and entry.name.startswith("20990101T100000Z_")
    saved = db.load_meta(entry)
    assert saved["duration_min"] == 90.0
    assert saved["device"] == "test-box" and "os" in saved["device_details"]
    assert saved["results_rows"] == 3
    assert db.check_db(tmp_path) == []
    index = db.index_table(tmp_path)
    assert list(index["run_id"]) == [entry.name]
    assert index["tags"].iloc[0] == "pilot"
    combined = db.combined_results(tmp_path)
    assert len(combined) == 3 and set(combined["run_id"]) == {entry.name}


def test_run_ids_never_collide():
    ids = {fake_meta(dataset="amazon")["run_id"] for _ in range(50)}
    assert len(ids) == 50


@pytest.mark.parametrize("column", ["node_id", "label", "group_id", "split", "prob",
                                    "Target"])  # fmt: skip
def test_row_level_columns_are_refused(tmp_path, column):
    table = fake_results_table().assign(**{column: 0})
    with pytest.raises(db.ResultsDBError, match="row-level"):
        db.add_entry(tmp_path, fake_meta(), table)
    assert db.entry_dirs(tmp_path) == []


def test_long_tables_are_refused(tmp_path):
    with pytest.raises(db.ResultsDBError, match="rows"):
        db.add_entry(tmp_path, fake_meta(), fake_results_table(db.MAX_ROWS + 1))


def test_times_must_be_utc():
    assert db.parse_utc("2099-01-01T12:00:00+02:00") == "2099-01-01T10:00:00Z"
    with pytest.raises(db.ResultsDBError, match="time zone"):
        db.parse_utc("2099-01-01T12:00:00")
    with pytest.raises(db.ResultsDBError, match="before"):
        fake_meta(finished_utc="2099-01-01T09:00:00Z")


def test_description_is_one_short_line(tmp_path):
    for text in ("two\nlines", "x" * (db.MAX_DESCRIPTION + 1)):
        meta = fake_meta()
        meta["description"] = text
        with pytest.raises(db.ResultsDBError, match="description"):
            db.add_entry(tmp_path, meta, fake_results_table())


def test_check_finds_edited_or_extra_files(tmp_path):
    entry = db.add_entry(tmp_path, fake_meta(), fake_results_table())
    (entry / "predictions.csv").write_text("a\n1\n", encoding="utf-8")
    assert "unexpected files" in db.check_db(tmp_path)[0]
    (entry / "predictions.csv").unlink()
    fake_results_table().assign(node_id=1).to_csv(entry / db.RESULTS, index=False)
    assert "row-level" in db.check_db(tmp_path)[0]


def fake_pipeline_outputs(tmp_path):
    """A finished feature-set folder and its run log, as the pipeline writes them."""
    fs_dir = tmp_path / "outputs" / "amazon_repeat2" / "pilot_m1_own"
    (fs_dir / "reports").mkdir(parents=True)
    summary = fake_model_a_summary(feature_set="pilot_m1_own")
    (fs_dir / "reports" / "model_a.json").write_text(json.dumps(summary), "utf-8")
    fake_results_table().to_csv(fs_dir / "reports" / "model_a_metrics.csv", index=False)
    log = tmp_path / "outputs" / "amazon_repeat2" / "runs" / "20200101T100000000000Z"
    log.mkdir(parents=True)
    info = {"started_utc": "2020-01-01T10:00:00.123456+00:00",
            "argv": ["run.py", "--dataset", "amazon"], "git_commit": "abc123",
            "packages": {"numpy": "0"}, "stages": ["df_analyze"]}  # fmt: skip
    (log / "run_info.json").write_text(json.dumps(info), "utf-8")
    config = {"dataset": "amazon", "feature_set": "pilot_m1_own", "seed": 1,
              "split": {"repeat": 2, "n_repeats": 3},
              "df_analyze": {"classifiers": ["lgbm"], "htune_trials": 3},
              "feature_sets": {"pilot_m1_own": ["own"]}}  # fmt: skip
    (log / "config.yaml").write_text(yaml.safe_dump(config), "utf-8")
    return fs_dir, log


def test_script_records_a_pipeline_run(tmp_path, capsys):
    fs_dir, log = fake_pipeline_outputs(tmp_path)
    script, store = load_script(), tmp_path / "db"
    code = script.main(["--db", str(store), "add", "--from-run", str(fs_dir),
                        "--run-log", str(log), "--device", "test-box",
                        "--description", "Synthetic pipeline run"])  # fmt: skip
    assert code == 0
    (entry,) = db.entry_dirs(store)
    meta = db.load_meta(entry)
    assert meta["dataset"] == "amazon" and meta["feature_set"] == "pilot_m1_own"
    assert meta["split_repeat"] == 2 and meta["started_utc"] == "2020-01-01T10:00:00Z"
    assert meta["headline"]["test_pr_auc"] == 0.5
    assert meta["headline"]["test_pr_auc_low"] == 0.4
    assert meta["settings"]["df_analyze.classifiers"] == ["lgbm"]
    assert meta["settings"]["blocks"] == ["own"]
    assert meta["command"] == "run.py --dataset amazon"
    assert "stages" not in meta
    pd.testing.assert_frame_equal(
        pd.read_csv(entry / db.RESULTS),
        pd.read_csv(fs_dir / "reports" / "model_a_metrics.csv"),
    )
    capsys.readouterr()
    assert script.main(["--db", str(store), "list", "--dataset", "amazon"]) == 0
    assert entry.name in capsys.readouterr().out
    assert script.main(["--db", str(store), "show", entry.name[:20]]) == 0
    assert script.main(["--db", str(store), "check"]) == 0
    out = tmp_path / "export"
    assert script.main(["--db", str(store), "export", "--out", str(out)]) == 0
    assert len(pd.read_csv(out / "index.csv")) == 1
    assert len(pd.read_csv(out / "all_results.csv")) == 3


def test_script_needs_a_start_time_and_a_table(tmp_path, capsys):
    script = load_script()
    csv = tmp_path / "summary.csv"
    fake_results_table().to_csv(csv, index=False)
    base = ["--db", str(tmp_path / "db"), "add", "--device", "test-box",
            "--description", "Synthetic probe", "--kind", "probe"]  # fmt: skip
    assert script.main([*base, "--csv", str(csv)]) == 2
    assert "start time" in capsys.readouterr().err
    assert script.main([*base, "--started-utc", "2099-01-01T10:00:00Z"]) == 2
    assert "--csv" in capsys.readouterr().err
    assert script.main([*base, "--csv", str(csv), "--dry-run",
                        "--started-utc", "2099-01-01T10:00:00Z"]) == 0  # fmt: skip
    assert db.entry_dirs(tmp_path / "db") == []
    assert script.main([*base, "--csv", str(csv),
                        "--started-utc", "2099-01-01T10:00:00Z"]) == 0  # fmt: skip
    assert len(db.entry_dirs(tmp_path / "db")) == 1
