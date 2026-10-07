"""Stopping and resuming: run state, per-unit checkpoints, and the CLI guards."""

from __future__ import annotations

from helpers import tmp_config

from fraud_detect.cli import main, plan_stages
from fraud_detect.config import get_paths, with_dataset
from fraud_detect.pipeline import STAGE_NAMES, Context
from fraud_detect.runstate import RunState, UnitStore, config_fingerprint


def test_run_state_round_trip_and_cleanup(tmp_path):
    state = RunState(pending=["a", "b"], config_sha256="x")
    state.save(tmp_path)
    loaded = RunState.load(tmp_path)
    assert loaded is not None and loaded.pending == ["a", "b"]
    loaded.mark_done("a", tmp_path)
    assert RunState.load(tmp_path).pending == ["b"]
    loaded.mark_done("b", tmp_path)
    assert RunState.load(tmp_path) is None  # file removed when nothing is left


def test_unit_store_saves_skips_and_resets(tmp_path):
    store = UnitStore(tmp_path / "gnn")
    assert not store.done("graphsage/seed-1")
    store.save("graphsage/seed-1", {"pr_auc": 0.5})
    assert store.done("graphsage/seed-1")
    assert not store.done("graphsage/seed-2")
    assert store.load("graphsage/seed-1")["pr_auc"] == 0.5
    assert [u["unit"] for u in store.all()] == ["graphsage/seed-1"]
    store.reset()
    assert store.all() == []


def test_unit_store_is_emptied_on_a_fresh_run_and_kept_on_resume(tmp_path):
    _, config = tmp_config(tmp_path)
    paths = get_paths(config)
    fresh = Context(config=config, paths=paths, run_dir=tmp_path)
    fresh.unit_store("gnn").save("m/seed-1", {})
    resumed = Context(config=config, paths=paths, run_dir=tmp_path, resume=True)
    assert resumed.unit_store("gnn").done("m/seed-1")
    assert not fresh.unit_store("gnn").done("m/seed-1")


def test_plan_starts_at_first_missing_output_and_runs_to_the_end(tmp_path):
    _, config = tmp_config(tmp_path)
    paths = get_paths(config)
    assert plan_stages(STAGE_NAMES, force=False, paths=paths) == STAGE_NAMES
    assert plan_stages(STAGE_NAMES[3:], force=True, paths=paths) == STAGE_NAMES[3:]


def test_cli_resume_guards(tmp_path, capsys):
    path, config = tmp_config(tmp_path)
    outputs = get_paths(config).outputs_dir
    assert main(["--config", str(path), "--resume"]) == 0
    assert "Nothing to resume" in capsys.readouterr().out

    RunState(pending=["audit"], config_sha256="another-config").save(outputs)
    assert main(["--config", str(path), "--resume"]) == 2  # config changed
    assert "config changed" in capsys.readouterr().err

    RunState(pending=["audit"], config_sha256=config_fingerprint(config)).save(outputs)
    assert main(["--config", str(path), "--only", "audit"]) == 2  # unfinished run
    assert "--resume" in capsys.readouterr().err
    assert main(["--config", str(path), "--resume", "--force"]) == 2


def test_run_state_is_per_dataset(tmp_path, capsys):
    path, config = tmp_config(tmp_path, dataset="amazon")
    RunState(pending=["audit"], config_sha256=config_fingerprint(config)).save(
        get_paths(config).outputs_dir
    )
    # An interrupted Amazon run neither blocks nor is resumed by a YelpChi run.
    assert main(["--config", str(path), "--dataset", "yelpchi", "--resume"]) == 0
    assert "Nothing to resume" in capsys.readouterr().out
    yelp = get_paths(with_dataset(config, "yelpchi"))
    assert RunState.load(yelp.outputs_dir) is None
    assert RunState.load(get_paths(config).outputs_dir).pending == ["audit"]
