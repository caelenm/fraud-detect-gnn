"""Split repeats (split.n_repeats): folders, seeds, the CLI loop and the summary.

Uses a fake dataset in tmp_path; summaries are invented numbers.
"""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest
import yaml
from helpers import fake_dataset

from fraud_detect.cli import main, write_repeat_summary
from fraud_detect.config import ConfigError, get_paths, with_dataset
from fraud_detect.models import repeats


def with_repeats(path, n: int) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["split"]["n_repeats"] = n
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_each_repeat_has_its_own_folders_and_split(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=200)
    with_repeats(path, 2)
    assert main(["--config", str(path), "--to", "split"]) == 0
    first = get_paths(config)
    second_cfg = copy.deepcopy(config)
    second_cfg["split"]["repeat"] = 1
    second = get_paths(second_cfg)
    assert first.processed_dir.name == "fake"
    assert second.processed_dir.name == "fake_repeat1"
    assert first.raw_dir == second.raw_dir  # one download
    a, b = pd.read_csv(first.test_ids), pd.read_csv(second.test_ids)
    assert set(a["node_id"]) != set(b["node_id"])  # independently seeded
    seeds = [json.loads(p.split_summary.read_text())["seed"] for p in (first, second)]
    assert seeds == [555, 1555]


def test_one_repeat_can_be_run_on_its_own(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=200)
    with_repeats(path, 3)
    assert main(["--config", str(path), "--to", "split", "--set", "split.repeat=2"]) == 0
    cfg = copy.deepcopy(config)
    for repeat, exists in ((0, False), (1, False), (2, True)):
        cfg["split"]["repeat"] = repeat
        assert get_paths(cfg).test_ids.exists() is exists


def test_repeat_must_be_in_range(tmp_path):
    _, config = fake_dataset(tmp_path)
    config["split"]["repeat"] = 1
    with pytest.raises(ConfigError, match="outside"):
        with_dataset(config, None)


def fake_summary(run: str, pr_auc: float) -> dict:
    model_a = {"model": "lgbm", "selection": "none", "pr_auc": pr_auc, "auroc": 0.9,
               "f1": 0.5, "precision": 0.5, "recall": 0.5}  # fmt: skip
    return {"run": run, "test_positive_rate": 0.1, "model_a": model_a}


def test_summary_is_mean_and_sd_over_repeats(tmp_path):
    summary = repeats.summarize([fake_summary("a", 0.6), fake_summary("b", 0.8)])
    pr = summary["metrics"]["pr_auc"]
    assert pr["mean"] == pytest.approx(0.7)
    assert pr["sd"] == pytest.approx(0.1414, abs=1e-4)
    assert "| pr_auc | 0.7000 | 0.1414 |" in repeats.markdown(summary, "fake", "m1_own")
    with pytest.raises(ValueError):
        repeats.summarize([fake_summary("a", 0.6)])

    _, config = fake_dataset(tmp_path)
    config["split"]["n_repeats"] = 2
    for repeat, value in ((0, 0.6), (1, 0.8)):
        cfg = copy.deepcopy(config)
        cfg["split"]["repeat"] = repeat
        out = get_paths(cfg).model_reports_dir / "model_a.json"
        out.parent.mkdir(parents=True)
        out.write_text(json.dumps(fake_summary(f"run{repeat}", value)))
    write_repeat_summary(config, 2)
    md = (get_paths(config).feature_set_dir / "repeats_summary.md").read_text()
    assert "Model A over 2 grouped splits" in md
