"""run.py stage selection and the dataset selector."""

from __future__ import annotations

import pytest

from fraud_detect.cli import main, select_stages
from fraud_detect.config import (
    DEFAULT_CONFIG,
    ConfigError,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.pipeline import STAGE_NAMES


def test_stage_order_matches_the_plan():
    assert STAGE_NAMES == [
        "download", "load", "split", "features", "audit", "df_analyze_input",
        "df_analyze", "select_model", "df_analyze_report", "web_report",
    ]  # fmt: skip


def test_select_all_by_default():
    assert select_stages(None, None, None) == STAGE_NAMES


def test_select_range_and_only():
    assert select_stages("load", "split", None) == ["load", "split"]
    assert select_stages(None, None, "audit") == ["audit"]


def test_invalid_selections():
    with pytest.raises(ValueError):
        select_stages("split", "load", None)
    with pytest.raises(ValueError):
        select_stages("load", None, "audit")


def test_list_runs_without_side_effects(capsys):
    assert main(["--list", "--dataset", "yelpchi"]) == 0
    out = capsys.readouterr().out
    assert "download" in out and "web_report" in out


def test_a_dataset_must_be_chosen(capsys):
    assert main(["--only", "load"]) == 2
    assert "--dataset" in capsys.readouterr().err
    assert main(["--only", "load", "--dataset", "elliptic"]) == 2
    assert "Unknown dataset" in capsys.readouterr().err


def test_paths_are_separate_per_dataset_and_feature_set():
    config = load_config(DEFAULT_CONFIG)
    yelp = get_paths(with_dataset(config, "yelpchi"))
    amazon = get_paths(with_dataset(config, "amazon"))
    assert yelp.processed_dir.name == "yelpchi" and amazon.outputs_dir.name == "amazon"
    assert yelp.raw_dir == amazon.raw_dir  # downloads are shared
    assert yelp.mat.name == "YelpChi.mat" and amazon.mat.name == "Amazon.mat"
    assert yelp.web_report.parts[-4:] == ("yelpchi", "m1_own", "report", "index.html")
    with pytest.raises(ConfigError, match="feature set"):
        with_dataset({**config, "feature_set": "m9_nothing"}, "yelpchi")
