"""run.py stage selection and the dataset selector."""

from __future__ import annotations

import pytest

from fraud_detect.cli import main, select_stages
from fraud_detect.config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    ConfigError,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.pipeline import STAGE_NAMES, get_stage


def test_stage_order_matches_the_plan():
    assert STAGE_NAMES == [
        "download", "load", "split", "features", "audit", "neighbour_agg",
        "gnn_embed", "gnn_oof", "df_analyze_input", "df_analyze", "select_model",
        "df_analyze_report", "web_report", "export_model", "gnn_classify",
    ]  # fmt: skip


def test_every_stage_has_a_numbered_wrapper_script():
    scripts = sorted(p.name for p in (REPO_ROOT / "scripts").glob("[0-9][0-9]_*.py"))
    assert scripts == [f"{n:02d}_{name}.py" for n, name in enumerate(STAGE_NAMES, 1)]
    for n, name in enumerate(STAGE_NAMES, 1):
        text = (REPO_ROOT / "scripts" / f"{n:02d}_{name}.py").read_text(encoding="utf-8")
        assert f'"--only", "{name}"' in text and f"Stage {n}:" in text


def test_export_model_is_the_last_stage_of_a_df_analyze_feature_set():
    # run.py reruns every stage after the first one with missing outputs, so
    # export_model must be the last stage with outputs for a df-analyze
    # feature set, or adding a stage would rerun df-analyze. gnn_classify
    # comes after it but has outputs only for GNN classifier rungs.
    assert STAGE_NAMES[-2:] == ["export_model", "gnn_classify"]
    config = load_config(DEFAULT_CONFIG)
    for feature_set in config["feature_sets"]:
        p = get_paths(with_dataset({**config, "feature_set": feature_set}, "amazon"))
        if not p.is_gnn:
            assert get_stage("gnn_classify").outputs(p) == []
        else:  # a GNN rung: no df-analyze stage has outputs
            assert all(get_stage(s).outputs(p) == [] for s in STAGE_NAMES[9:14])


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
    assert amazon.blocks == ("own",)
    m5 = get_paths(with_dataset({**config, "feature_set": "m5_own_gnnssl"}, "amazon"))
    assert m5.blocks == ("own", "gnnssl")
    assert m5.block_spec("gnnssl").name == "gnnssl.spec.json"
    with pytest.raises(ConfigError, match="feature set"):
        with_dataset({**config, "feature_set": "m9_nothing"}, "yelpchi")
