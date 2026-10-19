"""A clone that has only an unpacked bundle never retrains anything.

The bundle holds no raw data, so the download stage must count as done once
the load stage's outputs exist. The graph block stages count as done for a
feature set that does not use their block, so an m1_own bundle made before
they existed still skips everything. The dataset is synthetic and the
df-analyze run and model export are faked (tests/helpers.py); everything is
in tmp_path.
"""

from __future__ import annotations

import yaml
from helpers import fake_dataset, fake_df_analyze_run, fake_export_outputs

from fraud_detect.artifacts import pack, unpack
from fraud_detect.cli import main, plan_stages
from fraud_detect.config import get_paths, with_dataset
from fraud_detect.pipeline import STAGE_NAMES

CV = {("dummy", "none"): 0.2, ("lgbm", "none"): 0.6}


def unpacked_m1_bundle(tmp_path):
    """Run (and fake) every m1_own stage in clone a, pack it, and unpack it
    into clone b, which has its own folders and no way to download. Returns
    clone b's config file, config and root."""
    a = tmp_path / "a"
    a.mkdir()
    path_a, config_a = fake_dataset(a, n_nodes=200)
    assert main(["--config", str(path_a), "--to", "df_analyze_input"]) == 0
    paths_a = get_paths(config_a, root=a)
    fake_df_analyze_run(paths_a, CV)
    args = ["--config", str(path_a), "--from", "df_analyze_report", "--to", "web_report"]
    assert main(args) == 0
    fake_export_outputs(paths_a)
    assert plan_stages(STAGE_NAMES, False, paths_a) == []

    bundle = tmp_path / "bundle.tgz"
    pack(paths_a, bundle, [path_a])
    b = tmp_path / "b"
    unpack(bundle, b, [])

    # Another clone: the same config with its own folders, and no way to
    # download (the bundle has no raw files, and none may be needed).
    config_b = yaml.safe_load(path_a.read_text(encoding="utf-8"))
    for key in ("raw_dir", "processed_dir", "outputs_dir"):
        config_b["paths"][key] = str(b / key)
    config_b["download"]["base_url"] = (tmp_path / "no_server").as_uri()
    path_b = b / "config.yaml"
    path_b.write_text(yaml.safe_dump(config_b), encoding="utf-8")
    return path_b, config_b, b


def test_unpacked_bundle_leaves_no_stage_to_run(tmp_path, capsys):
    path_b, config_b, b = unpacked_m1_bundle(tmp_path)
    paths_b = get_paths(with_dataset(config_b, None), root=b)
    assert not paths_b.mat.exists()
    assert plan_stages(STAGE_NAMES, False, paths_b) == []

    capsys.readouterr()
    assert main(["--config", str(path_b)]) == 0
    out = capsys.readouterr().out
    assert "Stages to run" not in out
    assert out.count("outputs exist, skipping") == len(STAGE_NAMES)
    assert not paths_b.raw_dir.exists()


def test_a_new_feature_set_runs_only_its_block_and_the_model_stages(tmp_path):
    _, config_b, b = unpacked_m1_bundle(tmp_path)

    def planned(feature_set: str) -> list[str]:
        config = with_dataset({**config_b, "feature_set": feature_set}, None)
        return plan_stages(STAGE_NAMES, False, get_paths(config, root=b))

    model_stages = STAGE_NAMES[STAGE_NAMES.index("df_analyze_input") :]
    assert planned("m5_own_gnnssl") == ["gnn_embed", *model_stages]
    # gnn_embed is listed because it comes later, but has nothing to build.
    assert planned("m2_own_nbr") == ["neighbour_agg", "gnn_embed", *model_stages]
    # The same blocks under another name: only the model stages.
    assert planned("pilot_m1_own") == model_stages
