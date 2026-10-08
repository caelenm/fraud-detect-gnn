"""Packing and unpacking pipeline output bundles.

Files here are placeholder bytes, not data rows.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from fraud_detect.artifacts import (
    MANIFEST_NAME,
    ArtifactError,
    collect_files,
    pack,
    unpack,
)
from fraud_detect.config import DEFAULT_CONFIG, get_paths, load_config, with_dataset

INCLUDED = [
    "data/processed/amazon/nodes.parquet",
    "data/processed/amazon/split_summary.json",
    "data/processed/amazon/df_analyze/m1_own/train.parquet",
    "outputs/amazon/reports/audit.md",
    "outputs/amazon/m1_own/reports/split_check.json",
    "outputs/amazon/m1_own/df_analyze/latest.txt",
    "outputs/amazon/m1_own/df_analyze/20260102T000000Z/results.csv",
]
EXCLUDED = [
    "data/raw/care_gnn/Amazon.mat",
    "data/processed/yelpchi/nodes.parquet",  # another dataset
    "outputs/yelpchi/reports/audit.md",
    "outputs/amazon/runs/20260101T000000Z/run_info.json",
    "outputs/amazon/checkpoints/select_model/unit.json",
    "outputs/amazon/pipeline_state.json",
    "outputs/amazon/m1_own/df_analyze/20260101T000000Z/results.csv",  # not the latest
]


def make_repo(root: Path) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")
    for name in INCLUDED + EXCLUDED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"placeholder for {name}".encode())
    (root / "outputs/amazon/m1_own/df_analyze/latest.txt").write_text(
        "20260102T000000Z\n"
    )
    return root


def paths_for(root: Path, dataset: str = "amazon"):
    return get_paths(with_dataset(load_config(DEFAULT_CONFIG), dataset), root=root)


def configs(root: Path) -> list[Path]:
    return [root / "configs" / "default.yaml"]


def test_collect_files_includes_outputs_and_skips_raw_runs_and_old_runs(tmp_path):
    root = make_repo(tmp_path)
    names = {p.relative_to(root).as_posix() for p in collect_files(paths_for(root))}
    assert names == set(INCLUDED)


def test_round_trip_restores_every_file(tmp_path):
    src = make_repo(tmp_path / "a")
    bundle = tmp_path / "bundle.tgz"
    manifest = pack(paths_for(src), bundle, configs(src))
    assert {e["path"] for e in manifest["files"]} == set(INCLUDED)

    dst = tmp_path / "b"
    (dst / "configs").mkdir(parents=True)
    (dst / "configs" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")
    result = unpack(bundle, dst, configs(dst))
    assert sorted(result.written) == sorted(INCLUDED)
    assert result.warnings == []
    for name in INCLUDED:
        assert (dst / name).read_bytes() == (src / name).read_bytes()
    assert not (dst / "data/raw").exists()
    with pytest.raises(ArtifactError, match="not 'yelpchi'"):
        unpack(bundle, dst, configs(dst), dataset="yelpchi")

    again = unpack(bundle, dst, configs(dst))
    assert again.written == []
    assert sorted(again.unchanged) == sorted(INCLUDED)


def test_conflicting_file_needs_force(tmp_path):
    src = make_repo(tmp_path / "a")
    bundle = tmp_path / "bundle.tgz"
    pack(paths_for(src), bundle, configs(src))
    dst = tmp_path / "b"
    changed = dst / "data/processed/amazon/split_summary.json"
    changed.parent.mkdir(parents=True)
    changed.write_bytes(b"different")

    with pytest.raises(ArtifactError, match="--force"):
        unpack(bundle, dst, [])
    assert not (dst / "data/processed/amazon/nodes.parquet").exists()  # nothing written

    unpack(bundle, dst, [], force=True)
    assert changed.read_bytes() == (src / INCLUDED[1]).read_bytes()


def test_changed_config_gives_warning(tmp_path):
    src = make_repo(tmp_path / "a")
    bundle = tmp_path / "bundle.tgz"
    pack(paths_for(src), bundle, configs(src))
    dst = tmp_path / "b"
    (dst / "configs").mkdir(parents=True)
    (dst / "configs" / "default.yaml").write_text("seed: 2\n", encoding="utf-8")
    result = unpack(bundle, dst, configs(dst))
    assert any("configs/default.yaml differs" in w for w in result.warnings)


def write_bundle(path: Path, files: dict[str, bytes], manifest_files: list[dict]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        members = {MANIFEST_NAME: json.dumps(
            {"format_version": 1, "files": manifest_files}
        ).encode(), **files}  # fmt: skip
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize("name", ["../evil.txt", "/tmp/evil.txt", "C:/evil.txt"])
def test_unsafe_paths_are_refused(tmp_path, name):
    bundle = tmp_path / "bad.tgz"
    write_bundle(bundle, {name: b"x"}, [{"path": name, "size": 1, "sha256": "0"}])
    dst = tmp_path / "repo"
    dst.mkdir()
    with pytest.raises(ArtifactError, match="Unsafe path"):
        unpack(bundle, dst, [])
    assert not (tmp_path / "evil.txt").exists()


def test_checksum_mismatch_leaves_no_file(tmp_path):
    bundle = tmp_path / "corrupt.tgz"
    name = "data/processed/amazon/split_summary.json"
    write_bundle(bundle, {name: b"x"}, [{"path": name, "size": 1, "sha256": "0" * 64}])
    dst = tmp_path / "repo"
    with pytest.raises(ArtifactError, match="Checksum mismatch"):
        unpack(bundle, dst, [])
    assert not (dst / name).exists()
    assert not (dst / (name + ".part")).exists()


def test_not_a_bundle(tmp_path):
    bundle = tmp_path / "other.tgz"
    with tarfile.open(bundle, "w:gz") as tar:
        info = tarfile.TarInfo("readme.txt")
        tar.addfile(info, io.BytesIO(b""))
    with pytest.raises(ArtifactError, match=MANIFEST_NAME):
        unpack(bundle, tmp_path, [])
