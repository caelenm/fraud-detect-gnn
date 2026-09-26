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
from fraud_detect.config import DEFAULT_CONFIG, get_paths, load_config

INCLUDED = [
    "data/interim/complaints.parquet",
    "data/processed/split.csv",
    "data/processed/df_analyze/train.parquet",
    "outputs/reports/df_analyze_split_check.json",
    "outputs/df_analyze/latest.txt",
    "outputs/df_analyze/20260102T000000Z/results.csv",
]
EXCLUDED = [
    "data/raw/archive.csv",
    "data/interim/embed/embed_output.parquet",
    "outputs/runs/20260101T000000Z/run_info.json",
    "outputs/df_analyze/20260101T000000Z/results.csv",  # not the latest run
]


def make_repo(root: Path) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")
    for name in INCLUDED + EXCLUDED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"placeholder for {name}".encode())
    (root / "outputs/df_analyze/latest.txt").write_text("20260102T000000Z\n")
    return root


def paths_for(root: Path):
    return get_paths(load_config(DEFAULT_CONFIG), root=root)


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

    again = unpack(bundle, dst, configs(dst))
    assert again.written == []
    assert sorted(again.unchanged) == sorted(INCLUDED)


def test_conflicting_file_needs_force(tmp_path):
    src = make_repo(tmp_path / "a")
    bundle = tmp_path / "bundle.tgz"
    pack(paths_for(src), bundle, configs(src))
    dst = tmp_path / "b"
    changed = dst / "data/processed/split.csv"
    changed.parent.mkdir(parents=True)
    changed.write_bytes(b"different")

    with pytest.raises(ArtifactError, match="--force"):
        unpack(bundle, dst, [])
    assert not (dst / "data/interim/complaints.parquet").exists()  # nothing written

    unpack(bundle, dst, [], force=True)
    assert changed.read_bytes() == (src / "data/processed/split.csv").read_bytes()


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
    name = "data/processed/split.csv"
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
