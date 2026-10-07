"""Pack and unpack pipeline outputs so group members can skip long stages.

A bundle is one .tgz holding every stage output of one dataset under
data/processed/<dataset>/ and outputs/<dataset>/ (never the raw downloads)
plus a manifest with a SHA-256 checksum for each file.
Unpacking it into another clone puts each file back in the same place, so
`uv run run.py` there skips every stage that is already done, exactly as it
does on the machine that made the bundle.

Only the Python standard library is used for archiving, so this works the same
on Linux, macOS and Windows.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import tarfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from fraud_detect.config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    ConfigError,
    Paths,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.runlog import git_commit
from fraud_detect.runstate import CHECKPOINT_DIR, STATE_FILE

MANIFEST_NAME = "artifacts_manifest.json"
BUNDLE_PREFIX = "fraud_artifacts_"
BUNDLE_SUFFIX = ".tgz"
FORMAT_VERSION = 1
CHUNK = 1 << 20


class ArtifactError(RuntimeError):
    """A bundle cannot be packed or unpacked; the message says why."""


@dataclass
class UnpackResult:
    written: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _files_under(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [p for p in directory.rglob("*") if p.is_file()]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as e:
        raise ArtifactError(f"{path} is outside the repository ({root})") from e


def _latest_df_analyze_run(dfa_dir: Path) -> Path | None:
    latest = dfa_dir / "latest.txt"
    if not latest.is_file():
        return None
    stamp = latest.read_text(encoding="utf-8").strip()
    return dfa_dir / stamp if stamp and (dfa_dir / stamp).is_dir() else None


def collect_files(paths: Paths) -> list[Path]:
    """Every file of the active dataset that goes into a bundle.

    Included: data/processed/<dataset>/ and outputs/<dataset>/, except the
    per-invocation run logs (outputs/<dataset>/runs) and the resume
    bookkeeping. Of each feature set's df-analyze runs, only the latest
    finished one (named in its df_analyze/latest.txt) is included. The raw
    downloads are never included: everyone fetches them with the download
    stage, which verifies their checksums.
    """
    files = _files_under(paths.processed_dir)
    # Resume bookkeeping belongs to one machine's interrupted run; never share it.
    skip = (paths.runs_dir, paths.outputs_dir / CHECKPOINT_DIR)
    state = paths.outputs_dir / STATE_FILE
    for p in _files_under(paths.outputs_dir):
        if p == state or any(p.is_relative_to(d) for d in skip):
            continue
        run_dir = _df_analyze_run_dir(p)
        if run_dir is not None and run_dir != _latest_df_analyze_run(run_dir.parent):
            continue  # an older df-analyze run
        files.append(p)
    return sorted(set(files))


def _df_analyze_run_dir(path: Path) -> Path | None:
    """The df-analyze run directory (<...>/df_analyze/<stamp>) holding `path`."""
    for parent in path.parents:
        if parent.parent.name == "df_analyze":
            return parent
    return None


def pack(paths: Paths, out: Path, config_files: list[Path]) -> dict[str, Any]:
    """Write a bundle of the current outputs to `out` and return its manifest."""
    files = collect_files(paths)
    if not files:
        raise ArtifactError(
            "Nothing to pack: no stage outputs under data/ or outputs/. "
            "Run the pipeline first."
        )
    entries = [
        {
            "path": _relative(p, paths.root),
            "size": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in files
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": paths.processed_dir.name,
        "git_commit": git_commit(paths.root),
        "platform": platform.platform(),
        "configs": {
            _relative(c, paths.root): sha256_file(c) for c in config_files if c.is_file()
        },
        "files": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    # Parquet files are already compressed, so a low gzip level is enough.
    with tarfile.open(tmp, "w:gz", compresslevel=1) as tar:
        data = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(data)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(data))  # first member, read before the rest
        for entry in entries:
            tar.add(paths.root / entry["path"], arcname=entry["path"], recursive=False)
    os.replace(tmp, out)
    return manifest


def _safe_dest(root: Path, name: str) -> Path:
    """Where a bundle member goes; refuses absolute paths and '..' on any OS."""
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if (
        not name
        or posix.is_absolute()
        or windows.anchor
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise ArtifactError(f"Unsafe path in bundle: {name!r}")
    return root.joinpath(*posix.parts)


def _read_manifest(tar: tarfile.TarFile) -> dict[str, Any]:
    first = tar.next()
    if first is None or first.name != MANIFEST_NAME or not first.isfile():
        raise ArtifactError(
            f"{MANIFEST_NAME} is missing: this is not a bundle made by pack_artifacts.py"
        )
    f = tar.extractfile(first)
    assert f is not None
    manifest = json.loads(f.read().decode("utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ArtifactError(
            f"Bundle format {manifest.get('format_version')} is not supported "
            f"(expected {FORMAT_VERSION}); update this repository and retry."
        )
    return manifest


def compare_environment(
    manifest: dict[str, Any], root: Path, config_files: list[Path]
) -> list[str]:
    """Warnings about differences between the bundle's clone and this one."""
    warnings = []
    ours = git_commit(root)
    theirs = manifest.get("git_commit")
    if ours and theirs and ours != theirs:
        warnings.append(
            f"The bundle was made at commit {theirs[:10]}, this clone is at {ours[:10]}. "
            "If stage code changed in between, rerun those stages with --force."
        )
    recorded = manifest.get("configs", {})
    for c in config_files:
        name = _relative(c, root)
        if name in recorded and c.is_file() and sha256_file(c) != recorded[name]:
            warnings.append(
                f"{name} differs from the one used to make the bundle. Stages whose "
                "settings changed must be rerun with --force."
            )
    return warnings


def _extract_verified(
    tar: tarfile.TarFile, member: tarfile.TarInfo, dest: Path, expected: str
) -> None:
    """Write one member via a temporary file and keep it only if its checksum
    matches, so an interrupted or corrupt unpack never leaves a partial file
    that would make run.py skip a stage."""
    src = tar.extractfile(member)
    assert src is not None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    with tmp.open("wb") as out:
        for block in iter(lambda: src.read(CHUNK), b""):
            h.update(block)
            out.write(block)
    if h.hexdigest() != expected:
        tmp.unlink()
        raise ArtifactError(
            f"Checksum mismatch for {member.name}: the bundle is corrupt or "
            "incomplete. Download it again."
        )
    os.replace(tmp, dest)


def unpack(
    bundle: Path,
    root: Path,
    config_files: list[Path],
    force: bool = False,
    dataset: str | None = None,
) -> UnpackResult:
    """Unpack a bundle into the repository at `root`.

    Files already present with the same checksum are left alone. If any file
    exists with different contents, nothing is written unless `force` is set.
    With `dataset`, the bundle must have been packed for that dataset.
    """
    result = UnpackResult()
    with tarfile.open(bundle, "r:gz") as tar:
        manifest = _read_manifest(tar)
        packed = manifest.get("dataset")
        if dataset is not None and packed != dataset:
            raise ArtifactError(
                f"{bundle.name} holds dataset {packed!r}, not {dataset!r}"
            )
        entries = {e["path"]: e for e in manifest["files"]}
        result.warnings = compare_environment(manifest, root, config_files)

        conflicts = []
        for name, entry in entries.items():
            dest = _safe_dest(root, name)
            if not dest.exists():
                continue
            if dest.is_file() and sha256_file(dest) == entry["sha256"]:
                result.unchanged.append(name)
            else:
                conflicts.append(name)
        if conflicts and not force:
            listed = "\n  ".join(conflicts[:20])
            more = (
                f"\n  ... and {len(conflicts) - 20} more" if len(conflicts) > 20 else ""
            )
            raise ArtifactError(
                "These files already exist and differ from the bundle:\n  "
                f"{listed}{more}\nRerun with --force to overwrite them."
            )

        skip = set(result.unchanged)
        for member in tar:
            if member.name == MANIFEST_NAME:  # iteration restarts at the first member
                continue
            entry = entries.get(member.name)
            if entry is None or not member.isfile():
                raise ArtifactError(f"Unexpected entry in bundle: {member.name!r}")
            if member.name in skip:
                continue
            _extract_verified(tar, member, _safe_dest(root, member.name), entry["sha256"])
            result.written.append(member.name)

    missing = set(entries) - set(result.written) - skip
    if missing:
        raise ArtifactError(
            f"The bundle is incomplete: {len(missing)} files listed in its manifest "
            "are missing. Download it again."
        )
    return result


# --------------------------------------------------------------------------
# Command line (pack_artifacts.py and unpack_artifacts.py in the repo root)
# --------------------------------------------------------------------------
def _config_files(config_path: Path, paths: Paths) -> list[Path]:
    """Config files recorded in (and compared against) a bundle's manifest."""
    candidates = [config_path.resolve()]
    return [c for c in candidates if c.is_relative_to(paths.root.resolve())]


def _paths(config_path: Path, dataset: str | None) -> Paths:
    return get_paths(with_dataset(load_config(config_path), dataset))


def _mb(n: float) -> str:
    return f"{n / 1e6:,.0f} MB"


def print_stage_status(paths: Paths) -> None:
    """Show which stages run.py will skip because their outputs exist."""
    from fraud_detect.pipeline import STAGES  # noqa: PLC0415  (heavy imports)

    for n, stage in enumerate(STAGES, start=1):
        done = all(o.exists() for o in stage.outputs(paths))
        print(f"  {n:>2}. {stage.name:<18} {'done (skipped)' if done else 'will run'}")


def pack_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bundle finished pipeline outputs into one .tgz for group members."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", help="dataset to pack: yelpchi or amazon")
    parser.add_argument(
        "-o", "--output", type=Path, help="bundle to write (default: repo root)"
    )
    args = parser.parse_args(argv)
    try:
        paths = _paths(args.config, args.dataset)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"{BUNDLE_PREFIX}{paths.processed_dir.name}_{stamp}{BUNDLE_SUFFIX}"
    out = args.output or REPO_ROOT / name
    print("Hashing and packing outputs (this can take a few minutes)...", flush=True)
    try:
        manifest = pack(paths, out, _config_files(args.config, paths))
    except ArtifactError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    total = sum(e["size"] for e in manifest["files"])
    print(f"Packed {len(manifest['files'])} files ({_mb(total)}) into {out}")
    print(f"Bundle size: {_mb(out.stat().st_size)}")
    print("With this bundle unpacked, `uv run run.py` treats these stages as:")
    print_stage_status(paths)
    print(
        "Share it outside git (e.g. OneDrive or Teams): it contains row-level "
        "data and is too large for GitHub."
    )
    return 0


def find_bundle(directory: Path, dataset: str) -> Path:
    pattern = f"{BUNDLE_PREFIX}{dataset}_*{BUNDLE_SUFFIX}"
    found = sorted(directory.glob(pattern))
    if not found:
        raise ArtifactError(
            f"No {pattern} file in {directory}. Put the bundle next to "
            "unpack_artifacts.py, or pass its path."
        )
    if len(found) > 1:
        print(f"Found {len(found)} bundles; using the newest, {found[-1].name}")
    return found[-1]


def unpack_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unpack a bundle from pack_artifacts.py into this repository."
    )
    parser.add_argument(
        "bundle",
        type=Path,
        nargs="?",
        help=f"bundle to unpack (default: newest {BUNDLE_PREFIX}*{BUNDLE_SUFFIX} "
        "in the repo root)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset", help="dataset the bundle must hold: yelpchi or amazon"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing files that differ"
    )
    args = parser.parse_args(argv)
    try:
        paths = _paths(args.config, args.dataset)
        bundle = args.bundle or find_bundle(REPO_ROOT, paths.processed_dir.name)
        print(f"Unpacking {bundle} ...", flush=True)
        result = unpack(
            bundle,
            paths.root,
            _config_files(args.config, paths),
            args.force,
            dataset=paths.processed_dir.name,
        )
    except (ArtifactError, ConfigError, tarfile.TarError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for w in result.warnings:
        print(f"WARNING: {w}")
    print(
        f"Wrote {len(result.written)} files; {len(result.unchanged)} were already "
        "up to date. All checksums verified."
    )
    print("`uv run run.py` will now treat the stages as:")
    print_stage_status(paths)
    return 0
