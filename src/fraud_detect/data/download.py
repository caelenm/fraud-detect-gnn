"""Checksum-verified download of the CARE-GNN dataset files.

Each dataset is one zip (YelpChi.zip, Amazon.zip) in the CARE-GNN repository
at a pinned commit, holding one .mat file. Both the zip and the extracted
.mat are checked against SHA-256 values in the config (docs/RESEARCH_PLAN.md
§2), so everyone works with byte-identical data.

Rules:
- Safe to rerun: a file that is present with the right checksum is kept.
- A file present with a different checksum is never overwritten unless
  `force` is set (it may be someone's deliberate copy; stop and look).
- Downloads and extraction go to a temporary file first and are renamed only
  after the checksum matches, so a failed or interrupted run leaves nothing
  that could be mistaken for good data.

Only the Python standard library is used, so this behaves the same on Linux,
macOS and WSL.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """A file could not be fetched or failed its checksum; the message says why."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _check_existing(path: Path, expected: str, force: bool) -> bool:
    """True if `path` already holds the expected file (nothing to do)."""
    if not path.exists():
        return False
    actual = sha256_file(path)
    if actual == expected:
        return True
    if not force:
        raise DownloadError(
            f"{path} exists but its SHA-256 is {actual}, not the expected "
            f"{expected}. It was not overwritten. Inspect or delete it, or rerun "
            "with --force to replace it."
        )
    return False


def _copy_verified(source: Any, dest: Path, expected: str, what: str) -> None:
    """Stream `source` (a binary file object) to `dest` via a temporary file,
    keeping it only if its SHA-256 matches."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    try:
        with tmp.open("wb") as out:
            for block in iter(lambda: source.read(CHUNK), b""):
                h.update(block)
                out.write(block)
        if h.hexdigest() != expected:
            raise DownloadError(
                f"{what}: SHA-256 {h.hexdigest()} does not match the expected "
                f"{expected}. Nothing was saved."
            )
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def fetch(url: str, dest: Path, expected: str, force: bool = False) -> bool:
    """Download `url` to `dest` and verify it. Returns False if `dest` already
    held the expected file (nothing downloaded)."""
    if _check_existing(dest, expected, force):
        return False
    print(f"Downloading {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            _copy_verified(response, dest, expected, url)
    except OSError as e:  # URLError and network failures
        raise DownloadError(f"Could not download {url}: {e}") from e
    return True


def extract(zip_path: Path, member: str, dest: Path, expected: str,
            force: bool = False) -> bool:  # fmt: skip
    """Extract the zip member whose file name is `member` to `dest` and verify
    it. Only that member is read, never a path chosen by the archive. Returns
    False if `dest` already held the expected file."""
    if _check_existing(dest, expected, force):
        return False
    with zipfile.ZipFile(zip_path) as zf:
        matches = [
            info
            for info in zf.infolist()
            if not info.is_dir() and PurePosixPath(info.filename).name == member
        ]
        if len(matches) != 1:
            raise DownloadError(
                f"Expected exactly one {member} in {zip_path}, found "
                f"{[m.filename for m in matches]}"
            )
        print(f"Extracting {member}", flush=True)
        with zf.open(matches[0]) as source:
            _copy_verified(source, dest, expected, f"{zip_path.name}:{member}")
    return True


@dataclass(frozen=True)
class DatasetFiles:
    archive: Path
    mat: Path


def download_dataset(
    raw_dir: Path, base_url: str, data: dict[str, Any], force: bool = False
) -> DatasetFiles:
    """Fetch and verify one dataset's zip, then extract and verify its .mat."""
    archive = raw_dir / data["archive"]
    mat = raw_dir / data["mat"]
    url = f"{base_url.rstrip('/')}/{data['archive']}"
    if fetch(url, archive, data["archive_sha256"], force):
        print(f"Verified {archive.name} (SHA-256 {data['archive_sha256'][:12]}…)")
    else:
        print(f"Already present and verified: {archive}")
    if extract(archive, data["mat"], mat, data["mat_sha256"], force):
        print(f"Verified {mat.name} (SHA-256 {data['mat_sha256'][:12]}…)")
    else:
        print(f"Already present and verified: {mat}")
    return DatasetFiles(archive=archive, mat=mat)
