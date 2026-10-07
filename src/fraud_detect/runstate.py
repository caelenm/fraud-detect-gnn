"""Stopping and resuming pipeline runs.

Two levels of saving make a run safe to stop at any time (Ctrl+C, a crash, or
shutting the computer down):

1. **Stages.** When `run.py` starts running stages, it writes the list of
   stages still to run to `outputs/pipeline_state.json` and removes each one
   as it completes. `uv run run.py --resume` runs exactly the remaining
   stages. Without this, a later stage's *old* output (from a previous run)
   could be mistaken for a finished one after an interruption.

2. **Units inside a stage.** A long stage made of independent pieces (for
   example one fit per model and seed) saves each piece as it finishes with a
   `UnitStore`, obtained from `Context.unit_store()`. On a fresh run the store
   is emptied first; on `--resume` finished units are kept and skipped. Only
   the unit in progress when the run stopped is redone.

New stages get level 1 automatically. A stage longer than a few minutes that
can be split into units should use level 2 (see AGENTS.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

STATE_FILE = "pipeline_state.json"
CHECKPOINT_DIR = "checkpoints"


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file so a crash never leaves a half-written file."""
    atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    """Call `write(tmp)` to write a temporary file next to `path`, then rename it
    to `path`, so a crash never leaves a half-written file at `path`. The
    temporary name keeps the suffix (some writers, like np.savez, add one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, default=str) + "\n")


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    atomic_write(path, lambda tmp: frame.to_parquet(tmp, index=False))


def config_fingerprint(config: dict[str, Any]) -> str:
    """Hash of the effective config, so a resume can refuse a changed config."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class RunState:
    pending: list[str]
    config_sha256: str
    started_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)

    @staticmethod
    def path(outputs_dir: Path) -> Path:
        return outputs_dir / STATE_FILE

    @classmethod
    def load(cls, outputs_dir: Path) -> RunState | None:
        path = cls.path(outputs_dir)
        if not path.is_file():
            return None
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, outputs_dir: Path) -> None:
        atomic_write_text(self.path(outputs_dir), json.dumps(asdict(self), indent=2))

    def mark_done(self, stage: str, outputs_dir: Path) -> None:
        """Record a finished stage; delete the state file once none remain."""
        self.pending.remove(stage)
        self.completed.append(stage)
        if self.pending:
            self.save(outputs_dir)
        else:
            self.path(outputs_dir).unlink(missing_ok=True)


class UnitStore:
    """Finished units of one stage, one small JSON file each.

    A unit is anything independently repeatable inside a stage, identified by
    a string such as "gnn/seed-3". `save` writes atomically, so a unit is
    either fully recorded or absent.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _file(self, unit: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in unit)
        digest = hashlib.sha256(unit.encode("utf-8")).hexdigest()[:8]
        return self.directory / f"{safe}-{digest}.json"

    def done(self, unit: str) -> bool:
        return self._file(unit).is_file()

    def save(self, unit: str, result: dict[str, Any]) -> None:
        payload = {"unit": unit, "saved_utc": datetime.now(UTC).isoformat(), **result}
        atomic_write_text(self._file(unit), json.dumps(payload, indent=2, default=str))

    def load(self, unit: str) -> dict[str, Any]:
        return json.loads(self._file(unit).read_text(encoding="utf-8"))

    def all(self) -> list[dict[str, Any]]:
        if not self.directory.is_dir():
            return []
        return [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(self.directory.glob("*.json"))
        ]

    def reset(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)
