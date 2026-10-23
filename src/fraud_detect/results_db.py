"""The group's results database: one committed folder per recorded run.

    results/runs/<run_id>/meta.json     who-ran-what: device, UTC times, short
                                        description, dataset, feature set,
                                        settings, git commit, headline numbers
    results/runs/<run_id>/results.csv   the run's results table (aggregate
                                        metrics only, never row-level data)

Each entry is its own folder with a unique ID, so teammates who record runs
at the same time never edit the same file and their commits never conflict.
There is no committed index: `scripts/results_db.py list` and `export` build
it from the folders. Entries are append-only; a corrected entry is a new one
that names the old ID in `supersedes`.

The results table may hold any aggregate numbers (one row per model, per
contrast, per fold or per setting), but no node-level data: `check_results_table`
rejects ID, label and prediction columns and tables longer than MAX_ROWS.
"""

from __future__ import annotations

import json
import platform
import re
import secrets
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fraud_detect.runlog import git_commit

SCHEMA_VERSION = 1
META = "meta.json"
RESULTS = "results.csv"

KINDS = ("pipeline", "probe", "compare", "check", "other")
STATUSES = ("ok", "failed", "partial")

# A results table is a summary, not data: no row-level columns and few rows.
# (Amazon's test set alone has 3,456 rows, so a prediction dump cannot pass.)
FORBIDDEN_COLUMNS = frozenset(
    {"node_id", "label", "group_id", "split", "target", "is_labelled", "y",
     "y_true", "y_pred", "prob", "proba", "probability", "probs_test",
     "preds_test"}
)  # fmt: skip
MAX_ROWS = 1000
MAX_CSV_BYTES = 256_000
MAX_DESCRIPTION = 200
MAX_DEVICE = 60

# Settings copied from a run log's config.yaml (dotted paths).
LOGGED_SETTINGS = (
    "seed",
    "feature_set",
    "split.repeat",
    "split.n_repeats",
    "df_analyze.classifiers",
    "df_analyze.htune_trials",
    "df_analyze.htune_cls_metric",
    "df_analyze.commit",
    "gnn_embed",
    "neighbour_agg",
)

REQUIRED_META = (
    "schema_version",
    "run_id",
    "recorded_utc",
    "started_utc",
    "device",
    "device_details",
    "description",
    "kind",
    "status",
)


class ResultsDBError(ValueError):
    """An entry breaks the database's rules; the message says how."""


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: str) -> str:
    """An ISO 8601 time with a time zone (2026-10-21T14:05:00Z or +02:00),
    as UTC 'YYYY-MM-DDTHH:MM:SSZ'. A time without a zone is refused, so a
    local time is never stored as UTC by mistake."""
    try:
        moment = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError as e:
        raise ResultsDBError(f"Not an ISO 8601 time: {text!r}") from e
    if moment.tzinfo is None:
        raise ResultsDBError(
            f"Time {text!r} has no time zone; give UTC with a trailing Z "
            "(e.g. 2026-10-21T14:05:00Z)"
        )
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def minutes_between(started_utc: str, finished_utc: str) -> float:
    start = datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
    end = datetime.fromisoformat(finished_utc.replace("Z", "+00:00"))
    if end < start:
        raise ResultsDBError(f"finished_utc {finished_utc} is before {started_utc}")
    return round((end - start).total_seconds() / 60, 1)


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True, timeout=10,
            )  # fmt: skip
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def _gpus() -> list[dict[str, str]]:
    """NVIDIA GPUs from nvidia-smi (none on Macs or machines without one)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=20,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    return gpus


def detect_device() -> dict[str, Any]:
    """This machine's OS, CPU and GPUs. No host or user names are recorded:
    the `device` label of an entry is chosen by whoever records it."""
    release = platform.release()
    return {
        "os": platform.system(),
        "platform": platform.platform(),
        "wsl": "microsoft" in release.lower(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "gpus": _gpus(),
    }


# --------------------------------------------------------------------------
# Reading what a run already saved
# --------------------------------------------------------------------------
def _dotted(config: dict[str, Any], path: str) -> Any:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def from_run_log(run_log: Path) -> dict[str, Any]:
    """Fields from one `run.py` invocation's log folder
    (outputs/<dataset>/runs/<stamp>/: run_info.json and config.yaml)."""
    info_path = run_log / "run_info.json"
    if not info_path.is_file():
        raise ResultsDBError(f"{run_log} has no run_info.json (not a run log folder)")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fields: dict[str, Any] = {
        "started_utc": parse_utc(info["started_utc"]),
        "command": " ".join(info.get("argv") or []),
        "git_commit": info.get("git_commit"),
        "packages": info.get("packages") or {},
        "stages": info.get("stages") or [],
    }
    config_path = run_log / "config.yaml"
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        fields["dataset"] = config.get("dataset")
        fields["feature_set"] = config.get("feature_set")
        fields["split_repeat"] = _dotted(config, "split.repeat")
        settings = {k: _dotted(config, k) for k in LOGGED_SETTINGS}
        blocks = (config.get("feature_sets") or {}).get(config.get("feature_set"))
        settings["blocks"] = blocks
        fields["settings"] = {k: v for k, v in settings.items() if v is not None}
    return fields


def from_pipeline_run(feature_set_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fields and the results table of a finished feature-set run
    (outputs/<dataset>[_repeat<k>]/<feature_set>/): reports/model_a.json and
    reports/model_a_metrics.csv, written by the df_analyze_report stage."""
    reports = feature_set_dir / "reports"
    summary_path, table_path = reports / "model_a.json", reports / "model_a_metrics.csv"
    for path in (summary_path, table_path):
        if not path.is_file():
            raise ResultsDBError(
                f"{path} is missing; run the pipeline through df_analyze_report first"
            )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    a = summary.get("model_a") or {}
    intervals = {i["metric"]: i for i in summary.get("intervals") or []}
    pr, auroc = intervals.get("pr_auc", {}), intervals.get("auroc", {})
    headline = {
        "model_a": f"{a.get('model')} ({a.get('selection')})",
        "test_pr_auc": a.get("pr_auc"),
        "test_pr_auc_low": pr.get("low"),
        "test_pr_auc_high": pr.get("high"),
        "test_auroc": a.get("auroc"),
        "test_auroc_low": auroc.get("low"),
        "test_auroc_high": auroc.get("high"),
        "no_skill_pr_auc": summary.get("test_positive_rate"),
        "cv_pr_auc": a.get("cv_pr_auc"),
        "cv_pr_auc_std": a.get("cv_pr_auc_std"),
        "f1": a.get("f1"),
        "threshold": a.get("threshold"),
        "n_train": summary.get("n_train"),
        "n_test": summary.get("n_test"),
        "sanity_warning": summary.get("sanity_warning") or None,
    }
    match = re.search(r"_repeat(\d+)$", feature_set_dir.parent.name)
    fields = {
        "dataset": summary.get("dataset"),
        "feature_set": summary.get("feature_set") or feature_set_dir.name,
        "split_repeat": int(match.group(1)) if match else 0,
        "seed": summary.get("seed"),
        "df_analyze_run": summary.get("run"),
        "headline": {k: v for k, v in headline.items() if v is not None},
        # When model_a.json was written: the end of the scored run.
        "finished_utc": datetime.fromtimestamp(
            summary_path.stat().st_mtime, UTC
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    settings = {"htune_trials": summary.get("htune_trials")}
    fields["settings"] = {k: v for k, v in settings.items() if v is not None}
    return fields, pd.read_csv(table_path)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_results_table(table: pd.DataFrame) -> None:
    """Refuse a table that could hold row-level data."""
    bad = sorted(c for c in table.columns if str(c).strip().lower() in FORBIDDEN_COLUMNS)
    if bad:
        raise ResultsDBError(
            f"Results table has row-level columns {bad}; record aggregate results "
            "only (AGENTS.md: never commit row-level data)"
        )
    if len(table) > MAX_ROWS:
        raise ResultsDBError(
            f"Results table has {len(table)} rows (limit {MAX_ROWS}); that looks like "
            "row-level data. Record a summary instead."
        )
    if table.columns.empty:
        raise ResultsDBError("Results table has no columns")


def check_meta(meta: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_META if meta.get(k) in (None, "")]
    if missing:
        raise ResultsDBError(f"Entry is missing {missing}")
    description = str(meta["description"])
    if "\n" in description or len(description) > MAX_DESCRIPTION:
        raise ResultsDBError(
            f"description must be one line of at most {MAX_DESCRIPTION} characters"
        )
    if len(str(meta["device"])) > MAX_DEVICE:
        raise ResultsDBError(f"device label must be at most {MAX_DEVICE} characters")
    if meta["kind"] not in KINDS:
        raise ResultsDBError(f"kind {meta['kind']!r} is not one of {KINDS}")
    if meta["status"] not in STATUSES:
        raise ResultsDBError(f"status {meta['status']!r} is not one of {STATUSES}")
    for key in ("recorded_utc", "started_utc", "finished_utc"):
        if meta.get(key) is not None and parse_utc(meta[key]) != meta[key]:
            raise ResultsDBError(f"{key} must be UTC as YYYY-MM-DDTHH:MM:SSZ")


# --------------------------------------------------------------------------
# Writing and reading entries
# --------------------------------------------------------------------------
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "run"


def make_run_id(started_utc: str, label: str) -> str:
    """<start, UTC>_<label>_<6 random hex>, e.g.
    20261021T140500Z_amazon-pilot-m1-own_3fa2c1: sorts by start time and
    never collides between teammates."""
    stamp = started_utc.replace("-", "").replace(":", "")
    return f"{stamp}_{_slug(label)}_{secrets.token_hex(3)}"


def build_meta(
    *,
    description: str,
    device: str,
    started_utc: str,
    kind: str = "pipeline",
    status: str = "ok",
    root: Path | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """A complete meta.json: the given fields, this machine's details, the
    git commit and the time it was recorded. `fields` may add any of dataset,
    feature_set, split_repeat, seed, finished_utc, duration_min, command,
    settings, headline, tags, notes, supersedes, source, packages."""
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "",
        "recorded_utc": utc_now(),
        "started_utc": parse_utc(started_utc),
        "finished_utc": None,
        "duration_min": None,
        "device": device.strip(),
        "device_details": detect_device(),
        "description": description.strip(),
        "kind": kind,
        "status": status,
        "dataset": None,
        "feature_set": None,
        "split_repeat": None,
        "seed": None,
        "git_commit": git_commit(root) if root else None,
        "git_dirty": _git_dirty(root) if root else None,
    }
    meta.update({k: v for k, v in fields.items() if v is not None})
    if meta.get("finished_utc"):
        meta["finished_utc"] = parse_utc(meta["finished_utc"])
        if meta.get("duration_min") is None:
            meta["duration_min"] = minutes_between(
                meta["started_utc"], meta["finished_utc"]
            )
    label = (
        "_".join(str(meta[k]) for k in ("dataset", "feature_set") if meta.get(k)) or kind
    )
    if meta.get("split_repeat"):
        label += f"-r{meta['split_repeat']}"
    meta["run_id"] = make_run_id(meta["started_utc"], label)
    return meta


def _git_dirty(root: Path) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, capture_output=True, text=True, check=True, timeout=30,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip())


def add_entry(db_dir: Path, meta: dict[str, Any], table: pd.DataFrame) -> Path:
    """Check and write one entry as db_dir/<run_id>/; return its folder. The
    folder is written under a temporary name and renamed, so a crash never
    leaves half an entry."""
    check_meta(meta)
    check_results_table(table)
    meta = {**meta, "results_rows": len(table), "results_columns": list(table.columns)}
    final = db_dir / meta["run_id"]
    if final.exists():
        raise ResultsDBError(f"{final} already exists")
    tmp = db_dir / f".{meta['run_id']}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        table.to_csv(tmp / RESULTS, index=False)
        if (tmp / RESULTS).stat().st_size > MAX_CSV_BYTES:
            raise ResultsDBError(
                f"Results table is over {MAX_CSV_BYTES // 1000} kB; record a summary"
            )
        (tmp / META).write_text(
            json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
        )
        tmp.rename(final)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


def entry_dirs(db_dir: Path) -> list[Path]:
    if not db_dir.is_dir():
        return []
    return sorted(
        d for d in db_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def load_meta(entry: Path) -> dict[str, Any]:
    return json.loads((entry / META).read_text(encoding="utf-8"))


def check_db(db_dir: Path) -> list[str]:
    """Every problem in the database, one message each (empty: all good)."""
    problems = []
    for entry in entry_dirs(db_dir):
        try:
            if not (entry / META).is_file() or not (entry / RESULTS).is_file():
                raise ResultsDBError(f"needs both {META} and {RESULTS}")
            meta = load_meta(entry)
            check_meta(meta)
            if meta["run_id"] != entry.name:
                raise ResultsDBError(f"run_id {meta['run_id']!r} differs from its folder")
            if (entry / RESULTS).stat().st_size > MAX_CSV_BYTES:
                raise ResultsDBError(f"{RESULTS} is over {MAX_CSV_BYTES // 1000} kB")
            check_results_table(pd.read_csv(entry / RESULTS))
            extra = sorted(
                p.name for p in entry.iterdir() if p.name not in (META, RESULTS)
            )
            if extra:
                raise ResultsDBError(f"unexpected files {extra}")
        except (ResultsDBError, ValueError, KeyError) as e:
            problems.append(f"{entry.name}: {e}")
    return problems


INDEX_COLUMNS = (
    "run_id", "started_utc", "duration_min", "device", "dataset", "feature_set",
    "split_repeat", "kind", "status", "description",
)  # fmt: skip


def index_table(db_dir: Path) -> pd.DataFrame:
    """One row per entry: the main fields, then the flattened headline
    numbers (headline.test_pr_auc, ...), oldest first."""
    rows = []
    for entry in entry_dirs(db_dir):
        meta = load_meta(entry)
        row = {k: meta.get(k) for k in INDEX_COLUMNS}
        row["gpu"] = ", ".join(g["name"] for g in meta["device_details"].get("gpus", []))
        row["tags"] = ", ".join(meta.get("tags") or [])
        row["git_commit"] = (meta.get("git_commit") or "")[:9]
        row.update({f"headline.{k}": v for k, v in (meta.get("headline") or {}).items()})
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[*INDEX_COLUMNS, "gpu", "tags", "git_commit"])
    return pd.DataFrame(rows).sort_values(["started_utc", "run_id"], kind="stable")


def combined_results(db_dir: Path) -> pd.DataFrame:
    """Every entry's results table stacked, with run_id, device, dataset and
    feature_set in front (columns an entry lacks are empty)."""
    frames = []
    for entry in entry_dirs(db_dir):
        meta = load_meta(entry)
        table = pd.read_csv(entry / RESULTS)
        front = {k: meta.get(k) for k in ("run_id", "device", "dataset", "feature_set")}
        frames.append(pd.concat([pd.DataFrame([front] * len(table)), table], axis=1))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
