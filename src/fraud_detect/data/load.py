"""Stage 1 helpers: read the CFPB archive files and keep complaints with narratives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraud_detect import columns as C

# Date formats seen in CFPB exports. The first format that parses every
# non-missing value in a chunk is used; anything else is an error.
DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y")


class RawDataError(RuntimeError):
    """Raised when raw files are missing or do not look like CFPB exports."""


@dataclass(frozen=True)
class RawFileSummary:
    archive: str
    file: str
    n_rows: int
    n_with_narrative: int
    n_outside_date_range: int


def find_raw_files(raw_dir: Path, archive_names: list[str]) -> dict[str, list[Path]]:
    """Return the CSV files for each archive, failing loudly if any are missing."""
    found: dict[str, list[Path]] = {}
    for name in archive_names:
        archive_dir = raw_dir / name
        if not archive_dir.is_dir():
            raise RawDataError(
                f"Missing extracted archive directory: {archive_dir}\n"
                "Run ./download_dataset.sh first."
            )
        csvs = sorted(archive_dir.rglob("*.csv"))
        if not csvs:
            listing = sorted(
                str(p.relative_to(archive_dir)) for p in archive_dir.rglob("*")
            )
            raise RawDataError(
                f"No .csv files in {archive_dir}. Found: {listing[:20]}\n"
                "The loader expects CSV exports. Stop and report the file format."
            )
        found[name] = csvs
    return found


def map_raw_header(header: list[str], source: str) -> dict[str, str]:
    """Map raw header names to canonical names. Raises if a required column is
    missing, including the complaint narrative column."""
    raw_to_canonical: dict[str, str] = {}
    missing = []
    for canonical, aliases in C.RAW_COLUMN_ALIASES.items():
        matches = [a for a in aliases if a in header]
        if len(matches) > 1:
            raise RawDataError(f"{source}: ambiguous columns {matches} for {canonical}")
        if not matches:
            missing.append(canonical)
            continue
        raw_to_canonical[matches[0]] = canonical
    if missing:
        raise RawDataError(
            f"{source}: missing required columns {missing}.\n"
            f"Header found: {header}\n"
            "Stop and report this file rather than working around it."
        )
    return raw_to_canonical


def parse_dates(values: pd.Series, source: str) -> pd.Series:
    """Parse a date column with one of the known formats, all-or-nothing."""
    present = values.notna()
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        if parsed[present].notna().all():
            return parsed
    bad = values[present].head(5).tolist()
    raise RawDataError(f"{source}: could not parse dates such as {bad}")


def clean_chunk(chunk: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalise one chunk of canonical columns: trim strings, parse types."""
    chunk = chunk.copy()
    for col in chunk.columns:
        chunk[col] = chunk[col].str.strip()
        chunk.loc[chunk[col] == "", col] = pd.NA

    ids = pd.to_numeric(chunk[C.COMPLAINT_ID], errors="coerce")
    if ids.isna().any():
        bad = chunk.loc[ids.isna(), C.COMPLAINT_ID].head(5).tolist()
        raise RawDataError(f"{source}: non-numeric Complaint IDs such as {bad}")
    chunk[C.COMPLAINT_ID] = ids.astype("int64")
    chunk[C.DATE_RECEIVED] = parse_dates(chunk[C.DATE_RECEIVED], source)
    if chunk[C.DATE_RECEIVED].isna().any():
        raise RawDataError(f"{source}: rows with a missing Date received")
    chunk[C.STATE] = chunk[C.STATE].str.upper()
    return chunk


def read_archive_csv(
    path: Path,
    archive: str,
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
    chunksize: int,
) -> tuple[pd.DataFrame, RawFileSummary]:
    """Read one CSV in chunks, keeping rows with a non-empty narrative that were
    received inside [date_min, date_max]."""
    source = f"{archive}/{path.name}"
    header = pd.read_csv(path, nrows=0, dtype=str).columns.tolist()
    rename = map_raw_header(header, source)

    kept = []
    n_rows = n_narr = n_outside = 0
    reader = pd.read_csv(
        path,
        usecols=list(rename),
        dtype=str,
        keep_default_na=False,  # only empty strings count as missing
        na_values=[""],
        chunksize=chunksize,
    )
    for raw in reader:
        chunk = clean_chunk(raw.rename(columns=rename), source)
        n_rows += len(chunk)
        chunk = chunk[chunk[C.NARRATIVE].notna()]
        n_narr += len(chunk)
        in_range = chunk[C.DATE_RECEIVED].between(date_min, date_max)
        n_outside += int((~in_range).sum())
        kept.append(chunk[in_range])

    if n_narr == 0:
        raise RawDataError(
            f"{source}: the complaint narrative column is empty in every row. "
            "Stop and report this file."
        )
    summary = RawFileSummary(archive, path.name, n_rows, n_narr, n_outside)
    ordered = list(C.RAW_COLUMN_ALIASES)
    return pd.concat(kept, ignore_index=True)[ordered], summary


def combine_archives(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Concatenate archive frames and remove exact duplicate rows.

    Returns the combined frame and the number of exact duplicates dropped.
    Raises if the same Complaint ID appears with different content.
    """
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates()
    n_exact = before - len(df)
    dup_ids = df[C.COMPLAINT_ID].duplicated(keep=False)
    if dup_ids.any():
        ids = df.loc[dup_ids, C.COMPLAINT_ID].unique()[:10].tolist()
        raise RawDataError(
            f"{int(dup_ids.sum())} rows share a Complaint ID but differ in content "
            f"(e.g. IDs {ids}). Stop and report this."
        )
    return df.sort_values(C.COMPLAINT_ID, ignore_index=True), n_exact
