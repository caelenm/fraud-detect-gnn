"""Loading raw CSV exports (synthetic CSVs written to pytest's tmp_path)."""

from __future__ import annotations

import pandas as pd
import pytest
from synthetic import raw_csv_rows

from fraud_detect import columns as C
from fraud_detect.data.load import (
    RawDataError,
    combine_archives,
    find_raw_files,
    map_raw_header,
    read_archive_csv,
)

DATE_MIN, DATE_MAX = pd.Timestamp("2018-05-01"), pd.Timestamp("2023-08-31")


def write_csv(tmp_path, rows, name="complaints.csv"):
    path = tmp_path / "ARCHIVE" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_reads_narratives_in_range(tmp_path):
    path = write_csv(tmp_path, raw_csv_rows())
    df, summary = read_archive_csv(path, "ARCHIVE", DATE_MIN, DATE_MAX, chunksize=2)
    assert df[C.COMPLAINT_ID].tolist() == [1, 4]
    assert summary.n_rows == 4
    assert summary.n_with_narrative == 3
    assert summary.n_outside_date_range == 1
    assert df.loc[df[C.COMPLAINT_ID] == 4, C.NARRATIVE].item() == "another"
    assert df[C.STATE].tolist() == ["ZZ", "ZZ"]
    assert list(df.columns) == list(C.RAW_COLUMN_ALIASES)
    assert "zip_code" not in df.columns


def test_missing_narrative_column_stops(tmp_path):
    rows = [{k: v for k, v in r.items() if k != "Consumer complaint narrative"}
            for r in raw_csv_rows()]  # fmt: skip
    path = write_csv(tmp_path, rows)
    with pytest.raises(RawDataError, match="narrative"):
        read_archive_csv(path, "ARCHIVE", DATE_MIN, DATE_MAX, chunksize=10)


def test_empty_narrative_column_stops(tmp_path):
    rows = [r | {"Consumer complaint narrative": ""} for r in raw_csv_rows()]
    path = write_csv(tmp_path, rows)
    with pytest.raises(RawDataError, match="empty in every row"):
        read_archive_csv(path, "ARCHIVE", DATE_MIN, DATE_MAX, chunksize=10)


def test_us_style_dates_are_parsed(tmp_path):
    rows = [r | {"Date received": "06/01/2019"} for r in raw_csv_rows()]
    df, _ = read_archive_csv(
        write_csv(tmp_path, rows), "ARCHIVE", DATE_MIN, DATE_MAX, chunksize=10
    )
    assert (df[C.DATE_RECEIVED] == pd.Timestamp("2019-06-01")).all()


def test_api_style_headers_are_accepted():
    header = ["complaint_what_happened", *[a[1] for k, a in
              C.RAW_COLUMN_ALIASES.items() if k != C.NARRATIVE]]  # fmt: skip
    mapping = map_raw_header(header, "test")
    assert mapping["complaint_what_happened"] == C.NARRATIVE


def test_conflicting_duplicate_ids_stop():
    a = pd.DataFrame({C.COMPLAINT_ID: [1], C.NARRATIVE: ["x"]})
    b = pd.DataFrame({C.COMPLAINT_ID: [1], C.NARRATIVE: ["y"]})
    with pytest.raises(RawDataError, match="share a Complaint ID"):
        combine_archives([a, b])
    df, n = combine_archives([a, a.copy()])
    assert len(df) == 1 and n == 1


def test_missing_archive_directory_stops(tmp_path):
    with pytest.raises(RawDataError, match="download_dataset.sh"):
        find_raw_files(tmp_path, ["NOT_DOWNLOADED"])
