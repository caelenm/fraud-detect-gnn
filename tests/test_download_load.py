"""Download and load stages on fake CARE-GNN files (tests/synthetic.py).

Fake .zip/.mat files are written to pytest's tmp_path and "downloaded" from
there through a file:// URL. No real data is used or fetched.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from helpers import fake_dataset, load_edges, sha256

from fraud_detect import columns as C
from fraud_detect.cli import main
from fraud_detect.config import get_paths
from fraud_detect.data import care_gnn, download


# ---- Edge lists ----------------------------------------------------------
def test_edge_list_is_undirected_deduplicated_and_loop_free():
    rows = [0, 1, 2, 3, 1, 4]
    cols = [1, 0, 2, 1, 0, 0]
    data = [1, 1, 1, 1, 1, 0]  # (1, 0) stored twice; (4, 0) an explicit zero
    adj = sp.coo_matrix((data, (rows, cols)), shape=(5, 5))
    edges = care_gnn.edge_list(adj)
    assert edges.dtype == np.int32 and edges.shape == (2, 2)
    # (0,1) once although stored as (0,1), (1,0) and (1,0) again; the self-loop
    # (2,2) and the zero (4,0) are dropped; the one-directional (3,1) is kept.
    assert edges.T.tolist() == [[0, 1], [1, 3]]
    assert (edges[0] < edges[1]).all()


def test_groups_by_components_and_by_duplicate_rows():
    edges = np.array([[0, 3], [1, 4]], dtype=np.int32)  # edges 0-1 and 3-4
    assert care_gnn.component_groups(edges, 6).tolist() == [0, 0, 1, 2, 2, 3]
    features = np.array([[1.0, 2], [0, 0], [1, 2], [0, 0], [5, 5]])
    assert care_gnn.duplicate_groups(features).tolist() == [0, 1, 0, 1, 2]


# ---- Node table -----------------------------------------------------------
def test_unlabelled_prefix_is_never_a_labelled_example():
    labels = np.array([0, 0, 0, 1, 0, 1])
    nodes = care_gnn.node_table(labels, unlabelled_prefix=3, groups=np.arange(6))
    assert nodes[C.IS_LABELLED].tolist() == [False] * 3 + [True] * 3
    assert nodes[C.LABEL].isna().tolist() == [True] * 3 + [False] * 3
    assert nodes.loc[nodes[C.IS_LABELLED], C.LABEL].tolist() == [1, 0, 1]
    assert (nodes.loc[:2, C.SPLIT] == C.UNLABELLED).all()
    # The stored 0s of the prefix must not become negatives.
    assert (nodes[C.LABEL] == 0).sum() == 1
    with pytest.raises(care_gnn.DatasetError, match="unlabelled"):
        care_gnn.node_table(np.array([1, 0, 1]), unlabelled_prefix=1, groups=np.arange(3))


# ---- Download -------------------------------------------------------------
def test_fetch_rejects_a_bad_checksum_and_keeps_nothing(tmp_path):
    source = tmp_path / "server" / "x.bin"
    source.parent.mkdir()
    source.write_bytes(b"placeholder bytes")
    dest = tmp_path / "raw" / "x.bin"
    with pytest.raises(download.DownloadError, match="does not match"):
        download.fetch(source.as_uri(), dest, "0" * 64)
    assert not dest.exists() and not list(dest.parent.glob("*"))

    assert download.fetch(source.as_uri(), dest, sha256(source)) is True
    assert download.fetch(source.as_uri(), dest, sha256(source)) is False  # rerun: kept


def test_existing_file_with_another_checksum_needs_force(tmp_path):
    source = tmp_path / "x.bin"
    source.write_bytes(b"placeholder bytes")
    dest = tmp_path / "raw" / "x.bin"
    dest.parent.mkdir()
    dest.write_bytes(b"someone else's file")
    with pytest.raises(download.DownloadError, match="not overwritten"):
        download.fetch(source.as_uri(), dest, sha256(source))
    assert dest.read_bytes() == b"someone else's file"
    download.fetch(source.as_uri(), dest, sha256(source), force=True)
    assert dest.read_bytes() == b"placeholder bytes"


def test_extract_reads_only_the_named_member(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data/Fake.mat", b"placeholder")
        zf.writestr("../evil.mat", b"x")
    out = tmp_path / "raw" / "Fake.mat"
    download.extract(archive, "Fake.mat", out, sha256_bytes(b"placeholder"))
    assert out.read_bytes() == b"placeholder"
    assert not (tmp_path / "evil.mat").exists()
    with pytest.raises(download.DownloadError, match="exactly one"):
        download.extract(archive, "Other.mat", tmp_path / "o.mat", "0" * 64)


def sha256_bytes(data: bytes) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(data).hexdigest()


# ---- The stages end to end ------------------------------------------------
def test_download_and_load_stages(tmp_path):
    path, config = fake_dataset(tmp_path)
    assert main(["--config", str(path), "--to", "load"]) == 0
    p = get_paths(config)
    assert sha256(p.mat) == config["datasets"]["fake"]["mat_sha256"]

    nodes = pd.read_parquet(p.nodes)
    expected = config["datasets"]["fake"]["expected"]
    assert len(nodes) == expected["n_nodes"]
    assert int(nodes[C.IS_LABELLED].sum()) == expected["n_labelled"]
    assert nodes[C.NODE_ID].tolist() == list(range(len(nodes)))
    # The planted duplicate rows (nodes 8 and 9) share a group; others do not.
    assert nodes.loc[8, C.GROUP_ID] == nodes.loc[9, C.GROUP_ID]
    assert nodes[C.GROUP_ID].nunique() == len(nodes) - 1

    for name in ("aaa", "bbb", "homo"):
        edges = load_edges(p.relation_file(name))
        assert edges.shape[1] == expected["n_edges"][name]
        assert (edges[0] < edges[1]).all() and edges.dtype == np.int32
    assert load_edges(p.relation_file("aaa")).T.tolist()[:2] == [[0, 1], [2, 3]]


def test_a_count_mismatch_fails_loudly_and_writes_nothing(tmp_path, capsys):
    path, config = fake_dataset(tmp_path)
    text = path.read_text(encoding="utf-8")
    n = config["datasets"]["fake"]["expected"]["n_positive"]
    path.write_text(text.replace(f"n_positive: {n}", f"n_positive: {n + 1}"))
    assert main(["--config", str(path), "--to", "load"]) == 2
    err = capsys.readouterr().err
    assert f"n_positive: expected {n + 1}, found {n}" in err
    assert not get_paths(config).nodes.exists()
