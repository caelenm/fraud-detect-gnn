"""Near-duplicate narrative removal with MinHash locality-sensitive hashing.

Narratives are normalised (lower-cased, CFPB redaction marks collapsed,
punctuation removed) and represented as sets of word shingles. Pairs whose
estimated Jaccard similarity is at least the threshold are linked, and each
connected group of linked narratives keeps only its earliest complaint.
Exact duplicates (after normalisation) are always grouped together.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

from fraud_detect import columns as C

_REDACTION = re.compile(r"x{2,}")  # CFPB masks names/dates/amounts as XXXX
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    text = _REDACTION.sub(" x ", text.lower())
    return " ".join(_NON_WORD.sub(" ", text).split())


def shingles(text: str, k: int) -> set[str]:
    words = text.split()
    if len(words) <= k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, self.parent[i]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass(frozen=True)
class DedupResult:
    kept: pd.DataFrame
    n_input: int
    n_removed: int
    n_groups_with_duplicates: int
    n_groups_with_mixed_labels: int


def near_duplicate_groups(
    texts: list[str], shingle_words: int, threshold: float, num_perm: int, seed: int
) -> np.ndarray:
    """Return a group id per text; texts in the same group are near-duplicates."""
    n = len(texts)
    uf = _UnionFind(n)
    normalized = [normalize_text(t) for t in texts]

    first_by_hash: dict[str, int] = {}
    for i, t in enumerate(normalized):
        h = hashlib.sha1(t.encode("utf-8")).hexdigest()
        if h in first_by_hash:
            uf.union(first_by_hash[h], i)
        else:
            first_by_hash[h] = i

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures: list[MinHash] = []
    for i, t in enumerate(normalized):
        m = MinHash(num_perm=num_perm, seed=seed)
        m.update_batch([s.encode("utf-8") for s in shingles(t, shingle_words)])
        signatures.append(m)
        lsh.insert(str(i), m)
    for i, m in enumerate(signatures):
        for key in lsh.query(m):
            j = int(key)
            if j != i and m.jaccard(signatures[j]) >= threshold:
                uf.union(i, j)
    return np.array([uf.find(i) for i in range(n)])


def remove_near_duplicates(
    df: pd.DataFrame, shingle_words: int, threshold: float, num_perm: int, seed: int
) -> DedupResult:
    """Keep the earliest complaint (by date received, then Complaint ID) in each
    near-duplicate group."""
    ordered = df.sort_values([C.DATE_RECEIVED, C.COMPLAINT_ID], ignore_index=True)
    groups = near_duplicate_groups(
        ordered[C.NARRATIVE].tolist(), shingle_words, threshold, num_perm, seed
    )
    ordered = ordered.assign(_group=groups)
    sizes = ordered["_group"].value_counts()
    dup_groups = sizes[sizes > 1].index
    mixed = 0
    if C.LABEL in ordered.columns and len(dup_groups):
        n_labels = (
            ordered[ordered["_group"].isin(dup_groups)]
            .groupby("_group")[C.LABEL]
            .nunique()
        )
        mixed = int((n_labels > 1).sum())
    kept = ordered.drop_duplicates("_group", keep="first").drop(columns="_group")
    return DedupResult(
        kept=kept.sort_values(C.COMPLAINT_ID, ignore_index=True),
        n_input=len(df),
        n_removed=len(df) - len(kept),
        n_groups_with_duplicates=len(dup_groups),
        n_groups_with_mixed_labels=mixed,
    )
