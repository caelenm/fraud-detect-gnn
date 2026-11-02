"""Every model setup we ran, defined once.

A feature-set name in configs/default.yaml or results/runs/ says what was run:
`pilot_m5_own_gnnssl_no19d16` is setup E2, at the pilot budget, without f19,
with embeddings of size 16. The dashboard, scripts/gnn_report.py and
scripts/rq_report.py read those names through this module, so a setup cannot
have two names or two descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass

TABULAR = "tabular"
GRAPH_FEATURES = "tabular + graph features"
GNN = "GNN"
NULL_GRAPH = "null-graph control"
FAMILIES = (TABULAR, GRAPH_FEATURES, GNN, NULL_GRAPH)

PILOT = "pilot"
FULL = "full"
BUDGETS = (PILOT, FULL)
PILOT_PREFIX = "pilot_"
BUDGET_LABELS = {
    PILOT: "pilot (3 classifiers, 10 trials)",
    FULL: "full (6 classifiers, 100 trials)",
}

ALL = "all"
NO36 = "no36"
NO19 = "no19"
TOP4 = "top4"
TOP4NO19 = "top4no19"
ONLY36 = "only36"
VARIANTS = (ALL, NO19, NO36, TOP4NO19, TOP4, ONLY36)
VARIANT_LABELS = {
    ALL: "all features",
    NO19: "without f19",
    NO36: "without #36 (f13, f15, f18, f19)",
    TOP4NO19: "top 4 without f19",
    TOP4: "top 4 (with f19)",
    ONLY36: "the #36 four only",
}
DEFAULT_EMBED_SIZE = "8"
# Feature-set suffix -> (variant, embedding size; None: the default size).
SUFFIXES: dict[str, tuple[str, str | None]] = {
    "": (ALL, None),
    "no36": (NO36, None),
    "no19": (NO19, None),
    "no19d2": (NO19, "2"),
    "no19d4": (NO19, "4"),
    "no19d16": (NO19, "16"),
    "no19raw": (NO19, "raw 64"),
    "only36": (ONLY36, None),
    "top4": (TOP4, None),
    "top4no19": (TOP4NO19, None),
}


class UnknownFeatureSetError(ValueError):
    """A feature-set name this registry cannot read."""


@dataclass(frozen=True)
class Setup:
    base: str  # the feature-set name without budget prefix or variant suffix
    setup_id: str
    family: str
    label: str
    uses_graph: bool = True
    gnn_rung: bool = False  # scored by the gnn_classify stage, not df-analyze
    label_derived_column: bool = False  # invariant 1's exception (Model 6)
    embeddings: bool = False  # has an embedding size


SETUPS = (
    Setup("m1_own", "B", TABULAR, "df-analyze on own features", uses_graph=False),
    Setup("m2_own_nbr", "E1", GRAPH_FEATURES, "+ neighbour means"),
    Setup("m5_own_gnnssl", "E2", GRAPH_FEATURES, "+ DGI embeddings", embeddings=True),
    Setup("m5r_own_gnnrand", "E3", GRAPH_FEATURES, "+ untrained GNN embeddings",
          embeddings=True),
    Setup("m7_own_nbr_gnnssl", "E7", GRAPH_FEATURES,
          "+ neighbour means + DGI embeddings"),
    Setup("m7r_own_nbr_gnnrand", "E7r", GRAPH_FEATURES,
          "+ neighbour means + untrained GNN embeddings"),
    Setup("m8_own_nbr_gnnsup", "E8", GRAPH_FEATURES,
          "+ neighbour means + best GNN's out-of-fold score",
          label_derived_column=True),
    Setup("m8s_own_gnnsup", "E8s", GRAPH_FEATURES, "+ best GNN's out-of-fold score",
          label_derived_column=True),
    Setup("g5_sage", "E5", GNN, "GNN: heterogeneous GraphSAGE", gnn_rung=True),
    Setup("g5g_rgcn", "E5g", GNN, "GNN: R-GCN-style normalisation", gnn_rung=True),
    Setup("g6_mlp", "E6", GNN, "GNN without edges (MLP)", uses_graph=False,
          gnn_rung=True),
    Setup("g5n3_sage_rw", "N3", NULL_GRAPH, "GNN: GraphSAGE on the rewired graph",
          gnn_rung=True),
    Setup("n1_own_deg", "N1", NULL_GRAPH, "+ degree only"),
    Setup("n2_own_nbrrw", "N2", NULL_GRAPH, "+ neighbour means on a rewired graph"),
)  # fmt: skip
BY_ID = {s.setup_id: s for s in SETUPS}
_ORDER = {s.setup_id: i for i, s in enumerate(SETUPS)}
_SIZE_ORDER = {"2": 0, "4": 1, DEFAULT_EMBED_SIZE: 2, "16": 3, "raw 64": 4}
_LONGEST_FIRST = sorted(SETUPS, key=lambda s: len(s.base), reverse=True)


@dataclass(frozen=True)
class FeatureSet:
    """What one feature-set name means."""

    name: str
    setup: Setup
    variant: str
    embed_size: str | None  # None for setups without embeddings
    budget: str

    @property
    def display_id(self) -> str:
        """E2, or E2 (16) for a size other than the default."""
        if self.embed_size in (None, DEFAULT_EMBED_SIZE):
            return self.setup.setup_id
        return f"{self.setup.setup_id} ({self.embed_size})"

    @property
    def sort_key(self) -> tuple[int, int]:
        return _ORDER[self.setup.setup_id], _SIZE_ORDER.get(self.embed_size or "", 0)


def parse_feature_set(name: str) -> FeatureSet:
    """Read a feature-set name; UnknownFeatureSetError if it means nothing here."""
    budget = PILOT if name.startswith(PILOT_PREFIX) else FULL
    stem = name.removeprefix(PILOT_PREFIX)
    for setup in _LONGEST_FIRST:
        if stem != setup.base and not stem.startswith(setup.base + "_"):
            continue
        suffix = stem[len(setup.base) + 1 :]
        if suffix not in SUFFIXES:
            raise UnknownFeatureSetError(f"{name!r}: unknown variant {suffix!r}")
        variant, size = SUFFIXES[suffix]
        if size is not None and not setup.embeddings:
            raise UnknownFeatureSetError(f"{name!r}: {setup.setup_id} has no embeddings")
        if setup.embeddings and size is None:
            size = DEFAULT_EMBED_SIZE
        return FeatureSet(name, setup, variant, size, budget)
    raise UnknownFeatureSetError(f"{name!r} matches no setup")


def variant_label(variant: str, dataset: str) -> str:
    """How many features 'all' means differs per dataset."""
    if variant == ALL:
        return {"amazon": "all 25 features", "yelpchi": "all 32 features"}.get(
            dataset, VARIANT_LABELS[ALL]
        )
    return VARIANT_LABELS[variant]


def is_known(name: str) -> bool:
    try:
        parse_feature_set(name)
    except UnknownFeatureSetError:
        return False
    return True
