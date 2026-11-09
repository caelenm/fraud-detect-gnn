"""Supervised GNNs: classifiers (Models 3-4) and out-of-fold scores (Model 6).

Heterogeneous GraphSAGE (E5, Model 4; docs/GNN_MODELS_PLAN.md). One layer is

    h' = W_root h + b + sum_r W_r (A_hat_r h)

with A_hat_r = D_r^-1 A_r, the mean over a node's neighbours in relation r (0
for a node without any). That is what PyG's HeteroConv({r: SAGEConv(aggr=
"mean")}, aggr="sum") and RGCNConv(aggr="mean") compute on a graph with one
node type; tests/test_gnn_supervised.py checks both. The R-GCN-style variant
(E5g) uses A_hat_r = D_r^-1/2 A_r D_r^-1/2 instead. With no relations the
network is an MLP of the same width and depth (E6, Model 3), and on the
rewired graph of N2 it is the null-graph control N3.

Each product with A_hat_r is one sparse matrix product (no per-edge
messages: U-S-U alone has about 7.1M directed edges); its backward pass uses
the stored transpose. Layer 1's propagation of the fixed input is computed
once per graph. The full graph is processed at once (full batch).

Larger rungs (docs/BIGGER_GNN_PLAN.md, a draft): a rung may override these
settings with its own `train:` block (rung_config). For the GraphSAGE network
that adds depth, width, residual connections, LayerNorm and a jumping-
knowledge head (the head reads every layer's output). `model: sign` is SIGN
(Frasca et al., 2020): x and A_hat_r^k x for k = 1..hops are computed once,
without labels, and an MLP with one linear branch per input is trained on
them. The defaults are the 2026-10-31 network exactly, so the rungs already
run (E5, E5g, E6, N3) are unchanged.

GAT (`model: gat`, docs/GAT_DEEP_PLAN.md) keeps the root term and replaces
each relation's fixed A_hat_r with learned attention:

    h'_i = W_root h_i + b + sum_r || over heads k of
           sum_{j in N_r(i)} alpha^{r,k}_ij W^{r,k} h_j
    alpha^{r,k}_ij = softmax over j of LeakyReLU_0.2(a_dst^T W h_i + a_src^T W h_j)

per relation: PyG's GATConv(heads, concat=True, add_self_loops=False,
bias=False) for each relation, summed; a node with no neighbours in a
relation gets 0 from it. The weighted sum over edges is a custom autograd
function (_AttentionAggregate) built from per-head sparse products: forward
A_alpha V and grad_V = A_alpha^T G as CSR x dense products, grad_alpha as a
sampled dense product (torch.sparse.sampled_addmm) at the edges. No
edges x hidden tensor is made (R-S-R has about 6.8M directed edges), and on
the CPU every step is deterministic. The residual, layer_norm and jk options
apply to it as to the sage layers.

Training, per fit: binary cross-entropy on the fit nodes' labels, weighted by
negatives/positives; Adam with weight decay; early stopping on the PR-AUC of
a grouped, stratified fifth of the fit's own nodes (the inner validation),
keeping the epoch with the best one. `cross_fit` fits one model per saved CV
fold: fold k's out-of-fold predictions come from the model fit without fold
k, and every other node (test, unlabelled) gets the mean of the fold models.

Leakage rules (AGENTS.md):
- Labels enter only through each fit's training loss and its inner
  validation (invariant 1). A fit receives only the labels of the nodes it
  is fit and validated on; the functions here never receive a test label,
  and fold k's labels never reach fold k's out-of-fold predictions.
- `nested_cross_fit` builds Model 6's gnnsup block, the one documented
  exception to invariant 1 (2026-10-31): besides the out-of-fold scores, for
  each outer fold k it gives the other folds' scores from models fit without
  fold k, so the shared CV never fits on a score built with the held-out
  fold's labels.
- The feature scaler is fit on training nodes only (invariant 6, done by the
  caller); test and unlabelled nodes take part in message passing only
  (invariant 5).

Reproducibility: on the CPU a fit is bit-identical for a given seed.
Needs the optional `gnn` extra (torch).
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from itertools import product
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from fraud_detect.data.split import grouped_folds
from fraud_detect.features.graph import adjacency

NORMS = ("mean", "sym")
DEVICES = ("cpu", "cuda")
MODELS = ("sage", "sign", "gat")


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Setting:
    """One point of the tuning grid."""

    hidden: int
    dropout: float
    lr: float

    @property
    def key(self) -> str:
        return f"h{self.hidden}_d{self.dropout:g}_lr{self.lr:g}"


@dataclass(frozen=True)
class TrainConfig:
    """The gnn_classify config section (grid values as tuples)."""

    layers: int = 2
    hidden: tuple[int, ...] = (64, 128)
    dropout: tuple[float, ...] = (0.0, 0.3)
    lr: tuple[float, ...] = (1e-3, 5e-3)
    weight_decay: float = 5e-4
    max_epochs: int = 300
    patience: int = 20
    inner_folds: int = 5
    seeds: int = 5
    device: str = "cpu"
    # The architecture (larger rungs, docs/BIGGER_GNN_PLAN.md). The defaults
    # are the 2026-10-31 network.
    model: str = "sage"  # sage: RelSAGE layers; sign: SIGN; gat: RelGAT layers
    residual: bool = False  # sage: h + layer(h) from the second layer on
    layer_norm: bool = False  # LayerNorm on each layer's output, before ReLU
    jk: bool = False  # sage: the head reads every layer's output, concatenated
    hops: int = 3  # sign: A_hat_r^k x for k = 1..hops, per relation
    heads: int = 4  # gat: attention heads per relation, hidden / heads wide each

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(section) - known)
        if unknown:
            raise ValueError(f"Unknown gnn_classify settings: {unknown}")
        values = dict(section)
        for key in ("hidden", "dropout", "lr"):
            if key in values:
                raw = values[key]
                values[key] = tuple(raw if isinstance(raw, list | tuple) else [raw])
        cfg = cls(**values)
        if cfg.device not in DEVICES:
            raise ValueError(f"gnn_classify.device must be one of {DEVICES}")
        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "gnn_classify.device is cuda, but torch cannot use CUDA here"
            )
        if cfg.layers < 1 or cfg.seeds < 1 or cfg.inner_folds < 2 or cfg.patience < 1:
            raise ValueError("gnn_classify needs layers, seeds, patience >= 1 and "
                             "inner_folds >= 2")  # fmt: skip
        if not (cfg.hidden and cfg.dropout and cfg.lr):
            raise ValueError(
                "gnn_classify needs at least one hidden, dropout and lr value"
            )
        if cfg.model not in MODELS:
            raise ValueError(f"gnn_classify.model must be one of {MODELS}")
        flags = (cfg.residual, cfg.layer_norm, cfg.jk)
        if not all(isinstance(f, bool) for f in flags):
            raise ValueError("gnn_classify.residual, layer_norm and jk are true or false")
        if cfg.model == "sign" and (cfg.residual or cfg.jk):
            raise ValueError(
                "residual and jk apply to the sage model (and gat), not sign"
            )
        if not isinstance(cfg.hops, int) or isinstance(cfg.hops, bool) or cfg.hops < 1:
            raise ValueError("gnn_classify.hops must be a whole number >= 1")
        if cfg.model == "gat" and (
            not isinstance(cfg.heads, int)
            or isinstance(cfg.heads, bool)
            or cfg.heads < 1
            or any(int(h) % cfg.heads for h in cfg.hidden)
        ):
            raise ValueError(
                f"gat needs heads >= 1 dividing every hidden size ({cfg.heads} heads, "
                f"hidden {list(cfg.hidden)})"
            )
        return cfg

    def grid(self) -> list[Setting]:
        """Every setting, in a fixed order (ties go to the earlier one)."""
        return [
            Setting(int(h), float(d), float(r))
            for h, d, r in product(self.hidden, self.dropout, self.lr)
        ]


def rung_config(section: Mapping[str, Any], spec: Mapping[str, Any]) -> TrainConfig:
    """The TrainConfig of one rung: the gnn_classify section, with the rung's
    own `train:` block (gnn_models.<rung>.train) on top. A rung without one
    gets the section as it is."""
    return TrainConfig.from_config({**section, **(spec.get("train") or {})})


def fit_seed(seed: int, outer: int, inner: int | None = None) -> int:
    """The seed of one fit: the outer fold's fit (inner None) or an inner fit
    of nested cross-fitting. Distinct for every (seed, outer, inner)."""
    return seed * 100 + outer * 10 + (9 if inner is None else inner)


# --------------------------------------------------------------------------
# Graph input: normalised sparse adjacencies and the fixed first propagation
# --------------------------------------------------------------------------
def normalised_adjacency(edges: np.ndarray, n: int, norm: str) -> sp.csr_matrix:
    """A_hat for one relation: D^-1 A ("mean") or D^-1/2 A D^-1/2 ("sym"),
    without self-loops; rows of isolated nodes are zero."""
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
    a = adjacency(edges, n)
    deg = np.asarray(a.sum(axis=1)).ravel()
    power = 1.0 if norm == "mean" else 0.5
    inv = np.zeros_like(deg)
    np.power(deg, -power, out=inv, where=deg > 0)
    a_hat = sp.diags(inv) @ a if norm == "mean" else sp.diags(inv) @ a @ sp.diags(inv)
    a_hat = sp.csr_matrix(a_hat)
    a_hat.sort_indices()
    return a_hat


def _torch_csr(a: sp.csr_matrix, device: str) -> torch.Tensor:
    a = sp.csr_matrix(a)
    a.sort_indices()
    with warnings.catch_warnings():
        # torch labels sparse CSR "beta"; only products with dense matrices are
        # used, which the tests check against dense arithmetic and PyG.
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support")
        return torch.sparse_csr_tensor(
            torch.from_numpy(a.indptr.astype(np.int64)),
            torch.from_numpy(a.indices.astype(np.int64)),
            torch.from_numpy(a.data.astype(np.float32)),
            size=a.shape,
        ).to(device)


class _SparseProduct(torch.autograd.Function):
    """A_hat @ H, with gradient A_hat^T @ G with respect to H (A_hat^T is
    passed in, since a mean-normalised A_hat is not symmetric)."""

    @staticmethod
    def forward(
        ctx: Any, a: torch.Tensor, a_t: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        ctx.a_t = a_t
        return a @ h

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[None, None, torch.Tensor]:
        return None, None, ctx.a_t @ grad


@dataclass
class RelationEdges:
    """One relation's directed edges for attention, sorted by dst then src
    (edge e goes src[e] -> dst[e]), with the CSR structure of A (rows dst,
    columns src) and of A^T (rows src: edge t_perm[k] is the k-th)."""

    src: torch.Tensor
    dst: torch.Tensor
    indptr: torch.Tensor
    t_perm: torch.Tensor
    t_indptr: torch.Tensor
    t_indices: torch.Tensor
    n: int

    @classmethod
    def build(
        cls, src: np.ndarray, dst: np.ndarray, n: int, device: str = "cpu"
    ) -> RelationEdges:
        src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
        order = np.lexsort((src, dst))
        src, dst = src[order], dst[order]
        t_perm = np.lexsort((dst, src))

        def indptr(rows: np.ndarray) -> np.ndarray:
            return np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))])

        def tensor(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64)).to(device)

        return cls(src=tensor(src), dst=tensor(dst), indptr=tensor(indptr(dst)),
                   t_perm=tensor(t_perm), t_indptr=tensor(indptr(src)),
                   t_indices=tensor(dst[t_perm]), n=n)  # fmt: skip

    def csr(self, values: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        """A (or A^T) with `values` given in edge order, as sparse CSR."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sparse CSR tensor support")
            size = (self.n, self.n)
            if transpose:
                return torch.sparse_csr_tensor(
                    self.t_indptr, self.t_indices, values[self.t_perm], size
                )
            return torch.sparse_csr_tensor(self.indptr, self.src, values, size)


def directed_edges(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both directions of an undirected (2, E) edge list (src < dst, no
    self-loops), as (src, dst) int64 arrays sorted by dst, then src."""
    e = np.asarray(edges, dtype=np.int64)
    src = np.concatenate([e[0], e[1]])
    dst = np.concatenate([e[1], e[0]])
    order = np.lexsort((src, dst))
    return src[order], dst[order]


@dataclass
class GraphInput:
    """What every fit on one graph shares: the standardised features, each
    relation's (A_hat, A_hat^T), and A_hat x per relation (layer 1's fixed
    propagation)."""

    x: torch.Tensor
    adjs: list[tuple[torch.Tensor, torch.Tensor]]
    x_prop: list[torch.Tensor]
    relations: list[str]
    norm: str
    n_edges: dict[str, int]
    # SIGN's inputs per number of hops, computed once (hop_features).
    hop_cache: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    # gat: per relation, both directions of every edge (RelationEdges); the
    # sparse A_hat products above are not built for it.
    model: str = "sage"
    edge_index: list[RelationEdges] = field(default_factory=list)


def hop_features(g: GraphInput, hops: int) -> list[torch.Tensor]:
    """SIGN's fixed inputs: x, then A_hat_r^k x for k = 1..hops, relation by
    relation. Label-free (features and edges only, invariant 5); computed
    once per graph and kept on it. Without relations it is [x]."""
    if hops not in g.hop_cache:
        inputs = [g.x]
        for (a, _), h in zip(g.adjs, g.x_prop, strict=True):
            inputs.append(h)
            for _ in range(hops - 1):
                h = a @ h
                inputs.append(h)
        g.hop_cache[hops] = inputs
    return g.hop_cache[hops]


def graph_input(
    x: np.ndarray,
    edges: Mapping[str, np.ndarray],
    norm: str,
    device: str = "cpu",
    model: str = "sage",
) -> GraphInput:
    """The graph input for a feature matrix `x` (node order) and edge lists;
    no edges (an empty mapping) gives the edge-free network (E6). For
    model gat it holds each relation's directed edges instead of A_hat (norm
    unused)."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    n = len(x)
    features = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    adjs, x_prop, edge_index = [], [], []
    for e in edges.values():
        if model == "gat":
            edge_index.append(RelationEdges.build(*directed_edges(e), n, device))
            continue
        a_hat = normalised_adjacency(e, n, norm)
        a, a_t = _torch_csr(a_hat, device), _torch_csr(a_hat.T, device)
        adjs.append((a, a_t))
        x_prop.append(a @ features)
    return GraphInput(
        x=features,
        adjs=adjs,
        x_prop=x_prop,
        relations=list(edges),
        norm=norm,
        n_edges={r: int(np.asarray(e).shape[1]) for r, e in edges.items()},
        model="gat" if model == "gat" else "sage",
        edge_index=edge_index,
    )


class _AttentionAggregate(torch.autograd.Function):
    """out[i, k] = sum over edges (j -> i) of alpha[e, k] * v[j, k], per head k.

    alpha: (E, H) edge weights in `edges` order; v: (N, H, C). Per head:
    forward A_alpha v as a CSR x dense product; backward grad_v = A_alpha^T g
    (CSR x dense) and grad_alpha[e] = <g[dst_e], v[src_e]> (sampled dense
    product at the edges). Deterministic on the CPU."""

    @staticmethod
    def forward(
        ctx: Any, alpha: torch.Tensor, v: torch.Tensor, edges: RelationEdges
    ) -> torch.Tensor:
        out = torch.stack(
            [edges.csr(alpha[:, k].contiguous()) @ v[:, k] for k in range(v.shape[1])],
            dim=1,
        )
        ctx.save_for_backward(alpha, v)
        ctx.edges = edges
        return out

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:
        alpha, v = ctx.saved_tensors
        edges, heads = ctx.edges, v.shape[1]
        grad = grad.contiguous()
        grad_alpha = grad_v = None
        if ctx.needs_input_grad[1]:
            grad_v = torch.stack(
                [edges.csr(alpha[:, k], transpose=True) @ grad[:, k]
                 for k in range(heads)], dim=1,
            )  # fmt: skip
        if ctx.needs_input_grad[0]:
            pattern = edges.csr(alpha.new_zeros(len(edges.src)))
            grad_alpha = torch.stack(
                [torch.sparse.sampled_addmm(pattern, grad[:, k].contiguous(),
                                            v[:, k].T.contiguous(), beta=0.0).values()
                 for k in range(heads)], dim=1,
            )  # fmt: skip
        return grad_alpha, grad_v, None


def attention_aggregate(
    alpha: torch.Tensor, v: torch.Tensor, edges: RelationEdges
) -> torch.Tensor:
    return _AttentionAggregate.apply(alpha, v, edges)


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------
class RelSAGE(torch.nn.Module):
    """h' = W_root h + b + sum_r W_r (A_hat_r h)."""

    def __init__(self, in_dim: int, out_dim: int, n_relations: int) -> None:
        super().__init__()
        self.root = torch.nn.Linear(in_dim, out_dim)
        self.rel = torch.nn.ModuleList(
            torch.nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)
        )

    def forward(
        self,
        h: torch.Tensor,
        adjs: Sequence[tuple[torch.Tensor, torch.Tensor]],
        propagated: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        out = self.root(h)
        for j, lin in enumerate(self.rel):
            if propagated is not None:  # A_hat h already computed: W (A_hat h)
                out = out + lin(propagated[j])
            else:  # A_hat (h W), the same value with the smaller product
                a, a_t = adjs[j]
                out = out + _SparseProduct.apply(a, a_t, lin(h))
        return out


class RelGAT(torch.nn.Module):
    """h' = W_root h + b + sum_r GAT_r(h): per relation, multi-head attention
    over the incoming edges with `heads` heads of out_dim / heads each,
    concatenated. Attention dropout `attn_dropout` on the edge weights."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_relations: int,
        heads: int,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_dim % heads:
            raise ValueError(f"out_dim {out_dim} is not divisible by heads {heads}")
        self.heads, self.channels = heads, out_dim // heads
        self.attn_dropout = attn_dropout
        self.root = torch.nn.Linear(in_dim, out_dim)
        self.rel = torch.nn.ModuleList(
            torch.nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)
        )
        self.att_src = torch.nn.ParameterList(
            torch.nn.Parameter(torch.empty(heads, self.channels))
            for _ in range(n_relations)
        )
        self.att_dst = torch.nn.ParameterList(
            torch.nn.Parameter(torch.empty(heads, self.channels))
            for _ in range(n_relations)
        )
        for a in [*self.att_src, *self.att_dst]:
            torch.nn.init.xavier_uniform_(a)

    def relation(self, j: int, h: torch.Tensor, edges: RelationEdges) -> torch.Tensor:
        """One relation's attention term, (N, out_dim)."""
        from torch_geometric.utils import softmax  # noqa: PLC0415

        n, src, dst = h.shape[0], edges.src, edges.dst
        v = self.rel[j](h).view(n, self.heads, self.channels)
        score_src = (v * self.att_src[j]).sum(-1)  # (N, H)
        score_dst = (v * self.att_dst[j]).sum(-1)
        e = F.leaky_relu(score_src[src] + score_dst[dst], negative_slope=0.2)
        alpha = softmax(e, dst, num_nodes=n)  # over each node's incoming edges
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)
        return attention_aggregate(alpha, v, edges).reshape(n, -1)

    def forward(
        self, h: torch.Tensor, edge_index: Sequence[RelationEdges]
    ) -> torch.Tensor:
        out = self.root(h)
        for j, edges in enumerate(edge_index):
            out = out + self.relation(j, h, edges)
        return out


class GNNClassifier(torch.nn.Module):
    """`layers` RelSAGE (or, with conv "gat", RelGAT) layers, each followed
    by ReLU and dropout, then a linear head giving one logit per node.

    Options for the larger rungs, all off by default (then this is the
    2026-10-31 network, parameter for parameter):
    - layer_norm: LayerNorm on each layer's output, before the ReLU;
    - residual: from the second layer on, the layer's input is added to its
      output (the first layer changes the width, so it has none);
    - jk: the head reads every layer's output, concatenated (jumping
      knowledge), not only the last one's."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        n_relations: int,
        layers: int,
        dropout: float,
        residual: bool = False,
        layer_norm: bool = False,
        jk: bool = False,
        conv: str = "sage",
        heads: int = 4,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        pairs = list(zip(dims, dims[1:], strict=False))
        if conv == "gat":
            self.convs = torch.nn.ModuleList(
                RelGAT(a, b, n_relations, heads, attn_dropout=dropout) for a, b in pairs
            )
        elif conv == "sage":
            self.convs = torch.nn.ModuleList(RelSAGE(a, b, n_relations) for a, b in pairs)
        else:
            raise ValueError(f"conv must be sage or gat, got {conv!r}")
        self.conv = conv
        self.norms = (
            torch.nn.ModuleList(torch.nn.LayerNorm(hidden) for _ in range(layers))
            if layer_norm
            else None
        )
        self.head = torch.nn.Linear(hidden * layers if jk else hidden, 1)
        self.dropout = dropout
        self.residual = residual
        self.jk = jk

    def forward(self, g: GraphInput) -> torch.Tensor:
        h = g.x
        outputs = []
        for i, conv in enumerate(self.convs):
            if self.conv == "gat":
                z = conv(h, g.edge_index)
            else:
                z = conv(h, g.adjs, g.x_prop if i == 0 else None)
            if self.norms is not None:
                z = self.norms[i](z)
            z = F.relu(z)
            if self.residual and i > 0:
                z = z + h
            h = F.dropout(z, p=self.dropout, training=self.training)
            outputs.append(h)
        return self.head(torch.cat(outputs, dim=1) if self.jk else h).squeeze(-1)


class SIGNClassifier(torch.nn.Module):
    """SIGN (Frasca et al., 2020, "SIGN: Scalable Inception Graph Neural
    Networks"): one linear branch (width `hidden`) per precomputed input of
    hop_features, concatenated, then `layers - 1` hidden layers and a linear
    head. Each branch and hidden layer: optional LayerNorm, ReLU, dropout.
    The graph enters only through the fixed, label-free hop features."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        n_relations: int,
        hops: int,
        layers: int,
        dropout: float,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        n_inputs = 1 + n_relations * hops
        self.hops = hops
        self.branches = torch.nn.ModuleList(
            torch.nn.Linear(in_dim, hidden) for _ in range(n_inputs)
        )
        dims = [hidden * n_inputs] + [hidden] * (layers - 1)
        self.mlp = torch.nn.ModuleList(
            torch.nn.Linear(a, b) for a, b in zip(dims, dims[1:], strict=False)
        )
        if layer_norm:
            self.branch_norms = torch.nn.ModuleList(
                torch.nn.LayerNorm(hidden) for _ in range(n_inputs)
            )
            self.mlp_norms = torch.nn.ModuleList(
                torch.nn.LayerNorm(hidden) for _ in range(layers - 1)
            )
        else:
            self.branch_norms = self.mlp_norms = None
        self.head = torch.nn.Linear(dims[-1], 1)
        self.dropout = dropout

    def _block(self, z: torch.Tensor, norm: torch.nn.Module | None) -> torch.Tensor:
        if norm is not None:
            z = norm(z)
        return F.dropout(F.relu(z), p=self.dropout, training=self.training)

    def forward(self, g: GraphInput) -> torch.Tensor:
        inputs = hop_features(g, self.hops)
        h = torch.cat(
            [
                self._block(lin(x), None if self.branch_norms is None
                            else self.branch_norms[i])
                for i, (lin, x) in enumerate(zip(self.branches, inputs, strict=True))
            ],
            dim=1,
        )  # fmt: skip
        for i, lin in enumerate(self.mlp):
            h = self._block(lin(h), None if self.mlp_norms is None else self.mlp_norms[i])
        return self.head(h).squeeze(-1)


def build_model(g: GraphInput, setting: Setting, cfg: TrainConfig) -> torch.nn.Module:
    """The network of one fit, before training (draws its initial weights from
    torch's global generator: seed it first)."""
    if cfg.model == "sign":
        return SIGNClassifier(g.x.shape[1], setting.hidden, len(g.adjs), cfg.hops,
                              cfg.layers, setting.dropout, cfg.layer_norm)  # fmt: skip
    if (g.model == "gat") != (cfg.model == "gat"):
        raise ValueError(f"graph input built for {g.model!r}, network is {cfg.model!r}")
    if cfg.model == "gat":
        return GNNClassifier(g.x.shape[1], setting.hidden, len(g.edge_index),
                             cfg.layers, setting.dropout, residual=cfg.residual,
                             layer_norm=cfg.layer_norm, jk=cfg.jk, conv="gat",
                             heads=cfg.heads)  # fmt: skip
    return GNNClassifier(g.x.shape[1], setting.hidden, len(g.adjs), cfg.layers,
                         setting.dropout, residual=cfg.residual,
                         layer_norm=cfg.layer_norm, jk=cfg.jk)  # fmt: skip


def n_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# One fit
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    logits: np.ndarray  # every node, from the epoch with the best validation PR-AUC
    log: dict[str, Any]


def fit_predict(
    g: GraphInput,
    fit_nodes: np.ndarray,
    fit_y: np.ndarray,
    val_nodes: np.ndarray,
    val_y: np.ndarray,
    setting: Setting,
    cfg: TrainConfig,
    seed: int,
) -> FitResult:
    """Train on `fit_nodes` (labels `fit_y`), stop early on the PR-AUC of
    `val_nodes` (labels `val_y`), and return every node's logit from the best
    epoch. No other label is passed in."""
    start = time.perf_counter()
    device = g.x.device
    if cfg.model == "sign":
        hop_features(g, cfg.hops)  # computed once per graph, before the seed
    torch.manual_seed(seed)
    model = build_model(g, setting, cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=setting.lr, weight_decay=cfg.weight_decay
    )
    fit_idx = torch.as_tensor(np.asarray(fit_nodes), dtype=torch.long, device=device)
    val_idx = torch.as_tensor(np.asarray(val_nodes), dtype=torch.long, device=device)
    target = torch.as_tensor(np.asarray(fit_y), dtype=torch.float32, device=device)
    n_pos = float(np.sum(fit_y))
    pos_weight = torch.tensor((len(fit_y) - n_pos) / max(n_pos, 1.0), device=device)
    val_y = np.asarray(val_y, dtype=int)
    best_ap, best_epoch, best_logits, waited, epochs = -1.0, 0, None, 0, 0
    diverged = False
    for epoch in range(1, cfg.max_epochs + 1):
        epochs = epoch
        model.train()
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            model(g)[fit_idx], target, pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(g)
        scores = out[val_idx].cpu().numpy()
        if not np.isfinite(scores).all():
            diverged = True
            break
        ap = float(average_precision_score(val_y, scores))
        if ap > best_ap:
            best_ap, best_epoch, waited = ap, epoch, 0
            best_logits = out.cpu().numpy().astype(np.float64)
        else:
            waited += 1
            if waited >= cfg.patience:
                break
    if best_logits is None:
        raise RuntimeError(f"GNN fit diverged in its first epoch ({setting.key})")
    return FitResult(
        logits=best_logits,
        log={
            "epochs_run": epochs,
            "best_epoch": best_epoch,
            "best_val_pr_auc": best_ap,
            "diverged": diverged,
            "n_fit": len(fit_nodes),
            "n_val": len(val_nodes),
            "n_parameters": n_parameters(model),
            "seconds": round(time.perf_counter() - start, 2),
        },
    )


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _fit_rows(
    g: GraphInput,
    nodes: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    setting: Setting,
    cfg: TrainConfig,
    seed: int,
) -> FitResult:
    """One fit on the labelled `nodes`: a grouped, stratified 1/inner_folds of
    them is the inner validation, the rest the fit nodes."""
    val = grouped_folds(np.asarray(y, dtype=int), np.asarray(groups), cfg.inner_folds,
                        seed) == 0  # fmt: skip
    return fit_predict(g, nodes[~val], y[~val], nodes[val], y[val], setting, cfg, seed)


# --------------------------------------------------------------------------
# Cross-fitting on the saved CV folds
# --------------------------------------------------------------------------
@dataclass
class CrossFit:
    oof: (
        np.ndarray
    )  # P(fraud) per training row (train order), from the model without its fold
    all_nodes: np.ndarray  # mean P(fraud) of the fold models for every node
    logs: list[dict[str, Any]]


def _check_rows(train_nodes: np.ndarray, y_train: np.ndarray, folds: np.ndarray,
                groups: np.ndarray) -> None:  # fmt: skip
    if not (len(train_nodes) == len(y_train) == len(folds) == len(groups)):
        raise ValueError("train_nodes, y_train, folds and groups must align")
    if len(np.unique(folds)) < 2:
        raise ValueError("cross-fitting needs at least two folds")


def cross_fit(
    g: GraphInput,
    train_nodes: np.ndarray,
    y_train: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    setting: Setting,
    cfg: TrainConfig,
    seed: int,
) -> CrossFit:
    """One model per fold k, fit on the training rows of the other folds:
    fold k's out-of-fold P(fraud) from it, and every node's mean over the fold
    models. `y_train`, `folds` and `groups` describe the training rows only."""
    train_nodes, y_train = np.asarray(train_nodes), np.asarray(y_train, dtype=int)
    folds, groups = np.asarray(folds), np.asarray(groups)
    _check_rows(train_nodes, y_train, folds, groups)
    oof = np.full(len(train_nodes), np.nan)
    probs, logs = [], []
    for k in sorted(np.unique(folds)):
        held = folds == k
        res = _fit_rows(g, train_nodes[~held], y_train[~held], groups[~held], setting,
                        cfg, fit_seed(seed, int(k)))  # fmt: skip
        prob = sigmoid(res.logits)
        oof[held] = prob[train_nodes[held]]
        probs.append(prob)
        logs.append({"fold": int(k), **res.log})
    return CrossFit(oof=oof, all_nodes=np.mean(probs, axis=0), logs=logs)


def nested_cross_fit(
    g: GraphInput,
    train_nodes: np.ndarray,
    y_train: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    setting: Setting,
    cfg: TrainConfig,
    seed: int,
) -> tuple[CrossFit, dict[int, np.ndarray]]:
    """cross_fit, plus for each outer fold k the inner out-of-fold P(fraud) of
    the training rows outside fold k, each from a model fit without fold k and
    without the row's own fold (NaN on fold k's rows)."""
    outer = cross_fit(g, train_nodes, y_train, folds, groups, setting, cfg, seed)
    train_nodes, y_train = np.asarray(train_nodes), np.asarray(y_train, dtype=int)
    folds, groups = np.asarray(folds), np.asarray(groups)
    nested: dict[int, np.ndarray] = {}
    ks = sorted(int(k) for k in np.unique(folds))
    for k in ks:
        scores = np.full(len(train_nodes), np.nan)
        for j in ks:
            if j == k:
                continue
            fit = (folds != k) & (folds != j)
            res = _fit_rows(g, train_nodes[fit], y_train[fit], groups[fit], setting, cfg,
                            fit_seed(seed, k, j))  # fmt: skip
            held = folds == j
            scores[held] = sigmoid(res.logits)[train_nodes[held]]
            outer.logs.append({"outer_fold": k, "fold": j, **res.log})
        nested[k] = scores
    return outer, nested


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """log(p / (1 - p)), with p clipped to [eps, 1 - eps]."""
    q = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def fold_pr_auc(y: np.ndarray, prob: np.ndarray, folds: np.ndarray) -> list[float]:
    """PR-AUC of each fold's out-of-fold predictions, in fold order."""
    y, prob, folds = np.asarray(y, dtype=int), np.asarray(prob), np.asarray(folds)
    return [
        float(average_precision_score(y[folds == k], prob[folds == k]))
        for k in sorted(np.unique(folds))
    ]
