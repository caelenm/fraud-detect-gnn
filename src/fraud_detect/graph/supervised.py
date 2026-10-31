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
from dataclasses import dataclass, fields
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
        return cfg

    def grid(self) -> list[Setting]:
        """Every setting, in a fixed order (ties go to the earlier one)."""
        return [
            Setting(int(h), float(d), float(r))
            for h, d, r in product(self.hidden, self.dropout, self.lr)
        ]


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


def graph_input(
    x: np.ndarray, edges: Mapping[str, np.ndarray], norm: str, device: str = "cpu"
) -> GraphInput:
    """The graph input for a feature matrix `x` (node order) and edge lists;
    no edges (an empty mapping) gives the edge-free network (E6)."""
    n = len(x)
    features = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    adjs, x_prop = [], []
    for e in edges.values():
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
    )


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


class GNNClassifier(torch.nn.Module):
    """`layers` RelSAGE layers, each followed by ReLU and dropout, then a
    linear head giving one logit per node."""

    def __init__(
        self, in_dim: int, hidden: int, n_relations: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = torch.nn.ModuleList(
            RelSAGE(a, b, n_relations) for a, b in zip(dims, dims[1:], strict=False)
        )
        self.head = torch.nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, g: GraphInput) -> torch.Tensor:
        h = g.x
        for i, conv in enumerate(self.convs):
            h = conv(h, g.adjs, g.x_prop if i == 0 else None)
            h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)


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
    torch.manual_seed(seed)
    model = GNNClassifier(
        g.x.shape[1], setting.hidden, len(g.adjs), cfg.layers, setting.dropout
    ).to(device)
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
