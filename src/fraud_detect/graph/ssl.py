"""Model 5's `gnnssl` block: self-supervised node embeddings per relation.

For each relation, a Deep Graph Infomax model (Velickovic et al., ICLR 2019;
PyG's `DeepGraphInfomax`) learns a 64-d embedding of every node without any
label: an encoder must tell the real graph from one whose node feature rows
were shuffled, by agreeing with a summary of the whole graph. Each relation's
embedding is then compressed by PCA, giving gnnssl__<relation>_pc01, ...

Encoder: `layers` rounds of GCN propagation, each H <- PReLU(A_hat H W + b)
with A_hat = D^-1/2 (A + I) D^-1/2 stored as one sparse matrix. The full graph
is processed at once (11,944 nodes on Amazon) and messages are never copied
per edge: U-S-U alone has about 7.1M directed edges.

Leakage rules (AGENTS.md):
- No label reaches this module (invariant 1). The entry point takes the own
  features, edge lists and two masks derived from the split column;
  tests/test_gnn_ssl.py permutes the labels and expects identical output.
- The feature scaler that feeds DGI and each PCA are fit on training nodes
  only (`fit_mask`, invariant 6) and then applied to every node.
- The DGI loss and the summary vector cover non-test nodes only (`loss_mask`:
  train and unlabelled). Test nodes take part in message passing, which
  invariant 5 allows, but add no loss term.

Reproducibility: with a fixed seed the CPU result is bit-identical between
runs. CUDA's sparse products are not: two GPU runs differ by about 1e-7 per
step, which early stopping can turn into slightly different embeddings. The
device used is recorded with the result.

Needs the optional `gnn` extra (torch, torch_geometric).
"""

from __future__ import annotations

import copy
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import DeepGraphInfomax

from fraud_detect import columns as C
from fraud_detect.features.graph import adjacency, own_matrix

DEVICES = ("auto", "cpu", "cuda")
# The relation whose edges `usu_edge_fraction` thins out (Amazon's densest).
SAMPLED_RELATION = "usu"


@dataclass(frozen=True)
class SSLConfig:
    """The `gnn_embed` config section, plus the seed."""

    layers: int = 2
    hidden: int = 64
    out_dim: int = 64
    lr: float = 1e-3
    max_epochs: int = 300
    patience: int = 20
    pca_components: int = 8
    device: str = "auto"
    usu_edge_fraction: float = 1.0
    seed: int = 0

    @classmethod
    def from_config(cls, section: Mapping[str, Any], seed: int) -> SSLConfig:
        cfg = cls(**{**section, "seed": seed})
        if cfg.device not in DEVICES:
            raise ValueError(f"gnn_embed.device must be one of {DEVICES}")
        if cfg.layers < 1 or not 0 < cfg.usu_edge_fraction <= 1:
            raise ValueError("gnn_embed needs layers >= 1 and 0 < usu_edge_fraction <= 1")
        return cfg

    def edge_fraction(self, relation: str) -> float:
        return self.usu_edge_fraction if relation == SAMPLED_RELATION else 1.0


def resolve_device(name: str) -> str:
    """`auto`: CUDA if torch can use it, else the CPU. Never assumes CUDA."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("gnn_embed.device is cuda, but torch cannot use CUDA here")
    return name


def device_name(device: str) -> str | None:
    """The GPU's name for a CUDA device (for the run log), else None."""
    return torch.cuda.get_device_name(0) if device == "cuda" else None


# --------------------------------------------------------------------------
# Train-only transforms (invariant 6)
# --------------------------------------------------------------------------
def fit_scaler(x: np.ndarray, fit_mask: np.ndarray) -> StandardScaler:
    """Standardisation fit on the `fit_mask` rows (training nodes) only."""
    return StandardScaler().fit(x[fit_mask])


def fit_pca(z: np.ndarray, fit_mask: np.ndarray, k: int, seed: int) -> PCA:
    """PCA fit on the `fit_mask` rows (training nodes) only."""
    return PCA(n_components=k, svd_solver="full", random_state=seed).fit(z[fit_mask])


def check_masks(n: int, fit_mask: np.ndarray, loss_mask: np.ndarray) -> None:
    for name, mask in (("fit_mask", fit_mask), ("loss_mask", loss_mask)):
        if mask.dtype != bool or mask.shape != (n,):
            raise ValueError(f"{name} must be a boolean array with one entry per node")
    if not fit_mask.any() or (fit_mask & ~loss_mask).any():
        raise ValueError(
            "fit_mask (training nodes) must be non-empty and inside loss_mask"
        )


# --------------------------------------------------------------------------
# Encoder over a sparse, GCN-normalised adjacency
# --------------------------------------------------------------------------
def sample_edges(edges: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """A seeded random subset of `fraction` of the undirected edges."""
    if fraction >= 1.0:
        return edges
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(edges.shape[1], round(fraction * edges.shape[1]), False))
    return edges[:, keep]


def normalized_adjacency(edges: np.ndarray, n: int, device: str) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2 as a sparse CSR tensor (float32) on `device`."""
    a = adjacency(edges, n) + sp.identity(n, format="csr")
    d = 1.0 / np.sqrt(np.asarray(a.sum(axis=1)).ravel())
    a_hat = (sp.diags(d) @ a @ sp.diags(d)).tocsr()
    a_hat.sort_indices()
    with warnings.catch_warnings():
        # torch labels sparse CSR "beta"; only the product with a dense matrix
        # is used, which the tests check.
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support")
        return torch.sparse_csr_tensor(
            torch.from_numpy(a_hat.indptr.astype(np.int64)),
            torch.from_numpy(a_hat.indices.astype(np.int64)),
            torch.from_numpy(a_hat.data.astype(np.float32)),
            size=(n, n),
        ).to(device)


class _Propagate(torch.autograd.Function):
    """A_hat @ H for a symmetric sparse A_hat. Its gradient with respect to H is
    A_hat^T @ G = A_hat @ G, so the backward pass is the same sparse product
    (no transposed copy of A_hat, no per-edge messages)."""

    @staticmethod
    def forward(ctx: Any, a_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ctx.a_hat = a_hat
        return a_hat @ h

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, ctx.a_hat @ grad


class GCNEncoder(torch.nn.Module):
    """`layers` rounds of H <- PReLU(A_hat H W + b)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.weights = torch.nn.ModuleList(
            torch.nn.Linear(a, b, bias=False)
            for a, b in zip(dims, dims[1:], strict=False)
        )
        self.biases = torch.nn.ParameterList(
            torch.nn.Parameter(torch.zeros(b)) for b in dims[1:]
        )
        self.activations = torch.nn.ModuleList(torch.nn.PReLU(b) for b in dims[1:])

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = x
        for weight, bias, act in zip(self.weights, self.biases, self.activations,
                                     strict=True):  # fmt: skip
            h = act(_Propagate.apply(a_hat, weight(h)) + bias)
        return h


# --------------------------------------------------------------------------
# Deep Graph Infomax for one relation
# --------------------------------------------------------------------------
@dataclass
class RelationResult:
    embedding: np.ndarray  # (n_nodes, out_dim) float32, from the best epoch
    log: dict[str, Any]  # loss curve, best epoch, device, timing, edges used


def train_dgi(
    x: np.ndarray,
    edges: np.ndarray,
    loss_mask: np.ndarray,
    cfg: SSLConfig,
    device: str,
) -> RelationResult:
    """Train DGI on one relation and return every node's embedding from the
    epoch with the lowest loss. `x` is the standardised feature matrix (node
    order). The loss and summary use `loss_mask` nodes only."""
    start = time.perf_counter()
    n = len(x)
    torch.manual_seed(cfg.seed)
    a_hat = normalized_adjacency(edges, n, device)
    features = torch.as_tensor(x, dtype=torch.float32, device=device)
    loss_idx = torch.as_tensor(np.flatnonzero(loss_mask), device=device)
    shuffle = torch.Generator(device=device)
    shuffle.manual_seed(cfg.seed)

    def corruption(h: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        perm = torch.randperm(h.size(0), generator=shuffle, device=h.device)
        return h[perm], a

    def summary(z: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.sigmoid(z[loss_idx].mean(dim=0))

    model = DeepGraphInfomax(
        cfg.out_dim,
        encoder=GCNEncoder(x.shape[1], cfg.hidden, cfg.out_dim, cfg.layers),
        summary=summary,
        corruption=corruption,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    losses: list[float] = []
    best_loss, best_epoch, best_state, waited = float("inf"), 0, None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, s = model(features, a_hat)
        loss = model.loss(pos_z[loss_idx], neg_z[loss_idx], s)
        value = float(loss.detach())
        losses.append(value)
        if value < best_loss:
            # The parameters that produced this loss, before this step.
            best_loss, best_epoch, waited = value, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= cfg.patience:
                break
        loss.backward()
        optimizer.step()
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model.encoder(features, a_hat).cpu().numpy().astype(np.float32)
    return RelationResult(
        embedding=z,
        log={
            "device": device,
            "losses": losses,
            "epochs_run": len(losses),
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "stopped_early": len(losses) < cfg.max_epochs,
            "n_edges_used": int(edges.shape[1]),
            "seconds": round(time.perf_counter() - start, 2),
        },
    )


def embed_relation(
    x: np.ndarray, edges: np.ndarray, loss_mask: np.ndarray, cfg: SSLConfig, relation: str
) -> RelationResult:
    """One relation's embedding: edges thinned if configured, trained on the
    configured device; a CUDA out-of-memory error retries on the CPU."""
    fraction = cfg.edge_fraction(relation)
    used = sample_edges(edges, fraction, cfg.seed)
    device = resolve_device(cfg.device)
    fell_back = False
    try:
        result = train_dgi(x, used, loss_mask, cfg, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"WARNING: CUDA ran out of memory on {relation}; retrying on the CPU.")
        result = train_dgi(x, used, loss_mask, cfg, "cpu")
        fell_back = True
    result.log.update(
        {
            "relation": relation,
            "n_edges": int(edges.shape[1]),
            "edge_fraction": fraction,
            "cuda_oom_fell_back_to_cpu": fell_back,
        }
    )
    return result


def standardized_features(own_features: pd.DataFrame, fit_mask: np.ndarray) -> np.ndarray:
    """The own features in node order, standardised with a scaler fit on the
    training nodes only."""
    x, _ = own_matrix(own_features)
    if len(fit_mask) != len(x):
        raise ValueError(f"fit_mask has {len(fit_mask)} entries for {len(x)} nodes")
    return fit_scaler(x, fit_mask).transform(x)


# --------------------------------------------------------------------------
# PCA per relation -> the gnnssl block
# --------------------------------------------------------------------------
def pca_block(
    embeddings: Mapping[str, np.ndarray],
    fit_mask: np.ndarray,
    k: int,
    seed: int,
    name: str = C.GNNSSL_BLOCK,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """<name>__<relation>_pc01..pcK (name gnnssl, or e.g. gnnssl_no36) for each
    relation in order, from a PCA fit on the training nodes' embeddings, and
    each PCA's explained variance."""
    n = len(fit_mask)
    out: dict[str, np.ndarray] = {C.NODE_ID: np.arange(n, dtype=np.int64)}
    info = {}
    width = max(2, len(str(k)))
    for relation, z in embeddings.items():
        if z.shape[0] != n:
            raise ValueError(f"{relation}: {z.shape[0]} embedding rows for {n} nodes")
        pca = fit_pca(z, fit_mask, k, seed)
        scores = pca.transform(z)
        for j in range(k):
            column = f"{name}{C.BLOCK_SEPARATOR}{relation}_pc{j + 1:0{width}d}"
            out[column] = scores[:, j].astype(np.float64)
        ratio = pca.explained_variance_ratio_
        info[relation] = {
            "explained_variance_ratio": [float(v) for v in ratio],
            "explained_variance_total": float(ratio.sum()),
        }
    return pd.DataFrame(out), info


@dataclass
class SSLResult:
    block: pd.DataFrame  # the gnnssl block
    embeddings: dict[str, np.ndarray]  # raw (n_nodes, out_dim) per relation
    logs: dict[str, dict[str, Any]]  # training log and PCA variance per relation


def gnnssl_block(
    own_features: pd.DataFrame,
    edges: Mapping[str, np.ndarray],
    fit_mask: np.ndarray,
    loss_mask: np.ndarray,
    config: SSLConfig,
) -> SSLResult:
    """The whole gnnssl block in one call (the gnn_embed stage runs the same
    steps with one saved unit per relation). `fit_mask` marks the training
    nodes and `loss_mask` the non-test nodes, both derived from the split
    before the call; no label is passed."""
    check_masks(len(own_features), fit_mask, loss_mask)
    x = standardized_features(own_features, fit_mask)
    results = {r: embed_relation(x, e, loss_mask, config, r) for r, e in edges.items()}
    embeddings = {r: res.embedding for r, res in results.items()}
    block, pca_info = pca_block(embeddings, fit_mask, config.pca_components, config.seed)
    logs = {r: {**res.log, **pca_info[r]} for r, res in results.items()}
    return SSLResult(block=block, embeddings=embeddings, logs=logs)
