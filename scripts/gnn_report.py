"""One page of results for the 2026-10-31 GNN round (docs/GNN_MODELS_PLAN.md).

    uv run scripts/gnn_report.py

Reads what the pipeline and the summaries already saved, for Amazon (5
splits) and YelpChi (3 splits), every feature: each setup's Model A (or GNN
rung) per split from reports/model_a.json, the GNN rungs' chosen settings,
gnn_oof's choice of GNN per split, and the verdict tables
outputs/<dataset>/compare/gnn_<dataset>_{primary,secondary}.csv from
scripts/repeat_summary.py. Adds Holm-adjusted p values within each
pre-registered primary family (RESEARCH_PLAN §11, fifth amendment) and the
share of headroom closed, Δ / (1 − PR-AUC of A). Writes docs/GNN_RESULTS.md.
Aggregates only; nothing is chosen here.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detect.config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    ConfigError,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.runstate import atomic_write_text

DATASETS = {"amazon": 5, "yelpchi": 3}
DATASET_LABELS = {"amazon": "Amazon (main; all 25 features)",
                  "yelpchi": "YelpChi (validation; all 32 features)"}  # fmt: skip
# Feature set (without the pilot_ prefix) -> setup ID and description.
SETUPS = {
    "m1_own": ("B", "df-analyze on own features"),
    "m2_own_nbr": ("E1", "+ neighbour means"),
    "m5_own_gnnssl": ("E2", "+ DGI embeddings"),
    "m5r_own_gnnrand": ("E3", "+ untrained GNN embeddings"),
    "m7_own_nbr_gnnssl": ("E7", "+ neighbour means + DGI embeddings"),
    "m7r_own_nbr_gnnrand": ("E7r", "+ neighbour means + untrained GNN embeddings"),
    "m8_own_nbr_gnnsup": ("E8", "+ neighbour means + best GNN's out-of-fold score"),
    "m8s_own_gnnsup": ("E8s", "+ best GNN's out-of-fold score"),
    "g5_sage": ("E5", "GNN: heterogeneous GraphSAGE"),
    "g5g_rgcn": ("E5g", "GNN: R-GCN-style normalisation"),
    "g6_mlp": ("E6", "GNN without edges (MLP)"),
    "g5n3_sage_rw": ("N3", "GNN: GraphSAGE on the rewired graph"),
}


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p values, in the input order."""
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def split_config(config: dict, dataset: str, k: int, n: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["split"]["n_repeats"] = n
    cfg["split"]["repeat"] = k
    return with_dataset(cfg, dataset)


def setup_rows(config: dict, dataset: str, n: int) -> list[dict]:
    rows = []
    for name, (sid, what) in SETUPS.items():
        fs = f"pilot_{name}"
        found = []
        for k in range(n):
            cfg = split_config(config, dataset, k, n)
            path = (
                get_paths({**cfg, "feature_set": fs}).model_reports_dir / "model_a.json"
            )
            if path.is_file():
                found.append(json.loads(path.read_text(encoding="utf-8")))
        if not found:
            continue
        pr = [s["model_a"]["pr_auc"] for s in found]
        cv = [s["model_a"]["cv_pr_auc"] for s in found]
        if found[0].get("source") == "gnn":
            chosen = [
                f"h{s['gnn']['setting']['hidden']} d{s['gnn']['setting']['dropout']:g} "
                f"lr{s['gnn']['setting']['lr']:g}"
                for s in found
            ]
            seed_sd = [
                float(np.std([r["test_pr_auc"] for r in s["gnn"]["per_seed"]], ddof=1))
                for s in found
            ]
            model = "; ".join(chosen) + f" (per-seed SD ≤ {max(seed_sd):.3f})"
        else:
            model = "; ".join(f"{s['model_a']['model']} ({s['model_a']['selection']})"
                              for s in found)  # fmt: skip
        rows.append({
            "id": sid, "what": what, "feature_set": fs, "splits": len(found),
            "pr_mean": float(np.mean(pr)),
            "pr_sd": float(np.std(pr, ddof=1)) if len(pr) > 1 else float("nan"),
            "cv_mean": float(np.mean(cv)), "no_skill": float(np.mean(
                [s["test_positive_rate"] for s in found])), "model": model,
        })  # fmt: skip
    return rows


def gnn_choices(config: dict, dataset: str, n: int) -> list[str]:
    out = []
    for k in range(n):
        cfg = split_config(config, dataset, k, n)
        meta = get_paths(cfg).gnn_oof_dir / "meta.json"
        if meta.is_file():
            m = json.loads(meta.read_text(encoding="utf-8"))
            gains = ", ".join(
                f"{c['candidate'].removeprefix('pilot_')} {c['mean_gain']:+.4f} "
                f"(SE {c['se']:.4f})"
                for c in m["choice"]["comparisons"]
            )
            out.append(f"split {k}: {m['choice']['feature_set'].removeprefix('pilot_')} "
                       f"(gain over the first candidate: {gains})")  # fmt: skip
    return out


def label(name: str) -> str:
    return SETUPS.get(name.removeprefix("pilot_"), (name, ""))[0]


def contrast_table(frame: pd.DataFrame, n: int, family: bool) -> list[str]:
    splits = " | ".join(f"Δ {k}" for k in range(n))
    head = f"| Contrast | {splits} | Mean ± SD | Headroom | CV p |"
    head += " Holm p |" if family else ""
    head += " Verdict |"
    lines = [head, "|" + "---|" * (n + 5 + int(family))]
    holm_p = holm(frame["p"].tolist()) if family else [None] * len(frame)
    for r, hp in zip(frame.to_dict("records"), holm_p, strict=True):
        deltas = " | ".join(f"{r[f'test_delta_split{k}']:+.3f}" for k in range(n))
        headroom = r["test_delta_mean"] / (1 - r["test_pr_auc_a_mean"])
        line = (f"| {label(r['b'])} − {label(r['a'])} | {deltas} | "
                f"{r['test_delta_mean']:+.3f} ± {r['test_delta_sd']:.3f} | "
                f"{headroom:+.0%} | {r['p']:.2g} |")  # fmt: skip
        line += f" {hp:.2g} |" if family else ""
        lines.append(line + f" {r['verdict']} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "GNN_RESULTS.md")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    lines = [
        "# GNN round results (2026-10-31; generated by scripts/gnn_report.py)",
        "",
        "Plan: [`GNN_MODELS_PLAN.md`](GNN_MODELS_PLAN.md); pre-registration: "
        "RESEARCH_PLAN §11, fifth amendment. Every feature (as published work). "
        "Pilot df-analyze budget (LightGBM, logistic regression, CatBoost; 10 "
        "trials); GNN rungs: 8-setting grid on training CV, 5 seeds, cross-fit on "
        "the saved folds. Test PR-AUC of each setup's Model A (or GNN rung). "
        "Verdict rule of 2026-10-19: supported if Δ > 0 on every split, mean Δ > "
        "SD and the Nadeau–Bengio corrected p over the training CV folds < 0.05. "
        "Holm p: within the pre-registered primary family, a sensitivity check "
        "beside the verdict. Headroom: Δ / (1 − PR-AUC of A).",
    ]
    for dataset, n in DATASETS.items():
        lines += ["", f"## {DATASET_LABELS[dataset]}, {n} splits", ""]
        rows = setup_rows(config, dataset, n)
        lines += [
            "| Setup | What | Splits | Test PR-AUC | CV PR-AUC | No skill | "
            "Model A / GNN setting per split |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            sd = "" if np.isnan(r["pr_sd"]) else f" ± {r['pr_sd']:.3f}"
            lines.append(f"| {r['id']} | {r['what']} | {r['splits']} | "
                         f"{r['pr_mean']:.3f}{sd} | {r['cv_mean']:.3f} | "
                         f"{r['no_skill']:.3f} | {r['model']} |")  # fmt: skip
        choices = gnn_choices(config, dataset, n)
        if choices:
            lines += ["", "Best GNN for E8 (pre-registered one-SE rule, training "
                      "folds of each split):", *[f"- {c}" for c in choices]]  # fmt: skip
        base = split_config(config, dataset, 0, n)
        compare_dir = get_paths(base).outputs_dir / "compare"
        for kind, family in (("primary", True), ("secondary", False)):
            path = compare_dir / f"gnn_{dataset}_{kind}.csv"
            if not path.is_file():
                lines += ["", f"_{kind.capitalize()} contrasts: not summarised yet._"]
                continue
            frame = pd.read_csv(path)
            lines += ["", f"### {kind.capitalize()} contrasts (B − A as written)", ""]
            lines += contrast_table(frame, n, family)
    text = "\n".join(lines) + "\n"
    atomic_write_text(args.out, text)
    print(text)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
