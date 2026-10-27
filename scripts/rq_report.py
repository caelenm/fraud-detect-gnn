"""One page of Amazon results for the RQ plan and the professor's feedback.

    uv run scripts/rq_report.py --dataset amazon

Reads what the pilot summaries already saved (outputs/<dataset>/compare/
*_summary.csv from scripts/repeat_summary.py, plus the Z1 and N2 training-CV
probes in outputs/<dataset>/probe/) and writes outputs/<dataset>/compare/
rq_results.md: every model setup's mean test PR-AUC over the 3 splits, the
verdict of every pre-registered contrast, and the tables that answer the
professor's questions of 2026-10-21. Aggregates only; nothing is recomputed
and nothing is chosen here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from fraud_detect.config import (
    DEFAULT_CONFIG,
    ConfigError,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.runstate import atomic_write_text

# Feature-set name prefix -> experiment ID (longest prefixes first).
SETUPS = [
    ("m5r_own_gnnrand", "E3"),
    ("m5_own_gnnssl", "E2"),
    ("m2_own_nbr", "E1"),
    ("n1_own_deg", "N1"),
    ("n2_own_nbrrw", "N2"),
    ("m1_own", "B"),
]
SETUP_LABELS = {
    "B": "B: df-analyze on own features",
    "E1": "E1: + neighbour means",
    "E2": "E2: + DGI embeddings",
    "E3": "E3: + untrained GNN embeddings",
    "N1": "N1: + degree only",
    "N2": "N2: + neighbour means on a rewired graph",
}
# Variant suffix -> (column, embedding size).
VARIANTS = {
    "": ("all 25", ""),
    "no36": ("without #36", ""),
    "no19": ("without f19", ""),
    "no19d16": ("without f19", "16"),
    "no19raw": ("without f19", "raw 64"),
    "only36": ("#36 four only", ""),
    "top4": ("top 4 (with f19)", ""),
    "top4no19": ("top 4 without f19", ""),
}
COLUMNS = ["all 25", "without #36", "without f19", "top 4 without f19", "#36 four only",
           "top 4 (with f19)"]  # fmt: skip
ROW_ORDER = ["B", "E1", "E2", "E2 (16)", "E2 (raw 64)", "E3", "E3 (16)", "E3 (raw 64)",
             "N1", "N2"]  # fmt: skip
SUMMARIES = {
    "repeat_summary": "Phase 2 ladder (2026-10-20)",
    "gnnrand_summary": "Trained vs untrained embeddings (2026-10-23)",
    "only36_summary": "The #36 four only (2026-10-23)",
    "top4_summary": "Top 4 with f19 (2026-10-23)",
    "no19_summary": "Without f19 (2026-10-27)",
    "n2_summary": "Null graph N2, without f19 (2026-10-27)",
    "f5_summary": "Four features without f19 (2026-10-27)",
    "e3_summary": "Untrained embeddings without f19 (2026-10-27)",
    "size16_summary": "Embedding size 16, without f19 (2026-10-27)",
    "sizeraw_summary": "Raw 64-d embeddings, without f19 (2026-10-27)",
    "top4no19_summary": "Graph on the four features without f19 (2026-10-27)",
}


def parse(feature_set: str) -> tuple[str, str, str]:
    """(experiment ID, column, embedding size) of a pilot feature set."""
    name = feature_set.removeprefix("pilot_")
    for prefix, setup in SETUPS:
        if name == prefix or name.startswith(prefix + "_"):
            column, size = VARIANTS[name[len(prefix) + 1 :]]
            if setup in ("E2", "E3") and size:
                return f"{setup} ({size})", column, size
            return setup, column, size
    raise ValueError(f"Unknown feature set {feature_set!r}")


def label(feature_set: str) -> str:
    setup, column, _ = parse(feature_set)
    return f"{setup} [{column}]"


def scores(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean ± SD test PR-AUC over the splits of every feature set seen."""
    rows = {}
    for frame in frames.values():
        for r in frame.itertuples():
            for side in ("a", "b"):
                fs = getattr(r, side)
                rows[fs] = (getattr(r, f"test_pr_auc_{side}_mean"),
                            getattr(r, f"test_pr_auc_{side}_sd"))  # fmt: skip
    table = pd.DataFrame(index=ROW_ORDER, columns=COLUMNS, dtype=object)
    for fs, (mean, sd) in rows.items():
        setup, column, _ = parse(fs)
        table.loc[setup, column] = f"{mean:.3f} ± {sd:.3f}"
    return table.dropna(how="all").dropna(axis=1, how="all").fillna("")


def markdown(frame: pd.DataFrame, index: str | None = None) -> str:
    data = frame.reset_index() if index else frame
    if index:
        data = data.rename(columns={"index": index})
    head = "| " + " | ".join(map(str, data.columns)) + " |"
    rule = "|" + "---|" * len(data.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in data.itertuples(index=False)]  # fmt: skip
    return "\n".join([head, rule, *body])


def contrast_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in frame.itertuples():
        deltas = " / ".join(
            f"{getattr(r, f'test_delta_split{k}'):+.3f}" for k in range(3)
        )
        out.append({
            "Contrast": f"{label(r.b)} − {label(r.a)}",
            "Δ splits 0 / 1 / 2": deltas,
            "Mean ± SD": f"{r.test_delta_mean:+.3f} ± {r.test_delta_sd:.3f}",
            "CV p": f"{r.p:.2g}",
            "Verdict": r.verdict,
        })  # fmt: skip
    return pd.DataFrame(out)


def find(frames: dict[str, pd.DataFrame], a: str, b: str) -> pd.Series | None:
    for frame in frames.values():
        hit = frame[(frame["a"] == f"pilot_{a}") & (frame["b"] == f"pilot_{b}")]
        if len(hit):
            return hit.iloc[0]
    return None


def verdict_line(frames: dict[str, pd.DataFrame], a: str, b: str) -> str:
    r = find(frames, a, b)
    if r is None:
        return "no 3-split result"
    return (
        f"{r.test_delta_mean:+.3f} ± {r.test_delta_sd:.3f} ({r.verdict}, p = {r.p:.2g})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default="amazon")
    args = parser.parse_args(argv)
    try:
        config = with_dataset(load_config(args.config), args.dataset)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    p = get_paths(config)
    compare_dir = p.outputs_dir / "compare"
    frames = {
        name: pd.read_csv(compare_dir / f"{name}.csv")
        for name in SUMMARIES
        if (compare_dir / f"{name}.csv").is_file()
    }
    if not frames:
        print(f"error: no summaries in {compare_dir}", file=sys.stderr)
        return 1
    v = verdict_line
    lines = [
        f"# Amazon results: model setups and the professor's questions ({args.dataset})",
        "",
        "Pilot budget (LightGBM, logistic regression, CatBoost; 10 tuning trials; "
        "seed 555), 3 grouped split repeats, test PR-AUC of each setup's Model A "
        "(no-skill 0.095). Verdict rule, fixed before the runs: **supported** if Δ > 0 "
        "on all 3 splits, mean Δ > SD, and the Nadeau–Bengio corrected t-test over the "
        "15 grouped training CV folds gives p < 0.05. Otherwise a positive mean is **not "
        "distinguishable from zero** and a negative mean is labelled **worse**; p is "
        'two-sided, so a "worse" with p ≥ 0.05 is a lower mean, not a significant drop. '
        "Source: docs/RESEARCH_PLAN.md §11 and the results database.",
        "",
        "## 1. Every model setup (test PR-AUC, mean ± SD over 3 splits)",
        "",
        markdown(scores(frames), index="Setup"),
        "",
        "Setups: "
        + "; ".join(f"**{k}** {t.split(': ', 1)[1]}" for k, t in SETUP_LABELS.items())
        + ". E2/E3 use 8 PCA components per relation unless a size is given.",
        "",
        "## 2. The professor's questions (2026-10-21)",
        "",
        "### Embedding size (and DeepTune)",
        "",
    ]
    z1 = p.probe_dir / "z1_size_sweep_no19.csv"
    if z1.is_file():
        z = pd.read_csv(z1)
        z = z[z["block"] != "own"].copy()
        z["k"] = z["k"].astype(str)
        cv = z.groupby(["block", "k"], sort=False)["cv_pr_auc_mean"].mean().unstack(0)
        ranks = z.groupby("block")[
            ["eff_rank_upu", "eff_rank_usu", "eff_rank_uvu"]
        ].mean()
        cv.index.name = "d per relation"
        lines += [
            "Training CV (Z1, split 0, without f19; gnnrand averaged over 5 "
            "initialisations), PR-AUC by embedding size d:",
            "",
            markdown(cv.map(lambda x: f"{x:.3f}"), index="d per relation"),
            "",
            "Effective rank of the 64-d embedding (U-P-U / U-S-U / U-V-U): "
            + "; ".join(f"{b} {r.eff_rank_upu:.0f} / {r.eff_rank_usu:.0f} / "
                        f"{r.eff_rank_uvu:.0f}" for b, r in ranks.iterrows())
            + ".",
            "",
        ]  # fmt: skip
    lines += [
        "Test, without f19 (Δ vs B):",
        "",
        "| Size per relation | E2 (DGI) − B | E3 (untrained) − B |",
        "|---|---|---|",
        f"| 8 (default) | {v(frames, 'm1_own_no19', 'm5_own_gnnssl_no19')} | "
        f"{v(frames, 'm1_own_no19', 'm5r_own_gnnrand_no19')} |",
        f"| 16 | {v(frames, 'm1_own_no19', 'm5_own_gnnssl_no19d16')} | "
        f"{v(frames, 'm1_own_no19', 'm5r_own_gnnrand_no19d16')} |",
        f"| raw 64 | {v(frames, 'm1_own_no19', 'm5_own_gnnssl_no19raw')} | "
        f"{v(frames, 'm1_own_no19', 'm5r_own_gnnrand_no19raw')} |",
        "",
        "DeepTune has no label-free reducer for tabular or graph embeddings (its "
        "embeddings come from label-trained models; its autoencoder is image-only), "
        "so it was not used; an autoencoder in place of PCA (E4) is optional.",
        "",
        "### Does training the GNN help? (E2 − E3: same encoder, DGI-trained vs random)",
        "",
        "| Features | Size | E2 − E3 |",
        "|---|---|---|",
        f"| all 25 | 8 | {v(frames, 'm5r_own_gnnrand', 'm5_own_gnnssl')} |",
        f"| without #36 | 8 | "
        f"{v(frames, 'm5r_own_gnnrand_no36', 'm5_own_gnnssl_no36')} |",
        f"| without f19 | 8 | "
        f"{v(frames, 'm5r_own_gnnrand_no19', 'm5_own_gnnssl_no19')} |",
        f"| without f19 | 16 | "
        f"{v(frames, 'm5r_own_gnnrand_no19d16', 'm5_own_gnnssl_no19d16')} |",
        f"| without f19 | raw 64 | "
        f"{v(frames, 'm5r_own_gnnrand_no19raw', 'm5_own_gnnssl_no19raw')} |",
        "",
        "GraphSAGE trained end to end on the labels (Models 3–4) is not built yet.",
        "",
        "### Just the 4 important features",
        "",
        "| Four features | Four vs more (B) | + neighbour means (E1 − B) "
        "| + DGI (E2 − B) |",
        "|---|---|---|---|",
        f"| #36 four (f13, f15, f18, f19) | vs all 25: "
        f"{v(frames, 'm1_own', 'm1_own_only36')}"
        f" | {v(frames, 'm1_own_only36', 'm2_own_nbr_only36')} | "
        f"{v(frames, 'm1_own_only36', 'm5_own_gnnssl_only36')} |",
        f"| top 4 with f19 (f00, f18, f19, f23) | vs all 25: "
        f"{v(frames, 'm1_own', 'm1_own_top4')} | "
        f"{v(frames, 'm1_own_top4', 'm2_own_nbr_top4')} | "
        f"{v(frames, 'm1_own_top4', 'm5_own_gnnssl_top4')} |",
        f"| top 4 without f19 (per split) | vs the 24 without f19: "
        f"{v(frames, 'm1_own_no19', 'm1_own_top4no19')} | "
        f"{v(frames, 'm1_own_top4no19', 'm2_own_nbr_top4no19')} | "
        f"{v(frames, 'm1_own_top4no19', 'm5_own_gnnssl_top4no19')} |",
        "",
        "### More datasets",
        "",
        "Decided 2026-10-26: Amazon is the main dataset and YelpChi the validation "
        "dataset; no other dataset is in scope.",
        "",
        "## 3. The research questions",
        "",
        "### RQ1: how much does the f19 leak drive Amazon results?",
        "",
        f"- F4, dropping f19 (B without f19 − B all 25): "
        f"{v(frames, 'm1_own', 'm1_own_no19')}",
        "- Graph gain by feature variant (E1 − B / E2 − B):",
        f"  - all 25: {v(frames, 'm1_own', 'm2_own_nbr')} / "
        f"{v(frames, 'm1_own', 'm5_own_gnnssl')}",
        f"  - without f19: {v(frames, 'm1_own_no19', 'm2_own_nbr_no19')} / "
        f"{v(frames, 'm1_own_no19', 'm5_own_gnnssl_no19')}",
        f"  - without #36: {v(frames, 'm1_own_no36', 'm2_own_nbr_no36')} / "
        f"{v(frames, 'm1_own_no36', 'm5_own_gnnssl_no36')}",
        "",
        "### RQ2: is the graph gain real structure? (without f19)",
        "",
        f"- Degree (N1 − B): {v(frames, 'm1_own_no19', 'n1_own_deg_no19')}",
        f"- Neighbour columns on a random graph (N2 − N1): "
        f"{v(frames, 'n1_own_deg_no19', 'n2_own_nbrrw_no19')}",
        f"- Real structure (E1 − N2): "
        f"{v(frames, 'n2_own_nbrrw_no19', 'm2_own_nbr_no19')}",
        f"- Beyond degree (E1 − N1, secondary): "
        f"{v(frames, 'n1_own_deg_no19', 'm2_own_nbr_no19')}",
        "",
    ]
    n2 = p.probe_dir / "null_graph_probe_no19.csv"
    if n2.is_file():
        g = pd.read_csv(n2).groupby("block", sort=False)["cv_pr_auc_mean"]
        lines += [
            "Training-CV probe (split 0): " + "; ".join(
                f"{b} {m:.3f}" for b, m in g.mean().items()) + ".",
            "",
        ]  # fmt: skip
    lines += ["## 4. Every contrast", ""]
    for name, frame in frames.items():
        lines += [
            f"### {SUMMARIES[name]} (`{name}`)",
            "",
            markdown(contrast_rows(frame)),
            "",
        ]
    out = compare_dir / "rq_results.md"
    atomic_write_text(out, "\n".join(lines))
    print(f"Wrote {out} from {len(frames)} summaries: {', '.join(frames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
