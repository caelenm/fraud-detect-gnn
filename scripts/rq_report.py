"""One page of Amazon results for the RQ plan and the professor's feedback.

    uv run scripts/rq_report.py --dataset amazon

Reads what the pilot summaries already saved (outputs/<dataset>/compare/
*_summary.csv from scripts/repeat_summary.py, plus the Z1 and N2 training-CV
probes in outputs/<dataset>/probe/) and writes outputs/<dataset>/compare/
rq_results.md: every model setup's mean test PR-AUC over its splits, the
verdict of every pre-registered contrast, and the tables that answer the
professor's questions of 2026-10-21. Where a contrast has both a 3-split and
a 5-split verdict, the 5-split one is used (decided 2026-10-28). Full-budget
runs (no pilot_ prefix, split 0 only) are listed apart, descriptively.
Aggregates only; nothing is recomputed and nothing is chosen here.
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
from fraud_detect.dashboard import registry as reg
from fraud_detect.runstate import atomic_write_text

# The setups are defined in fraud_detect.dashboard.registry.
SETUP_LABELS = {
    setup_id: f"{setup_id}: {reg.BY_ID[setup_id].label}"
    for setup_id in ("B", "E1", "E2", "E3", "N1", "N2")
}
# Feature variant -> this report's column.
VARIANT_COLUMNS = {
    reg.ALL: "all 25",
    reg.NO36: "without #36",
    reg.NO19: "without f19",
    reg.ONLY36: "#36 four only",
    reg.TOP4: "top 4 (with f19)",
    reg.TOP4NO19: "top 4 without f19",
}
COLUMNS =["all 25", "without #36", "without f19", "top 4 without f19", "#36 four only",
           "top 4 (with f19)"]  # fmt: skip
ROW_ORDER = ["B", "E1", "E2 (2)", "E2 (4)", "E2", "E2 (16)", "E2 (raw 64)",
             "E3 (2)", "E3 (4)", "E3", "E3 (16)", "E3 (raw 64)", "N1", "N2"]  # fmt: skip
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
    "top4no19_summary": "Graph on the four features without f19 (2026-10-29)",
    "sizeraw_e3_summary": "Raw 64-d untrained embeddings, without f19 (2026-10-29)",
    "size8_5split": "Size 8, without f19, 5 splits (2026-10-29)",
    "size16_5split": "Size 16, without f19, 5 splits (2026-10-29)",
    "sizeraw_5split": "Raw 64, without f19, 5 splits (2026-10-29)",
    "size24_summary": "Sizes 2 and 4, without f19 (2026-10-29)",
    "full_no19_summary": "Full budget, without f19, split 0 only (2026-10-29 and -30)",
}
# Summaries of full-budget runs: one split, descriptive, kept out of the
# pilot tables.
FULL_SUMMARIES = {"full_no19_summary"}


def n_splits(frame: pd.DataFrame) -> int:
    return sum(c.startswith("test_delta_split") for c in frame.columns)


def parse(feature_set: str) -> tuple[str, str, str]:
    """(experiment ID, column, embedding size) of a pilot feature set."""
    fs = reg.parse_feature_set(feature_set)
    size = "" if fs.embed_size in (None, reg.DEFAULT_EMBED_SIZE) else fs.embed_size
    return fs.display_id, VARIANT_COLUMNS[fs.variant], size


def label(feature_set: str) -> str:
    setup, column, _ = parse(feature_set)
    full = "" if feature_set.startswith("pilot_") else ", full budget"
    return f"{setup} [{column}{full}]"


def scores(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean ± SD test PR-AUC of every pilot feature set seen, over the most
    splits any summary has for it (marked when that is not 3)."""
    rows: dict[str, tuple[int, float, float]] = {}
    for name, frame in frames.items():
        if name in FULL_SUMMARIES:
            continue
        n = n_splits(frame)
        for r in frame.itertuples():
            for side in ("a", "b"):
                fs = getattr(r, side)
                if n > rows.get(fs, (0,))[0]:
                    rows[fs] = (n, getattr(r, f"test_pr_auc_{side}_mean"),
                                getattr(r, f"test_pr_auc_{side}_sd"))  # fmt: skip
    table = pd.DataFrame(index=ROW_ORDER, columns=COLUMNS, dtype=object)
    for fs, (n, mean, sd) in rows.items():
        setup, column, _ = parse(fs)
        mark = "" if n == 3 else f" ({n} splits)"
        table.loc[setup, column] = f"{mean:.3f} ± {sd:.3f}{mark}"
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


def contrast_rows(frame: pd.DataFrame, descriptive: bool = False) -> pd.DataFrame:
    n = n_splits(frame)
    out = []
    for r in frame.itertuples():
        deltas = " / ".join(
            f"{getattr(r, f'test_delta_split{k}'):+.3f}" for k in range(n)
        )
        sd = "" if n == 1 else f" ± {r.test_delta_sd:.3f}"
        out.append({
            "Contrast": f"{label(r.b)} − {label(r.a)}",
            f"Δ splits {' / '.join(map(str, range(n)))}": deltas,
            "Mean ± SD": f"{r.test_delta_mean:+.3f}{sd}",
            "CV p": f"{r.p:.2g}",
            "Verdict": "descriptive (one split)" if descriptive else r.verdict,
        })  # fmt: skip
    return pd.DataFrame(out)


def find(frames: dict[str, pd.DataFrame], a: str, b: str) -> tuple[pd.Series, int] | None:
    """The pilot contrast B − A from the summary with the most splits."""
    best = None
    for name, frame in frames.items():
        if name in FULL_SUMMARIES:
            continue
        hit = frame[(frame["a"] == f"pilot_{a}") & (frame["b"] == f"pilot_{b}")]
        if len(hit) and (best is None or n_splits(frame) > best[1]):
            best = (hit.iloc[0], n_splits(frame))
    return best


def verdict_line(frames: dict[str, pd.DataFrame], a: str, b: str) -> str:
    found = find(frames, a, b)
    if found is None:
        return "no result yet"
    r, n = found
    return (
        f"{r.test_delta_mean:+.3f} ± {r.test_delta_sd:.3f} "
        f"({r.verdict}, p = {r.p:.2g}, {n} splits)"
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
        "seed 555), 3 grouped split repeats (5 for B, E2 and E3 at sizes 8, 16 and raw "
        "64), test PR-AUC of each setup's Model A (no-skill 0.095). Verdict rule, fixed "
        "before the runs: **supported** if Δ > 0 on every split, mean Δ > SD, and the "
        "Nadeau–Bengio corrected t-test over the grouped training CV folds (5 per "
        "split) gives p < 0.05. Otherwise a positive mean is **not distinguishable "
        "from zero** and a negative mean is labelled **worse**; p is two-sided, so a "
        '"worse" with p ≥ 0.05 is a lower mean, not a significant drop. Where a '
        "contrast has 3- and 5-split verdicts, the 5-split one is shown (decided "
        "2026-10-28). Source: docs/RESEARCH_PLAN.md §11 and the results database.",
        "",
        "## 1. Every model setup (test PR-AUC, mean ± SD over 3 splits unless marked)",
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
        f"| 2 | {v(frames, 'm1_own_no19', 'm5_own_gnnssl_no19d2')} | "
        f"{v(frames, 'm1_own_no19', 'm5r_own_gnnrand_no19d2')} |",
        f"| 4 | {v(frames, 'm1_own_no19', 'm5_own_gnnssl_no19d4')} | "
        f"{v(frames, 'm1_own_no19', 'm5r_own_gnnrand_no19d4')} |",
        f"| 8 (default) | {v(frames, 'm1_own_no19', 'm5_own_gnnssl_no19')} | "
        f"{v(frames, 'm1_own_no19', 'm5r_own_gnnrand_no19')} |",
        f"| 16 | {v(frames, 'm1_own_no19', 'm5_own_gnnssl_no19d16')} | "
        f"{v(frames, 'm1_own_no19', 'm5r_own_gnnrand_no19d16')} |",
        f"| raw 64 | {v(frames, 'm1_own_no19', 'm5_own_gnnssl_no19raw')} | "
        f"{v(frames, 'm1_own_no19', 'm5r_own_gnnrand_no19raw')} |",
        "",
        "Against the default size 8. By the size rule, fixed before the runs, a size "
        "below 8 is enough if its E − B is supported and 8 − d is not; a size above 8 "
        "is needed only if d − 8 is supported:",
        "",
        "| Contrast | E2 (DGI) | E3 (untrained) |",
        "|---|---|---|",
        f"| 8 − 2 | {v(frames, 'm5_own_gnnssl_no19d2', 'm5_own_gnnssl_no19')} | "
        f"{v(frames, 'm5r_own_gnnrand_no19d2', 'm5r_own_gnnrand_no19')} |",
        f"| 8 − 4 | {v(frames, 'm5_own_gnnssl_no19d4', 'm5_own_gnnssl_no19')} | "
        f"{v(frames, 'm5r_own_gnnrand_no19d4', 'm5r_own_gnnrand_no19')} |",
        f"| 16 − 8 | {v(frames, 'm5_own_gnnssl_no19', 'm5_own_gnnssl_no19d16')} | "
        f"{v(frames, 'm5r_own_gnnrand_no19', 'm5r_own_gnnrand_no19d16')} |",
        f"| raw 64 − 8 | {v(frames, 'm5_own_gnnssl_no19', 'm5_own_gnnssl_no19raw')} | "
        f"{v(frames, 'm5r_own_gnnrand_no19', 'm5r_own_gnnrand_no19raw')} |",
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
        f"| without f19 | 2 | "
        f"{v(frames, 'm5r_own_gnnrand_no19d2', 'm5_own_gnnssl_no19d2')} |",
        f"| without f19 | 4 | "
        f"{v(frames, 'm5r_own_gnnrand_no19d4', 'm5_own_gnnssl_no19d4')} |",
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
    full = [f for name, f in frames.items() if name in FULL_SUMMARIES]
    if full:
        seen: dict[str, float] = {}
        for frame in full:
            for r in frame.itertuples():
                seen[r.a], seen[r.b] = r.test_pr_auc_a_mean, r.test_pr_auc_b_mean
        lines += [
            "### Full budget (6 classifiers, 100 trials; split 0 only, descriptive)",
            "",
            "Test PR-AUC: "
            + "; ".join(f"{label(fs)} {m:.3f}" for fs, m in seen.items())
            + ". Contrasts are in section 4; one split gives no verdict.",
            "",
        ]
    lines += ["## 4. Every contrast", ""]
    for name, frame in frames.items():
        lines += [
            f"### {SUMMARIES[name]} (`{name}`)",
            "",
            markdown(contrast_rows(frame, descriptive=name in FULL_SUMMARIES)),
            "",
        ]
    out = compare_dir / "rq_results.md"
    atomic_write_text(out, "\n".join(lines))
    print(f"Wrote {out} from {len(frames)} summaries: {', '.join(frames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
