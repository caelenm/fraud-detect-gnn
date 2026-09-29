"""Static HTML report of Model A: parameters, results, and confidence cards.

The page is one self-contained file (no scripts, no external requests) that
opens in any browser. It is written to outputs/report/index.html and travels
with the output bundle. It contains complaint narratives, so like every other
output it must never be committed or posted publicly.

Confidence is read directly from the model's output: the probability it gives
the class it predicted, P(fraud) if it predicted fraud, else 1 - P(fraud). It
ranges from 0.5 (a coin flip) to 1.0 (certain). Only test complaints are shown.

Sections for Model B (the GNN) and inter-model agreement are placeholders
until those stages exist.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detect import columns as C

CSS_PATH = Path(__file__).with_name("report.css")

MODEL_LABELS = {
    "catboost": "CatBoost",
    "lgbm": "LightGBM",
    "rf": "Random forest (LightGBM)",
    "gandalf": "GANDALF",
    "knn": "k-nearest neighbours",
    "lr": "Logistic regression",
    "mlp": "MLP",
    "dummy": "Dummy baseline",
}
# df-analyze's long model names (tuning log) -> config names.
LONG_MODEL_NAMES = {
    "CatBoost Classifier": "catboost",
    "LightGBM Classifier": "lgbm",
    "LightGBM Random Forest Classifier": "rf",
    "GANDALF - Gated Adaptive Network": "gandalf",
    "K-Neighbours Classifier": "knn",
    "Logistic Regression": "lr",
    "Multilayer Perceptron": "mlp",
    "Dummy Classifier": "dummy",
}
FEATURE_SET_LABELS = {
    "none": "All features",
    "assoc": "Filter: association",
    "pred": "Filter: prediction",
    "embed_linear": "Embedded: linear",
    "embed_lgbm": "Embedded: LightGBM",
    "wrap": "Wrapper",
}


class WebReportError(RuntimeError):
    """Raised when the report's inputs are missing or do not line up."""


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def feature_set_key(selection: str, embed_selector: Any = "") -> str:
    """One name per feature set, e.g. ('embed', 'linear') -> 'embed_linear'."""
    selector = "" if pd.isna(embed_selector) else str(embed_selector)
    if selection == "embed" and selector:
        return f"embed_{selector}"
    return str(selection)


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def feature_set_label(key: str) -> str:
    return FEATURE_SET_LABELS.get(key, key)


def parse_params(params: Any) -> dict[str, Any]:
    """df-analyze stores tuned hyperparameters as JSON or a Python dict repr."""
    if isinstance(params, dict):
        return params
    if not isinstance(params, str):
        raise WebReportError(f"Unreadable tuned parameters: {params!r}")
    try:
        parsed = json.loads(params)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(params)
    if not isinstance(parsed, dict):
        raise WebReportError(f"Unreadable tuned parameters: {params!r}")
    return parsed


def find_entry(
    predictions: list[dict[str, Any]],
    model_a: dict[str, Any],
    model_names: dict[str, str],
) -> dict[str, Any]:
    """Model A's entry in df-analyze's prediction_results."""
    want = (
        model_a["model"],
        feature_set_key(model_a["selection"], model_a["embed_selector"]),
    )
    for entry in predictions:
        key = (
            model_names.get(entry["model_cls"], entry["model_cls"]),
            feature_set_key(entry["selection"], entry.get("embed_select_model") or ""),
        )
        if key == want:
            return entry
    raise WebReportError(
        f"Model A {want} has no entry in df-analyze's prediction results"
    )


def excerpt(text: Any, max_chars: int) -> str:
    """Whitespace-collapsed narrative, cut at a word boundary."""
    if not isinstance(text, str):
        return ""
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.;:") + " …"


def confidence_table(
    prob_fraud: np.ndarray, predicted: np.ndarray, actual: np.ndarray, meta: pd.DataFrame
) -> pd.DataFrame:
    """Test complaints with their prediction, label and output confidence.

    `meta` holds one row per test complaint in the same order as the
    predictions (Complaint ID, product, state, date, narrative)."""
    prob_fraud = np.asarray(prob_fraud, dtype=float)
    predicted = np.asarray(predicted, dtype=int)
    actual = np.asarray(actual, dtype=int)
    if not (len(prob_fraud) == len(predicted) == len(actual) == len(meta)):
        raise WebReportError(
            f"Row counts differ: {len(prob_fraud)} probabilities, {len(predicted)} "
            f"predictions, {len(actual)} labels, {len(meta)} complaints"
        )
    if ((prob_fraud < 0) | (prob_fraud > 1)).any():
        raise WebReportError("Fraud probabilities outside [0, 1]")
    table = meta.reset_index(drop=True).copy()
    table["prob_fraud"] = prob_fraud
    table["predicted"] = predicted
    table["actual"] = actual
    table["confidence"] = np.where(predicted == 1, prob_fraud, 1.0 - prob_fraud)
    table["correct"] = predicted == actual
    return table


def most_and_least_confident(
    table: pd.DataFrame, n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top `n` by confidence, highest first; bottom `n`, lowest first. Ties are
    broken by Complaint ID so the order is reproducible."""
    most = table.sort_values(
        ["confidence", C.COMPLAINT_ID], ascending=[False, True], kind="stable"
    ).head(n)
    least = table.sort_values(
        ["confidence", C.COMPLAINT_ID], ascending=[True, True], kind="stable"
    ).head(n)
    return most, least


@dataclass
class ReportInputs:
    summary: dict[str, Any]  # model_a.json
    metrics: pd.DataFrame  # model_a_metrics.csv, sorted by shared-CV PR-AUC
    params: dict[str, Any]  # Model A's tuned hyperparameters
    n_features: int  # features Model A used
    samples: pd.DataFrame  # confidence_table() for the test set
    header: dict[str, str]  # build, generated, git commit, df-analyze run
    run_config: list[tuple[str, str]]  # label, value
    budget: pd.DataFrame | None = None  # tuning_budget.csv
    n_samples: int = 25
    excerpt_chars: int = 600
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _num(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def prob_text(value: float) -> str:
    """A probability readable at the extremes, where 3 decimals would round
    many different values to 0.000 or 1.000."""
    v = float(value)
    if v < 1e-3:
        return f"{v:.1e}"
    if v > 1 - 1e-3:
        return f"1 − {1 - v:.1e}"
    return f"{v:.3f}"


def _int(value: Any) -> str:
    return f"{int(value):,}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def _label(value: int) -> str:
    return "fraud" if int(value) == 1 else "not fraud"


def _tile(label: str, value: str, sub: str, hero: bool = False, badge: str = "") -> str:
    badge_html = f' <span class="badge accent">{escape(badge)}</span>' if badge else ""
    return (
        f'<div class="panel tile{" hero" if hero else ""}">'
        f'<div class="label">{escape(label)}{badge_html}</div>'
        f'<div class="value">{escape(value)}</div>'
        f'<div class="sub">{escape(sub)}</div></div>'
    )


def _kv(rows: list[tuple[str, str]]) -> str:
    items = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows)
    return f'<dl class="kv">{items}</dl>'


def _budget_row(budget: pd.DataFrame | None, model: str, fs_key: str) -> pd.Series | None:
    if budget is None or budget.empty:
        return None
    short = budget["model"].map(LONG_MODEL_NAMES).fillna(budget["model"])
    match = budget[(short == model) & (budget["selection"] == fs_key)]
    return None if match.empty else match.iloc[0]


# How each budget label is shown. "early stop" is df-analyze's rule for ending the
# hyperparameter SEARCH (15 tries in a row found no better settings); it says
# nothing about the model failing to learn, so it is shown as "search settled".
STOP_LABELS = {
    "early stop": ("search settled", "good"),
    "all trials": ("all trials run", "good"),
    "time limit": ("time limit hit", "warn"),
}


def _stop_badge(stopped_by: str) -> str:
    text, kind = STOP_LABELS.get(stopped_by, (stopped_by, ""))
    return f'<span class="badge {kind}">{escape(text)}</span>'


def _header(inputs: ReportInputs) -> str:
    h = inputs.header
    meta = "".join(
        f"<span>{escape(k)}</span><b>{escape(v)}</b>"
        for k, v in (
            ("Build", h["build"]),
            ("Generated", h["generated"]),
            ("Git commit", h["git_commit"]),
            ("df-analyze run", h["run"]),
        )
    )
    return (
        '<header class="top"><div>'
        '<div class="eyebrow">CS555 · Fraud complaint detection</div>'
        "<h1>Model report</h1></div>"
        f'<div class="meta">{meta}</div></header>'
        '<div class="notice"><b>Contains complaint narratives.</b> Share within the '
        "group only; never commit this file or post it publicly.</div>"
    )


def _headline(inputs: ReportInputs) -> str:
    s, a = inputs.summary, inputs.summary["model_a"]
    fs = feature_set_label(feature_set_key(a["selection"], a["embed_selector"]))
    tiles = [
        _tile("Model A", model_label(a["model"]), f"feature set: {fs}", hero=True),
        _tile(
            "Test PR-AUC",
            _num(a["pr_auc"]),
            f"no-skill baseline: {_num(s['test_positive_rate'])}",
            badge="headline",
        ),
        _tile(
            "CV PR-AUC (train)",
            _num(a["cv_pr_auc"]),
            f"± {_num(a['cv_pr_auc_std'])} over {s['cv_folds']} folds",
        ),
        _tile("Test AUROC", _num(a["auroc"]), "threshold-free"),
        _tile(
            "Test set",
            _int(s["n_test"]),
            f"complaints · {_pct(s['test_positive_rate'])} fraud",
        ),
    ]
    return (
        "<section><h2>Headline</h2>"
        '<p class="lede">Model A is the df-analyze combination with the highest '
        "cross-validated PR-AUC on the training set. Test-set numbers are reported, "
        "never used for choosing.</p>"
        f'<div class="tiles">{"".join(tiles)}</div></section>'
    )


def _model_a_detail(inputs: ReportInputs) -> str:
    a = inputs.summary["model_a"]
    fs_key = feature_set_key(a["selection"], a["embed_selector"])
    b = _budget_row(inputs.budget, a["model"], fs_key)
    trials = (
        f"{_int(b['trials_completed'])} of {_int(b['trials_requested'])} · "
        f"{STOP_LABELS.get(b['stopped_by'], (b['stopped_by'], ''))[0]}"
        if b is not None
        else "not recorded"
    )
    config = _kv(
        [
            ("Classifier", model_label(a["model"])),
            ("Feature set", feature_set_label(fs_key)),
            ("Features used", _int(inputs.n_features)),
            ("Selected by", "shared-CV PR-AUC (train only)"),
            ("Tuning trials", trials),
            ("Tuning effect", _tuning_effect(a)),
        ]
    )
    param_rows = (
        "".join(
            f'<tr><td>{escape(str(k))}</td><td class="mono">{escape(_param(v))}</td></tr>'
            for k, v in sorted(inputs.params.items())
        )
        or '<tr><td colspan="2">No tuned hyperparameters recorded</td></tr>'
    )
    metric_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{_num(a[key])}</td></tr>"
        for name, key in (
            ("PR-AUC", "pr_auc"),
            ("AUROC", "auroc"),
            ("F1 (fraud class)", "f1"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("Balanced accuracy", "balanced_accuracy"),
            ("Accuracy", "accuracy"),
        )
    )
    cm = (
        '<div class="cm"><div></div><div class="h">Predicted not fraud</div>'
        '<div class="h">Predicted fraud</div>'
        '<div class="h">Actual<br>not fraud</div>'
        f'<div class="cell ok"><div class="n">{_int(a["tn"])}</div>'
        '<div class="t">true negative</div></div>'
        f'<div class="cell err"><div class="n">{_int(a["fp"])}</div>'
        '<div class="t">false positive</div></div>'
        '<div class="h">Actual<br>fraud</div>'
        f'<div class="cell err"><div class="n">{_int(a["fn"])}</div>'
        '<div class="t">false negative</div></div>'
        f'<div class="cell ok"><div class="n">{_int(a["tp"])}</div>'
        '<div class="t">true positive</div></div></div>'
    )
    return (
        "<section><h2>Model A in detail</h2>"
        '<p class="lede">The selected model\'s configuration and its behaviour on the '
        "frozen test set.</p>"
        '<div class="grid-2"><div class="panel panel-pad">'
        f'<h3 class="sub">Configuration</h3>{config}'
        '<h3 class="sub spaced">Tuned hyperparameters</h3>'
        '<div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Value</th>'
        f"</tr></thead><tbody>{param_rows}</tbody></table></div></div>"
        '<div class="panel panel-pad"><h3 class="sub">Test metrics</h3>'
        '<div class="table-scroll"><table><thead><tr><th>Metric</th><th>Value</th>'
        f"</tr></thead><tbody>{metric_rows}</tbody></table></div>"
        '<h3 class="sub spaced">Confusion matrix <span class="badge">test set, '
        "model's own decision rule</span></h3>"
        f"{cm}</div></div></section>"
    )


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _gain(value: Any) -> str:
    """Signed PR-AUC change from tuning, coloured by direction."""
    if _missing(value):
        return '<span class="col-info">—</span>'
    v = float(value)
    kind = "good" if v > 0 else "bad" if v < 0 else ""
    return f'<span class="gain {kind}">{v:+.3f}</span>'


def _tuning_effect(a: dict[str, Any]) -> str:
    default = a.get("cv_pr_auc_default")
    if _missing(default):
        return "untuned score not recorded"
    return (
        f"CV PR-AUC {float(default):.3f} untuned → {float(a['cv_pr_auc']):.3f} tuned "
        f"({float(a['cv_pr_auc']) - float(default):+.3f})"
    )


def _param(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _sample(rank: int, row: pd.Series, excerpt_chars: int) -> str:
    correct = bool(row["correct"])
    fill = max(0.0, min(1.0, (float(row["confidence"]) - 0.5) / 0.5)) * 100
    date = pd.Timestamp(row[C.DATE_RECEIVED]).strftime("%Y-%m-%d")
    text = excerpt(row[C.NARRATIVE], excerpt_chars)
    return (
        '<div class="sample"><div class="sample-top">'
        f'<span class="rank">#{rank}</span>'
        f'<span class="id">Complaint {escape(str(row[C.COMPLAINT_ID]))}</span>'
        '<span class="spacer"></span>'
        f'<span class="badge">predicted {_label(row["predicted"])}</span>'
        f'<span class="badge">actual {_label(row["actual"])}</span>'
        f'<span class="badge {"good" if correct else "bad"}">'
        f"{'correct' if correct else 'wrong'}</span></div>"
        '<div class="confbar"><span>confidence</span><div class="track">'
        f'<div class="fill" style="width:{fill:.1f}%"></div></div>'
        f'<span class="num">{prob_text(row["confidence"])}</span></div>'
        '<div class="facts">'
        f"<span>P(fraud) <b>{prob_text(row['prob_fraud'])}</b></span>"
        f"<span>Product <b>{escape(str(row[C.PRODUCT]))}</b></span>"
        f"<span>State <b>{escape(_or_dash(row[C.STATE]))}</b></span>"
        f"<span>Received <b>{date}</b></span></div>"
        f'<div class="narrative">{escape(text)}</div></div>'
    )


def _or_dash(value: Any) -> str:
    return (
        "—"
        if value is None or (isinstance(value, float) and np.isnan(value))
        else str(value)
    )


def _card(title: str, blurb: str, rows: pd.DataFrame, n: int, excerpt_chars: int) -> str:
    items = "".join(
        _sample(i, row, excerpt_chars) for i, (_, row) in enumerate(rows.iterrows(), 1)
    )
    n_correct = int(rows["correct"].sum())
    return (
        '<div class="panel conf-card"><div class="conf-head">'
        f'<h3>{escape(title)} <span class="badge">top {n}</span></h3>'
        f"<p>{escape(blurb)}</p></div>"
        f'<div class="conf-list">{items}</div>'
        f'<div class="conf-foot">Scroll for more · {n_correct} of {len(rows)} '
        "correct</div></div>"
    )


def _confidence(inputs: ReportInputs) -> str:
    most, least = most_and_least_confident(inputs.samples, inputs.n_samples)
    n = inputs.n_samples
    return (
        "<section><h2>Model A's most and least confident test predictions</h2>"
        '<p class="lede">Confidence is read directly from the model\'s output: the '
        "probability it assigns to the class it predicts, P(fraud) if it predicts "
        "fraud, else 1 − P(fraud). It ranges from 0.5 (a coin flip) to 1.0 (certain). "
        "Only test complaints are shown; correct and wrong predictions are both "
        "included, and confident mistakes are the most informative.</p>"
        '<div class="grid-2">'
        + _card(
            "Most confident",
            "Highest output confidence, most certain first.",
            most,
            n,
            inputs.excerpt_chars,
        )
        + _card(
            "Least confident",
            "Output confidence closest to 0.5: the model's most uncertain calls.",
            least,
            n,
            inputs.excerpt_chars,
        )
        + "</div></section>"
    )


def _all_models(inputs: ReportInputs) -> str:
    a = inputs.summary["model_a"]
    a_key = (a["model"], feature_set_key(a["selection"], a["embed_selector"]))
    rows = []
    for r in inputs.metrics.to_dict("records"):
        key = (r["model"], feature_set_key(r["selection"], r.get("embed_selector", "")))
        row_class = ' class="selected"' if key == a_key else ""
        rows.append(
            f"<tr{row_class}>"
            f"<td>{escape(model_label(key[0]))}</td>"
            f'<td class="l">{escape(feature_set_label(key[1]))}</td>'
            f"<td>{_num(r['cv_pr_auc'])}</td><td>{_num(r['cv_pr_auc_std'])}</td>"
            f"<td>{_num(r.get('cv_pr_auc_default'))}</td>"
            f"<td>{_gain(r.get('cv_tuning_gain'))}</td>"
            f'<td class="col-info">{_num(r["tuning_score"])}</td>'
            f"<td>{_num(r['pr_auc'])}</td><td>{_num(r['auroc'])}</td>"
            f"<td>{_num(r['f1'])}</td><td>{_num(r['precision'])}</td>"
            f"<td>{_num(r['recall'])}</td><td>{_num(r['balanced_accuracy'])}</td></tr>"
        )
    tuning_metric = escape(str(a.get("tuning_metric", "")))
    return (
        "<section><h2>Every tuned model</h2>"
        '<p class="lede">Each classifier was tuned on each of df-analyze\'s feature '
        "sets. Rows are sorted by cross-validated PR-AUC on the training set, the only "
        "score comparable across models. The highlighted row is Model A. "
        "<b>Untuned</b> is the same model and feature set with df-analyze's default "
        "hyperparameters on the same folds, so <b>tuning gain</b> is what the "
        "hyperparameter search added. Tuning optimises balanced accuracy, not PR-AUC, "
        "so the gain can be negative.</p>"
        '<div class="panel table-scroll"><table><thead>'
        '<tr><th class="l" colspan="2"></th>'
        '<th class="group" colspan="4">Training set · shared CV PR-AUC</th>'
        '<th class="group col-info">df-analyze</th>'
        '<th class="group" colspan="6">Test set · for information only</th></tr>'
        '<tr><th class="l">Model</th><th class="l">Feature set</th>'
        "<th>tuned</th><th>± std</th><th>untuned</th><th>tuning gain</th>"
        '<th class="col-info">tuning score *</th>'
        "<th>PR-AUC</th><th>AUROC</th><th>F1</th><th>Precision</th><th>Recall</th>"
        "<th>Bal. acc.</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        f'<p class="lede small" style="margin-top:8px">* df-analyze\'s own tuning score '
        f"({tuning_metric}). Shown for reference; not comparable across models "
        "(GANDALF is scored on one validation split, the others over 5 folds), so "
        "never used to choose.</p></section>"
    )


def _budget(inputs: ReportInputs) -> str:
    if inputs.budget is None or inputs.budget.empty:
        return ""
    rows = "".join(
        "<tr>"
        f"<td>{escape(model_label(LONG_MODEL_NAMES.get(r['model'], r['model'])))}</td>"
        f'<td class="l">{escape(feature_set_label(str(r["selection"])))}</td>'
        f"<td>{_int(r['trials_completed'])} / {_int(r['trials_requested'])}</td>"
        f"<td>{float(r['elapsed_s']) / 60:.1f} min</td>"
        f"<td>{float(r['time_limit_s']) / 60:.0f} min</td>"
        f'<td class="l">{_stop_badge(str(r["stopped_by"]))}</td></tr>'
        for r in inputs.budget.dropna(subset=["trials_completed"]).to_dict("records")
    )
    return (
        "<section><h2>Tuning budget</h2>"
        '<p class="lede">Tuning is a search: df-analyze tries up to 100 hyperparameter '
        "settings per model and keeps the best. The search ends when all tries have "
        "run, when the model's time limit is hit, or once 50 tries have run and the "
        "last 15 found no better settings (<b>search settled</b>). That is the search "
        "ending, not the model failing to learn; see <b>tuning gain</b> above for what "
        "tuning added. Runs cut off by the time limit are flagged.</p>"
        '<div class="panel table-scroll"><table><thead><tr>'
        '<th class="l">Model</th><th class="l">Feature set</th><th>Trials</th>'
        '<th>Time used</th><th>Time limit</th><th class="l">Stopped by</th></tr>'
        f"</thead><tbody>{rows}</tbody></table></div></section>"
    )


def _run_config(inputs: ReportInputs) -> str:
    half = (len(inputs.run_config) + 1) // 2
    left, right = inputs.run_config[:half], inputs.run_config[half:]
    return (
        "<section><h2>Run configuration</h2>"
        '<p class="lede">Everything needed to reproduce this build.</p>'
        f'<div class="grid-2"><div class="panel panel-pad">{_kv(left)}</div>'
        f'<div class="panel panel-pad">{_kv(right)}</div></div></section>'
    )


def _pending_panel(title: str, body: str) -> str:
    return (
        '<div class="panel pending"><div class="panel-pad">'
        f'<h3>{title} <span class="badge">pending</span></h3>{body}</div></div>'
    )


PENDING = (
    "<section><h2>Model A vs. Model B</h2>"
    '<p class="lede">Reserved for the graph neural network. These panels fill in '
    "once the GNN stages exist.</p>"
    '<div class="grid-2">'
    + _pending_panel(
        "Head-to-head",
        "Model A, the heterogeneous GraphSAGE (Model B) and its graph-free control on "
        "the same test complaints."
        "<ul><li>PR-AUC, AUROC, F1, recall (mean ± std over seeds for the GNN)</li>"
        "<li>Effect of the graph: GNN minus graph-free control</li></ul>",
    )
    + _pending_panel(
        "Inter-model agreement",
        "Where Model A and Model B agree and disagree on the same complaints."
        "<ul><li>Agreement rate and Cohen's kappa on predicted labels</li>"
        "<li>Correlation of their fraud probabilities</li>"
        "<li>Complaints each model gets right that the other gets wrong</li>"
        "<li>Confidence cards for both models side by side</li></ul>",
    )
    + _pending_panel(
        "Feature-group importance",
        "Permutation importance by group (text, company, product, region) for both "
        "models.",
    )
    + _pending_panel(
        "Edge-type ablation",
        "Drop in GNN PR-AUC when company, product, region or similarity edges are "
        "removed.",
    )
    + "</div></section>"
)

HOW_TO_READ = """
<section><h2>How to read this report</h2>
<ul class="notes panel">
<li><b>What is predicted:</b> whether the consumer filed the complaint under a
fraud, scam, identity-theft or unauthorized-transaction category (1) or any other
issue (0), per <code>configs/categories.yaml</code>; categories too ambiguous to call
are excluded from the data. Every category's treatment is listed in
<code>docs/LABEL_RULE.md</code>. It is the consumer's chosen category, not verified
fraud. The model never sees the Issue or Sub-issue fields.</li>
<li><b>PR-AUC</b> is the headline metric because fraud is the minority class. A
model that guesses scores the test fraud rate.</li>
<li><b>Model A is chosen on the training set only</b>: every tuned combination is
refit on the same folds and scored the same way. Test columns never influence the
choice.</li>
<li><b>df-analyze's tuning score</b> is shown for reference. It is computed
differently for different models, so it cannot rank them.</li>
<li><b>Confidence</b> is the probability the model gives its own prediction. High
confidence is not the same as being right.</li>
<li>df-analyze's <code>5-fold</code> results table refits models on test-set folds and is
deliberately excluded.</li>
<li>Single split and single seed for Model A: no variance estimate across splits.</li>
</ul></section>
"""


def render(inputs: ReportInputs) -> str:
    """The complete, self-contained HTML page."""
    css = CSS_PATH.read_text(encoding="utf-8")
    body = "".join(
        [
            _header(inputs),
            _headline(inputs),
            _model_a_detail(inputs),
            _confidence(inputs),
            _all_models(inputs),
            _budget(inputs),
            _run_config(inputs),
            PENDING,
            HOW_TO_READ,
            "<footer>Generated by <code>uv run run.py --only web_report</code> · build "
            f"{escape(inputs.header['build'])} · Contains complaint narratives: share "
            "with the group only, never commit or post publicly.</footer>",
        ]
    )
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Fraud Detection Report</title><style>{css}</style></head>"
        f'<body><div class="wrap">{body}</div></body></html>\n'
    )
