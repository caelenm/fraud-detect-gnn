"""
=====================================================================
SYNTHETIC TEST DATA. NOT REAL CFPB DATA.
=====================================================================

Every row used by the test suite is built by a function in this module and
nowhere else. All values are invented and deliberately look fake (company
names like "SYNTHETIC-COMPANY-A", narratives that say "synthetic"). Only the
category strings (e.g. "Fraud or scam") mirror CFPB category names, because
the label rule matches on them.

Nothing here is written to data/ or outputs/. Tests that need files write
them to pytest's temporary directory (`tmp_path`), which is deleted
automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detect import columns as C

FAKE_COMPANIES = ["SYNTHETIC-COMPANY-A", "SYNTHETIC-COMPANY-B", "SYNTHETIC-COMPANY-C"]
FAKE_STATES = ["ZZ", "YY", "XX"]
RESPONSES = ["Closed with explanation", "Closed with monetary relief"]
TAG_VALUES = [None, "Older American", "Servicemember", "Older American, Servicemember"]

# (product, issue, sub_issue, expected label under SYNTHETIC_CATEGORIES)
CATEGORY_EXAMPLES: list[tuple[str, str, str | None, int]] = [
    ("Money transfer, virtual currency, or money service", "Fraud or scam",
     "Fraud or scam", 1),
    ("Credit card or prepaid card", "Problem with a purchase shown on your statement",
     "Card was charged for something you did not purchase with the card", 1),
    ("Credit card or prepaid card", "Problem with a purchase shown on your statement",
     "Credit card company isn't resolving a dispute about a purchase on your statement",
     0),
    ("Debt collection", "Attempts to collect debt not owed",
     "Debt was result of identity theft", 1),
    ("Debt collection", "Attempts to collect debt not owed", "Debt is not yours", 0),
    ("Checking or savings account", "Managing an account", None, 0),
    ("Credit reporting", "Identity theft protection or other monitoring services",
     None, 0),
]  # fmt: skip

SYNTHETIC_CATEGORIES = {
    "confirmed": True,
    "target_products": [
        "Money transfer, virtual currency, or money service",
        "Credit card or prepaid card",
        "Debt collection",
        "Checking or savings account",
    ],
    "positive": [
        {"issue": "Fraud or scam"},
        {
            "product": "Credit card or prepaid card",
            "issue": "Problem with a purchase shown on your statement",
            "sub_issue": "Card was charged for something you did not purchase "
            "with the card",
        },
        {
            "product": "Debt collection",
            "issue": "Attempts to collect debt not owed",
            "sub_issue": "Debt was result of identity theft",
        },
    ],
    "reviewed_negative": [
        {"issue": "Identity theft protection or other monitoring services"},
    ],
}


def fake_narrative(i: int) -> str:
    """Distinct synthetic narratives: each uses its own made-up words."""
    rng = np.random.default_rng(i)
    words = ["".join(rng.choice(list("abcdefghij"), size=6)) for _ in range(40)]
    return f"synthetic narrative {i} " + " ".join(words)


def complaints_with_categories() -> pd.DataFrame:
    """One loaded complaint per CATEGORY_EXAMPLES row."""
    rows = []
    for i, (product, issue, sub_issue, _) in enumerate(CATEGORY_EXAMPLES):
        rows.append(
            {
                C.COMPLAINT_ID: 1000 + i,
                C.DATE_RECEIVED: pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                C.PRODUCT: product,
                C.SUB_PRODUCT: "Synthetic sub-product",
                C.ISSUE: issue,
                C.SUB_ISSUE: sub_issue,
                C.NARRATIVE: fake_narrative(i),
                C.COMPANY: FAKE_COMPANIES[i % 3],
                C.STATE: FAKE_STATES[i % 3],
                C.TAGS: None,
                C.SUBMITTED_VIA: "Web",
                C.COMPANY_RESPONSE: RESPONSES[i % 2],
                C.TIMELY_RESPONSE: "Yes",
            }
        )
    return pd.DataFrame(rows)


def labeled_sample(n: int = 200, positive_rate: float = 0.2, seed: int = 0):
    """A labelled sample shaped like the output of the `sample` stage.

    Features are random, so no feature should predict the label.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < positive_rate).astype("int8")
    labels[:2] = [0, 1]  # guarantee both classes
    return pd.DataFrame(
        {
            C.COMPLAINT_ID: np.arange(5000, 5000 + n),
            C.DATE_RECEIVED: pd.Timestamp("2019-01-01")
            + pd.to_timedelta(rng.integers(0, 1500, n), unit="D"),
            C.PRODUCT: rng.choice(["Synthetic product 1", "Synthetic product 2"], n),
            C.SUB_PRODUCT: rng.choice(["Synthetic sub 1", "Synthetic sub 2", None], n),
            C.NARRATIVE: [fake_narrative(i) for i in range(n)],
            C.COMPANY: rng.choice(FAKE_COMPANIES, n),
            C.STATE: rng.choice(FAKE_STATES, n),
            C.TAGS: rng.choice(TAG_VALUES, n),
            C.SUBMITTED_VIA: "Web",
            C.COMPANY_RESPONSE: rng.choice([*RESPONSES, None], n),
            C.TIMELY_RESPONSE: rng.choice(["Yes", "No"], n),
            C.LABEL: labels,
        }
    )


def raw_csv_rows() -> list[dict[str, str]]:
    """Rows as they would appear in a raw CFPB CSV export (all strings)."""
    base = {
        "Date received": "2019-06-01",
        "Product": "Debt collection",
        "Sub-product": "Synthetic sub-product",
        "Issue": "Attempts to collect debt not owed",
        "Sub-issue": "Debt is not yours",
        "Consumer complaint narrative": "synthetic narrative text",
        "Company public response": "",
        "Company": "SYNTHETIC-COMPANY-A",
        "State": "zz",
        "ZIP code": "000XX",
        "Tags": "",
        "Consumer consent provided?": "Consent provided",
        "Submitted via": "Web",
        "Date sent to company": "2019-06-02",
        "Company response to consumer": "Closed with explanation",
        "Timely response?": "Yes",
        "Consumer disputed?": "N/A",
        "Complaint ID": "1",
    }
    return [
        base,
        base | {"Complaint ID": "2", "Consumer complaint narrative": ""},  # no narrative
        base | {"Complaint ID": "3", "Date received": "2017-01-01"},  # before range
        base | {"Complaint ID": "4", "Consumer complaint narrative": "  another  "},
    ]


# (config name, df-analyze estimator class) for fake df-analyze results.
FAKE_DFA_MODELS = [("dummy", "DummyClassifier"), ("lgbm", "LightGBMClassifier"),
                   ("catboost", "CatBoostClassifier")]  # fmt: skip


def fake_df_analyze_results(
    y_test: np.ndarray, cv_scores: dict[tuple[str, str], float], seed: int = 0
) -> tuple[list[dict], pd.DataFrame]:
    """Invented df-analyze outputs: `prediction_results` entries and a
    `tuned_models` table, one per (model, selection) key in `cv_scores`.
    Probabilities are random noise plus a model-specific signal; they describe
    no real model."""
    rng = np.random.default_rng(seed)
    classes = dict(FAKE_DFA_MODELS)
    entries, tuned = [], []
    for i, ((model, selection), score) in enumerate(cv_scores.items()):
        signal = 0.0 if model == "dummy" else 0.15 * (i + 1)
        p1 = np.clip(rng.uniform(0, 0.6, len(y_test)) + signal * y_test, 0, 1)
        entries.append(
            {
                "model_cls": classes[model],
                "selection": selection,
                "embed_select_model": "linear" if selection == "embed" else None,
                "metric": "Accuracy",
                "score": score,
                "probs_test": np.column_stack([1 - p1, p1]).tolist(),
                "preds_test": (p1 > 0.5).astype(int).tolist(),
            }
        )
        tuned.append(
            {
                "selection": selection,
                "embed_selector": "linear" if selection == "embed" else "none",
                "model": model,
                "params": "{}",
                "metric": "acc",
                "score": score,
                "test_idx": 0,
            }
        )
    return entries, pd.DataFrame(tuned)


def fake_test_meta(n: int) -> pd.DataFrame:
    """Invented test-complaint details for the web report, in test order."""
    return pd.DataFrame(
        {
            C.COMPLAINT_ID: [f"SYN-{i:04d}" for i in range(n)],
            C.DATE_RECEIVED: pd.date_range("2020-01-01", periods=n, freq="D"),
            C.PRODUCT: ["Debt collection"] * n,
            C.STATE: [FAKE_STATES[i % len(FAKE_STATES)] for i in range(n)],
            C.NARRATIVE: [fake_narrative(i) for i in range(n)],
        }
    )


def fake_shared_cv(
    pr_auc: dict[tuple[str, str], float],
    std: float = 0.01,
    default_offset: float = -0.05,
) -> pd.DataFrame:
    """Invented output of scripts/dfa/cv_select.py: for each (model, selection)
    key, a tuned row and a default-settings row whose PR-AUC is the tuned one
    plus `default_offset`. The scores describe no real model."""
    classes = dict(FAKE_DFA_MODELS)
    rows = []
    for (model, selection), score in pr_auc.items():
        for settings, value in (("tuned", score), ("default", score + default_offset)):
            rows.append(
                {
                    "model_cls": classes[model],
                    "selection": selection,
                    "embed_selector": "linear" if selection == "embed" else "",
                    "settings": settings,
                    "pr_auc_mean": value,
                    "pr_auc_std": std,
                    "auroc_mean": 0.5 + value / 2,
                    "bal_acc_mean": 0.5,
                    "error": "",
                }
            )
    return pd.DataFrame(rows)
