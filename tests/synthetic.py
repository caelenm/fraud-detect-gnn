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
