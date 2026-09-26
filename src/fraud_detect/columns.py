"""Canonical column names used throughout the pipeline.

Raw CFPB exports use human-readable headers ("Consumer complaint narrative")
while the CFPB API uses snake_case names ("complaint_what_happened"). Both are
mapped to the canonical names below when the raw files are loaded.
"""

from __future__ import annotations

COMPLAINT_ID = "complaint_id"
DATE_RECEIVED = "date_received"
PRODUCT = "product"
SUB_PRODUCT = "sub_product"
ISSUE = "issue"
SUB_ISSUE = "sub_issue"
NARRATIVE = "narrative"
COMPANY = "company"
STATE = "state"
TAGS = "tags"
SUBMITTED_VIA = "submitted_via"
COMPANY_RESPONSE = "company_response"
TIMELY_RESPONSE = "timely_response"
LABEL = "label"

# Canonical name -> accepted raw header spellings. Only these columns are read
# from the raw files; every other raw column (company public response, ZIP
# code, consent, date sent to company, consumer disputed) is ignored.
RAW_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    COMPLAINT_ID: ("Complaint ID", "complaint_id"),
    DATE_RECEIVED: ("Date received", "date_received"),
    PRODUCT: ("Product", "product"),
    SUB_PRODUCT: ("Sub-product", "sub_product"),
    ISSUE: ("Issue", "issue"),
    SUB_ISSUE: ("Sub-issue", "sub_issue"),
    NARRATIVE: ("Consumer complaint narrative", "complaint_what_happened"),
    COMPANY: ("Company", "company"),
    STATE: ("State", "state"),
    TAGS: ("Tags", "tags"),
    SUBMITTED_VIA: ("Submitted via", "submitted_via"),
    COMPANY_RESPONSE: ("Company response to consumer", "company_response"),
    TIMELY_RESPONSE: ("Timely response?", "timely"),
}

# Columns that define the label. They must never appear in any feature
# matrix, node feature, or edge (AGENTS.md leakage invariant 1).
LABEL_SOURCE_COLUMNS: tuple[str, ...] = (ISSUE, SUB_ISSUE)

# Per-complaint outcome columns. They are used only to build company-level
# statistics from training complaints, never as per-complaint features.
COMPANY_OUTCOME_COLUMNS: tuple[str, ...] = (COMPANY_RESPONSE, TIMELY_RESPONSE)


def is_label_derived(column: str) -> bool:
    """Return True if a column name looks like Issue/Sub-issue or something built
    from them. Used as a guard on every feature table."""
    lowered = column.lower().replace("-", "_").replace(" ", "_")
    return "issue" in lowered
