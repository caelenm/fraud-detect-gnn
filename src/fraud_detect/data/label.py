"""Label rule: an explicit, human-reviewed allow-list of CFPB categories.

A complaint is positive (1) if its (product, issue, sub_issue) matches any
entry in the `positive` list of `configs/categories.yaml`, and negative (0)
otherwise. Complaints matching an `excluded` entry are removed from the
dataset before sampling: they are too ambiguous to label either way.
Keyword matching is used only to *flag candidates* for review; it never
assigns labels. The label stage refuses to run until:

* the file is marked `confirmed: true`,
* every entry matches at least one complaint (catches typos),
* no complaint matches more than one of positive, reviewed-negative and
  excluded, and
* every keyword candidate inside the target products has been classified as
  positive, reviewed-negative or excluded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fraud_detect import columns as C

WILDCARD = "*"

# Deliberately broad: false positives here only cost review time.
CANDIDATE_PATTERN = re.compile(
    r"fraud|scam|identity theft|unauthori[sz]ed|not authorized|stolen|"
    r"without (?:my|your) (?:consent|knowledge|permission)|"
    r"did not (?:make|authorize|purchase)|don'?t recognize|someone else",
    flags=re.IGNORECASE,
)


class LabelConfigError(RuntimeError):
    """Raised when the category review file is unconfirmed or inconsistent."""


@dataclass(frozen=True)
class CategoryRule:
    """One allow-list entry. `None` for sub_issue matches only missing values;
    '*' matches anything, including missing values."""

    product: str
    issue: str
    sub_issue: str | None

    @staticmethod
    def from_dict(entry: dict[str, Any]) -> CategoryRule:
        unknown = set(entry) - {"product", "issue", "sub_issue"}
        if unknown or "issue" not in entry:
            raise LabelConfigError(
                f"Bad category entry {entry}: needs 'issue' and optionally "
                "'product' and 'sub_issue'"
            )
        sub_issue = entry.get("sub_issue", WILDCARD)
        return CategoryRule(
            product=str(entry.get("product", WILDCARD)),
            issue=str(entry["issue"]),
            sub_issue=None if sub_issue is None else str(sub_issue),
        )

    def matches(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        if self.product != WILDCARD:
            mask &= df[C.PRODUCT] == self.product
        if self.issue != WILDCARD:
            mask &= df[C.ISSUE] == self.issue
        if self.sub_issue is None:
            mask &= df[C.SUB_ISSUE].isna()
        elif self.sub_issue != WILDCARD:
            mask &= df[C.SUB_ISSUE] == self.sub_issue
        return mask.fillna(False).astype(bool)


@dataclass(frozen=True)
class CategoryConfig:
    confirmed: bool
    target_products: tuple[str, ...]
    positive: tuple[CategoryRule, ...]
    reviewed_negative: tuple[CategoryRule, ...]
    excluded: tuple[CategoryRule, ...] = ()

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CategoryConfig:
        for key in ("confirmed", "target_products", "positive", "reviewed_negative"):
            if key not in data:
                raise LabelConfigError(f"categories file is missing '{key}'")

        def rules(key: str) -> tuple[CategoryRule, ...]:
            return tuple(CategoryRule.from_dict(e) for e in data.get(key) or [])

        return CategoryConfig(
            confirmed=data["confirmed"] is True,
            target_products=tuple(str(p) for p in data["target_products"]),
            positive=rules("positive"),
            reviewed_negative=rules("reviewed_negative"),
            excluded=rules("excluded"),  # optional: older files have none
        )

    def lists(self) -> tuple[tuple[str, tuple[CategoryRule, ...]], ...]:
        return (
            ("positive", self.positive),
            ("reviewed_negative", self.reviewed_negative),
            ("excluded", self.excluded),
        )


def any_match(df: pd.DataFrame, rules: tuple[CategoryRule, ...]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for rule in rules:
        mask |= rule.matches(df)
    return mask


def is_candidate(df: pd.DataFrame) -> pd.Series:
    """Rows whose Issue or Sub-issue text matches the candidate keywords."""
    issue = df[C.ISSUE].fillna("").str.contains(CANDIDATE_PATTERN)
    sub = df[C.SUB_ISSUE].fillna("").str.contains(CANDIDATE_PATTERN)
    return (issue | sub).astype(bool)


def validate_categories(df: pd.DataFrame, cfg: CategoryConfig) -> None:
    """Check the reviewed allow-list against the loaded complaints.

    `df` must contain all loaded complaints (all products).
    """
    problems: list[str] = []
    if not cfg.confirmed:
        problems.append(
            "the categories file is not confirmed. Review it against "
            "outputs/reports/category_values.csv, then set `confirmed: true`."
        )

    products = set(df[C.PRODUCT].dropna().unique())
    missing_products = [p for p in cfg.target_products if p not in products]
    if missing_products:
        problems.append(f"target_products not found in the data: {missing_products}")

    masks = {}
    for kind, rules in cfg.lists():
        for rule in rules:
            if not rule.matches(df).any():
                problems.append(f"{kind} entry matches no complaints: {rule}")
        masks[kind] = any_match(df, rules)

    kinds = list(masks)
    for i, a in enumerate(kinds):
        for b in kinds[i + 1 :]:
            both = masks[a] & masks[b]
            if both.any():
                problems.append(f"complaints match both {a} and {b}: {_combos(df[both])}")

    in_target = df[C.PRODUCT].isin(cfg.target_products)
    covered = masks["positive"] | masks["reviewed_negative"] | masks["excluded"]
    uncovered = in_target & is_candidate(df) & ~covered
    if uncovered.any():
        problems.append(
            "keyword candidates in target products are not classified as positive, "
            f"reviewed_negative or excluded: {_combos(df[uncovered])}"
        )

    if problems:
        raise LabelConfigError("Label rule is not ready:\n- " + "\n- ".join(problems))


def _combos(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    keys = [C.PRODUCT, C.ISSUE, C.SUB_ISSUE]
    rows = df[keys].astype("string").fillna("<missing>").drop_duplicates()
    return [tuple(r) for r in rows.itertuples(index=False)]


def apply_labels(df: pd.DataFrame, cfg: CategoryConfig) -> pd.DataFrame:
    """Keep target products, drop excluded categories, add the binary label, and
    drop Issue/Sub-issue."""
    out = df[df[C.PRODUCT].isin(cfg.target_products)]
    out = out[~any_match(out, cfg.excluded)].copy()
    out[C.LABEL] = any_match(out, cfg.positive).astype("int8")
    return out.drop(columns=list(C.LABEL_SOURCE_COLUMNS)).reset_index(drop=True)


def excluded_report(df: pd.DataFrame, cfg: CategoryConfig) -> pd.DataFrame:
    """Complaints in the target products removed by the `excluded` list, per
    category (for the label stage's report)."""
    in_target = df[df[C.PRODUCT].isin(cfg.target_products)]
    dropped = in_target[any_match(in_target, cfg.excluded)]
    keys = [C.PRODUCT, C.ISSUE, C.SUB_ISSUE]
    counts = dropped[keys].astype("string").fillna("<missing>").value_counts()
    return counts.rename("n_complaints").reset_index()


def candidate_report(df: pd.DataFrame, cfg: CategoryConfig | None) -> pd.DataFrame:
    """Per-category counts with keyword-candidate and current-rule flags."""
    keys = [C.PRODUCT, C.ISSUE, C.SUB_ISSUE]
    combos = df[keys].drop_duplicates().reset_index(drop=True)
    counts = df[keys].astype("string").fillna("<missing>").value_counts()
    report = combos.copy()
    shown = combos.astype("string").fillna("<missing>")
    report["n_complaints"] = [counts[tuple(r)] for r in shown.itertuples(index=False)]
    report["keyword_candidate"] = is_candidate(combos).to_numpy()
    if cfg is not None:
        report["in_target_products"] = combos[C.PRODUCT].isin(cfg.target_products)
        report["rule_positive"] = any_match(combos, cfg.positive).to_numpy()
        report["rule_reviewed_negative"] = any_match(
            combos, cfg.reviewed_negative
        ).to_numpy()
        report["rule_excluded"] = any_match(combos, cfg.excluded).to_numpy()
    report[keys] = shown
    return report.sort_values(
        [C.PRODUCT, "n_complaints"], ascending=[True, False], ignore_index=True
    )
