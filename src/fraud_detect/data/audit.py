"""Label shortcut audit: can product name and date predict the label by themselves?

The label is the Issue/Sub-issue the consumer chose. The CFPB has renamed
products and sub-issues over the years; for example, "Credit card or prepaid
card" became "Credit card" and "Prepaid card" in 2023. If fraud and not-fraud
categories came from different taxonomy versions, the product name and the
date received would reveal the label without the model reading any text, and
a random split would reward that shortcut.

These tables make such a pattern visible: the fraud rate by product name and
year, by sub-product and year, and the months in which each product name
occurs. They contain counts and rates only, never row-level data. Deciding
what to do about a shortcut (for example a time-based split) is a design
decision for the group, so this module only reports.
"""

from __future__ import annotations

import pandas as pd

from fraud_detect import columns as C

ALL_PRODUCTS = "All products"
NONE_LABEL = "(none)"


def rate_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Complaints, fraud complaints and fraud rate for each combination of keys."""
    grouped = df.groupby(keys, dropna=False)[C.LABEL]
    table = grouped.agg(n_complaints="size", n_fraud="sum", fraud_rate="mean")
    return table.reset_index()


def _with_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[C.PRODUCT, C.SUB_PRODUCT, C.DATE_RECEIVED, C.LABEL]].copy()
    out["year"] = pd.to_datetime(out[C.DATE_RECEIVED]).dt.year.astype(int)
    out[C.SUB_PRODUCT] = out[C.SUB_PRODUCT].fillna(NONE_LABEL)
    return out


def product_year(df: pd.DataFrame) -> pd.DataFrame:
    """Fraud rate by product name and year, plus an all-products row per year."""
    data = _with_year(df)
    per_product = rate_table(data, [C.PRODUCT, "year"])
    overall = rate_table(data, ["year"]).assign(**{C.PRODUCT: ALL_PRODUCTS})
    table = pd.concat([per_product, overall[per_product.columns]], ignore_index=True)
    return table.sort_values([C.PRODUCT, "year"], ignore_index=True)


def sub_product_year(df: pd.DataFrame) -> pd.DataFrame:
    """Fraud rate by product, sub-product and year (sub-products were renamed too)."""
    table = rate_table(_with_year(df), [C.PRODUCT, C.SUB_PRODUCT, "year"])
    return table.sort_values([C.PRODUCT, C.SUB_PRODUCT, "year"], ignore_index=True)


def product_span(df: pd.DataFrame) -> pd.DataFrame:
    """First and last month in which each product name occurs, with its overall
    fraud rate. Names that start or stop inside the window are renamed ones."""
    data = df[[C.PRODUCT, C.DATE_RECEIVED, C.LABEL]].copy()
    month = pd.to_datetime(data[C.DATE_RECEIVED]).dt.to_period("M").astype(str)
    data["month"] = month
    table = data.groupby(C.PRODUCT).agg(
        first_month=("month", "min"),
        last_month=("month", "max"),
        n_complaints=(C.LABEL, "size"),
        n_fraud=(C.LABEL, "sum"),
        fraud_rate=(C.LABEL, "mean"),
    )
    return table.reset_index().sort_values([C.PRODUCT], ignore_index=True)


def _cell(rate: float, n: int) -> str:
    return f"{rate:.3f} ({n:,})"


def pivot_markdown(table: pd.DataFrame) -> str:
    """A product × year grid: each cell is 'fraud rate (complaints)'."""
    years = sorted(table["year"].unique())
    names = set(table[C.PRODUCT])
    products = sorted(names - {ALL_PRODUCTS})
    if ALL_PRODUCTS in names:
        products.append(ALL_PRODUCTS)
    cells = {
        (r[C.PRODUCT], r["year"]): _cell(r["fraud_rate"], r["n_complaints"])
        for _, r in table.iterrows()
    }
    lines = [
        "| Product | " + " | ".join(str(y) for y in years) + " |",
        "|---|" + "|".join("---:" for _ in years) + "|",
    ]
    for product in products:
        row = [cells.get((product, y), "–") for y in years]
        name = f"**{product}**" if product == ALL_PRODUCTS else product
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def _span_markdown(span: pd.DataFrame) -> str:
    lines = [
        "| Product | First month | Last month | Complaints | Fraud rate |",
        "|---|---|---|---:|---:|",
    ]
    for _, r in span.iterrows():
        lines.append(
            f"| {r[C.PRODUCT]} | {r['first_month']} | {r['last_month']} | "
            f"{int(r['n_complaints']):,} | {r['fraud_rate']:.3f} |"
        )
    return "\n".join(lines)


def audit_report(labeled: pd.DataFrame, sample: pd.DataFrame) -> str:
    """Markdown summary of the audit for the labelled data and the 30k sample."""
    return "\n".join(
        [
            "# Label shortcut audit: fraud rate by product × year",
            "",
            "Each cell is the fraud rate, with the number of complaints in "
            "brackets. The label is the consumer's chosen Issue/Sub-issue "
            "(see docs/LABEL_RULE.md).",
            "",
            "**What to look for:** a product name that exists only in some "
            "years (a renamed product) and whose fraud rate differs sharply from "
            "its predecessor's in the same period. That would mean the product "
            "name and date reveal the label without the text, and a time-based "
            "split should be considered (an open decision; see AGENTS.md).",
            "",
            "## All labelled complaints (before deduplication and sampling)",
            "",
            pivot_markdown(product_year(labeled)),
            "",
            "## The sample used by the models (train and test together)",
            "",
            pivot_markdown(product_year(sample)),
            "",
            "## When each product name occurs (all labelled complaints)",
            "",
            _span_markdown(product_span(labeled)),
            "",
            "Sub-product × year rates are in `label_audit_sub_product_year.csv`.",
            "",
        ]
    )
