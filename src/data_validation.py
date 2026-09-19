"""Data validation and data-quality report.

Builds a structured report (dict + markdown + JSON) covering schema, types,
missingness, duplicates, class distribution, cardinalities, timestamp range,
numerical/categorical summaries and the fraud rate. Runs on whatever schema
the loader produced (configurable, never brittle).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

NUMERIC_SUMMARY_COLS = ["amount", "transaction_hour"]
CAT_COLS = ["is_night", "new_customer", "new_merchant", "new_device",
            "is_fraud", "velocity_8min", "amount_spike",
            "customer_history_count", "merchant_history_count",
            "device_history_count"]


def _summary(value, key):
    text = f"{value:,}"
    if key in ("rows", "total_transactions"):
        return text
    return text


def build_validation_report(df: pd.DataFrame, load_meta: dict | None = None) -> dict:
    """Compute the data-quality summary and return it as a nested dict."""
    if load_meta is None:
        load_meta = df.attrs.get("meta", {})

    n = len(df)
    numeric_cols = [c for c in NUMERIC_SUMMARY_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    missing = {}
    for c in df.columns:
        miss = int(df[c].isna().sum())
        if miss > 0:
            missing[c] = miss

    duplicates_df = int(df.duplicated(subset=["transaction_id"]).sum()) if "transaction_id" in df.columns else 0

    class_dist = df["is_fraud"].value_counts().sort_index().to_dict()
    fraud_n = int(class_dist.get(1, 0))
    fraud_pct = round(100.0 * fraud_n / n, 4) if n else 0.0

    num_summary = {}
    for c in numeric_cols:
        s = df[c].describe()
        num_summary[c] = {
            "mean": round(float(s["mean"]), 4),
            "std": round(float(s["std"]), 4),
            "min": round(float(s["min"]), 4),
            "p25": round(float(s["25%"]), 4),
            "median": round(float(s["50%"]), 4),
            "p75": round(float(s["75%"]), 4),
            "max": round(float(s["max"]), 4),
        }

    cat_summary = {}
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).to_dict()
        cat_summary[c] = {str(k): int(v) for k, v in vc.items()}

    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    days = (ts_max - ts_min).total_seconds() / 86400.0 if n else 0.0

    uniq = {
        "customers": int(df["customer_id"].nunique()),
        "merchants": int(df["merchant_id"].nunique()),
        "devices": int(df["device_id"].nunique()),
    }

    report = {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": n,
        "columns": int(df.shape[1]),
        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
        "missing_values": missing,
        "duplicates_in_transaction_id": duplicates_df,
        "duplicate_txns_dropped_at_load": load_meta.get("duplicate_txns_dropped", 0),
        "class_distribution": class_dist,
        "fraud_count": fraud_n,
        "fraud_percent": fraud_pct,
        "unique_counts": uniq,
        "timestamp_range": {"min": str(ts_min), "max": str(ts_max),
                            "days_span": round(days, 1)},
        "numerical_summary": num_summary,
        "categorical_summary": cat_summary,
        "load_meta": {k: v for k, v in load_meta.items() if k != "mapped_columns"},
        "column_mapping_used": load_meta.get("mapped_columns", {}),
    }
    return report


def format_markdown(report: dict) -> str:
    lines = []
    lines.append("# Data Validation Report")
    lines.append("")
    lines.append(f"Generated: {report['report_generated_at']}")
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Total transactions: **{report['rows']:,}**")
    lines.append(f"- Total columns: **{report['columns']}**")
    lines.append(f"- Fraud cases: **{report['fraud_count']:,}** "
                 f"({report['fraud_percent']:.2f}%)")
    lines.append("")

    lines.append("## Data types")
    lines.append("")
    lines.append("| Column | dtype |")
    lines.append("|---|---|")
    for c, d in report["dtypes"].items():
        lines.append(f"| {c} | {d} |")
    lines.append("")

    lines.append("## Missing value report")
    lines.append("")
    if report["missing_values"]:
        lines.append("| Column | Missing rows |")
        lines.append("|---|---|")
        for c, v in report["missing_values"].items():
            lines.append(f"| {c} | {v:,} |")
    else:
        lines.append("No missing values found.")
    lines.append("")

    lines.append("## Duplicates")
    lines.append("")
    lines.append(f"- Duplicate `transaction_id` rows in the file: "
                 f"**{report['duplicates_in_transaction_id']:,}**")
    lines.append(f"- Duplicates dropped at load time: "
                 f"**{report['duplicate_txns_dropped_at_load']:,}**")
    lines.append("")

    lines.append("## Class distribution")
    lines.append("")
    lines.append("| Class | Count |")
    lines.append("|---|---|")
    for k, v in report["class_distribution"].items():
        lines.append(f"| {k} | {v:,} |")
    lines.append("")
    lines.append(f"- Fraud percentage: **{report['fraud_percent']:.4f}%**")
    lines.append("")

    lines.append("## Unique entities")
    lines.append("")
    for k, v in report["unique_counts"].items():
        lines.append(f"- Unique {k}: **{v:,}**")
    lines.append("")

    lines.append("## Timestamp range")
    lines.append("")
    lines.append(f"- Min: `{report['timestamp_range']['min']}`")
    lines.append(f"- Max: `{report['timestamp_range']['max']}`")
    lines.append(f"- Span: **{report['timestamp_range']['days_span']} days**")
    lines.append("")

    lines.append("## Numerical feature summary")
    lines.append("")
    for c, s in report["numerical_summary"].items():
        lines.append(f"### `{c}`")
        lines.append("")
        lines.append("| stat | value |")
        lines.append("|---|---|")
        for k, v in s.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.append("## Categorical feature summary")
    lines.append("")
    for c, vc in report["categorical_summary"].items():
        lines.append(f"### `{c}`")
        lines.append("")
        lines.append("| value | count |")
        lines.append("|---|---|")
        for k, v in vc.items():
            lines.append(f"| {k} | {v:,} |")
        lines.append("")

    return "\n".join(lines)


def write_validation_report(df: pd.DataFrame, out_dir: str,
                            load_meta: dict | None = None) -> str:
    """Write ``data_validation.json`` and ``01_data_validation.md`` and return
    the markdown text."""
    os.makedirs(out_dir, exist_ok=True)
    report = build_validation_report(df, load_meta)
    json_path = os.path.join(out_dir, "data_validation.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    md_path = os.path.join(out_dir, "01_data_validation.md")
    md_text = format_markdown(report)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_text)
    return md_text