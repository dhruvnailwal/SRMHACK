"""Data loading with tolerant schema mapping.

The dataset may differ slightly from the documented one (different column
names, timestamp formats, extra columns). Instead of failing we map common
aliases to a canonical schema, sort chronologically, and attach machine
timestamps. The mapping table is recorded so the report can show what was
mapped.
"""
from __future__ import annotations

import os

import pandas as pd

from src.config import Settings

# ------------------------------------------------------------------ aliases
ALIASES = {
    "transaction_id": ["transaction_id", "txn_id", "id", "transactionid"],
    "timestamp": ["timestamp", "event_time", "time", "transactiontime", "tx_time"],
    "customer_id": ["customer_id", "customer_key", "cid", "user_id"],
    "merchant_id": ["merchant_id", "merchant_key", "mid", "merchant"],
    "device_id": ["device_id", "device_key", "did", "device"],
    "amount": ["amount", "amt", "transaction_amount"],
    "transaction_hour": ["transaction_hour", "hour", "hr"],
    "is_night": ["is_night", "night", "night_flag"],
    "is_fraud": ["is_fraud", "fraud", "label", "isFraud", "class"],
    "new_customer": ["new_customer", "is_new_customer"],
    "new_merchant": ["new_merchant", "is_new_merchant"],
    "new_device": ["new_device", "is_new_device"],
    "velocity_8min": ["velocity_8min", "velocity", "txns_8min"],
    "amount_spike": ["amount_spike", "spike", "amount_flag"],
    "customer_history_count": ["customer_history_count", "cust_hist_count", "customer_count"],
    "merchant_history_count": ["merchant_history_count", "merch_hist_count", "merchant_count"],
    "device_history_count": ["device_history_count", "device_hist_count", "device_count"],
}


class SchemaMap:
    """Resolves raw columns to canonical names, recording every mapping."""

    def __init__(self, columns, aliases=None, require_label=True):
        aliases = aliases or ALIASES
        col_lookup = {str(c).strip().lower(): c for c in columns}
        self.mapping = {}
        self.unmapped = []
        for canonical, candidate_names in aliases.items():
            found = None
            for cand in candidate_names:
                if cand in col_lookup:
                    found = col_lookup[cand]
                    break
            if found is not None:
                self.mapping[canonical] = found
            else:
                self.unmapped.append(canonical)
        self.required = ["transaction_id", "timestamp", "amount", "customer_id",
                         "merchant_id", "device_id"]
        if require_label:
            self.required.append("is_fraud")
        self.missing_required = [c for c in self.required if c not in self.mapping]

    def rename(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {v: k for k, v in self.mapping.items()}
        return df.rename(columns=rename_map)


def load_dataframe(cfg: Settings, path: str | None = None,
                   require_label: bool = True) -> tuple[pd.DataFrame, SchemaMap]:
    """Load, map, coerce and chronologically sort the raw CSV.

    Returns (canonical_df, schema_map). Raises ValueError if any required
    canonical column cannot be mapped. ``require_label=False`` allows
    scoring unlabelled files (``is_fraud`` is filled with 0).
    """
    path = path or cfg.raw_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            "Generate it first with:  python scripts/generate_dataset.py"
        )
    df = pd.read_csv(path)

    schema = SchemaMap(df.columns, require_label=require_label)
    if schema.missing_required:
        raise ValueError("Required columns missing after alias mapping: "
                         + ", ".join(schema.missing_required))

    df = schema.rename(df)

    # ---- treat a missing label as all-legitimate when scoring unlabelled ----
    if not require_label and "is_fraud" not in df.columns:
        df["is_fraud"] = 0

    # ---- drop whole-row duplicates on the transaction id ----
    before = len(df)
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")
    dup_dropped = before - len(df)

    # ---- type coercion ----
    if "amount" in df.columns and df["amount"].dtype == object:
        df["amount"] = (df["amount"].astype(str)
                        .str.replace(r",(?=\d{3}(?:\.\d*)?$)", "",
                                     regex=True))
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    df["amount"] = df["amount"].where(df["amount"] > 0)

    # ---- parse timestamps ----
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["_parse_ok"] = ts.notna()
    df["timestamp"] = ts
    bad_ts = df["_parse_ok"][df["_parse_ok"] == False].shape[0]  # noqa: E712
    df = df[df["_parse_ok"]].drop(columns="_parse_ok")
    df["ts_sec"] = df["timestamp"].astype("int64") // 10**9

    # guaranteed stable chronological order (tie-break by id)
    df = df.sort_values(["ts_sec", "transaction_id"]).reset_index(drop=True)

    # ---- optional raw-derived columns backfilled when absent ----
    if "transaction_hour" not in df.columns:
        df["transaction_hour"] = df["timestamp"].dt.hour
    if "is_night" not in df.columns:
        hour = df["transaction_hour"].astype(int)
        df["is_night"] = ((hour < 6) | (hour >= 23)).astype(int)

    meta = {
        "path": path,
        "rows_loaded": len(df),
        "duplicate_txns_dropped": int(dup_dropped),
        "unparseable_timestamps": int(bad_ts),
        "mapped_columns": schema.mapping,
        "unmapped_optional_columns": schema.unmapped,
        "columns": list(df.columns),
    }
    df.attrs["meta"] = meta
    df.attrs["schema_map"] = schema
    return df, schema


def valid_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with the (rare) unresolved amount; keeps the pipeline honest."""
    return df[df["amount"].notna()].reset_index(drop=True)