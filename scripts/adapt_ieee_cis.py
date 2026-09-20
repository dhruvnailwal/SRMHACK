"""Adapt the IEEE-CIS Fraud Detection dataset to the canonical pipeline schema.

The pipeline's ``data_loader`` requires five entity columns plus a target:
    transaction_id, timestamp, amount, customer_id, merchant_id, device_id
        [, is_fraud]

IEEE-CIS ``train_transaction.csv`` has none of these directly (it uses
``TransactionID/TransactionDT/TransactionAmt`` + anonymised ``card*`` /
``addr*`` / ``dist*`` features and a separate ``train_identity.csv``).  This
script performs the *identity-free* mapping:

    transaction_id  <- TransactionID
    timestamp       <- base_epoch + TransactionDT seconds (hour-of-day exact)
    amount          <- TransactionAmt
    customer_id     <- card1        (card = the account-level actor)
    merchant_id     <- addr1        (billing region as counterparty proxy)
    device_id       <- DeviceType|DeviceInfo from identity (else "NA_DEVICE")
    is_fraud        <- isFraud      (train only)

Raw behaviour-describing columns (``card2/5``, ``dist1``, email domains) are
kept as unmapped optional columns so the loader tolerates the rest of the
schema drift.  No raw identity is ever fed to the model - the ids only key
the streaming state, exactly as the documented design requires.

Usage:
    python scripts/adapt_ieee_cis.py                     # full train set
    python scripts/adapt_ieee_cis.py --limit 200000      # smoke slice
    python scripts/adapt_ieee_cis.py --out data/raw/ieee_cis_canonical.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_EPOCH = pd.Timestamp("2017-10-31", tz="UTC")

_ID = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
       "ProductCD", "card1", "card2", "card5", "addr1", "addr2", "dist1",
       "P_emaildomain", "R_emaildomain"]
_IDENTITY = ["TransactionID", "DeviceType", "DeviceInfo"]


def _device_id(types: pd.Series, info: pd.Series) -> pd.Series:
    out = info.fillna("").astype(str) + "|" + types.fillna("").astype(str)
    out = out.str.strip("|")
    return out.where(out != "", "NA_DEVICE")


def adapt(trans_path: str, identity_path: str,
          out_path: str, limit: int | None = None) -> dict:
    cols = dict()
    df = pd.read_csv(trans_path, usecols=_ID, low_memory=False)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)
    n = len(df)
    print(f"loaded {n:,} transactions from {os.path.basename(trans_path)}")

    # ---- identity join (24% coverage; rest get NA_DEVICE) --------------
    if os.path.exists(identity_path):
        idf = pd.read_csv(identity_path, usecols=_IDENTITY)
        df = df.merge(idf, on="TransactionID", how="left")
    else:
        df["DeviceType"] = np.nan
        df["DeviceInfo"] = np.nan

    ts = BASE_EPOCH + pd.to_timedelta(df["TransactionDT"], unit="s")
    hour = (df["TransactionDT"].to_numpy() // 3600) % 24

    out = pd.DataFrame({
        "transaction_id": df["TransactionID"].astype(str),
        "timestamp": ts,
        "amount": df["TransactionAmt"],
        "customer_id": ("card_" + df["card1"].fillna(-1).astype(str)),
        "merchant_id": ("addr_" + df["addr1"].fillna(-1).astype(str)),
        "device_id": _device_id(df["DeviceType"], df["DeviceInfo"]),
        "transaction_hour": hour.astype(int),
        "is_fraud": df["isFraud"].fillna(0).astype(int),
        # optional raw columns (tolerated, never used as features)
        "ProductCD": df["ProductCD"],
        "card2": df["card2"],
        "card5": df["card5"],
        "addr2": df["addr2"],
        "dist1": df["dist1"],
        "P_emaildomain": df["P_emaildomain"],
        "R_emaildomain": df["R_emaildomain"],
    }).sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    meta = {
        "rows": n,
        "out_path": out_path,
        "fraud_rate_pct": round(float(out["is_fraud"].mean() * 100), 4),
        "unique_customers": int(out["customer_id"].nunique()),
        "unique_merchants": int(out["merchant_id"].nunique()),
        "unique_devices": int(out["device_id"].nunique()),
        "pct_with_device": round(
            100.0 * (out["device_id"] != "NA_DEVICE").mean(), 2),
        "timestamp_range": (str(out["timestamp"].min()),
                            str(out["timestamp"].max())),
        "mapping": {"transaction_id": "TransactionID",
                    "timestamp": "BASE_EPOCH + TransactionDT s",
                    "amount": "TransactionAmt",
                    "customer_id": "card1",
                    "merchant_id": "addr1 (proxy)",
                    "device_id": "DeviceType|DeviceInfo (identity)",
                    "is_fraud": "isFraud"},
    }
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--trans", default=os.path.join(
        ROOT, "data", "raw", "train_transaction.csv"))
    ap.add_argument("--identity", default=os.path.join(
        ROOT, "data", "raw", "train_identity.csv"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "data", "raw", "ieee_cis_canonical.csv"))
    ap.add_argument("--limit", type=int, default=None,
                    help="only adapt the first N transactions (smoke runs)")
    args = ap.parse_args()

    meta = adapt(args.trans, args.identity, args.out, args.limit)
    print("adapted schema:")
    for k, v in meta["mapping"].items():
        print(f"  {k:>15} <- {v}")
    print(f"  {meta['rows']:,} rows, {meta['fraud_rate_pct']:.2f}% fraud, "
          f"{meta['unique_customers']:,} customers (cards), "
          f"{meta['unique_merchants']:,} merchants (addr1), "
          f"{meta['unique_devices']:,} devices, "
          f"{meta['pct_with_device']:.1f}% with device info")
    print(f"  timestamp range: {meta['timestamp_range'][0]} -> "
          f"{meta['timestamp_range'][1]}")
    print(f"wrote {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())