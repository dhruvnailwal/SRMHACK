"""Preprocessing and evaluation-split construction.

Three splits are produced (all documented, all leakage-aware):

A. TIME-BASED SPLIT
   The chronologically sorted stream is cut at fixed quantiles:
   train 60% / validation 20% / test 20% (by position). No shuffling.

B. UNSEEN-CUSTOMER SPLIT
   A disjoint hold-out group of customers (configurable share, default 20%)
   is randomly selected and its rows are *removed from the training fold*.
   Their remaining rows stay in validation/test and evaluate the "unseen"
   cohort. Rows of these customers never influence training or priors.

C. FIRST-TIME TRANSACTION SPLIT
   Rows where the customer, device or merchant is first seen in the stream
   (cold-start flags), reported separately for false-positive analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings


def history_stratum(n_prior: int, tenure_days: float, cfg: Settings) -> str:
    if n_prior <= cfg.splits["cold_max_prior"]:
        return "cold"
    if n_prior < cfg.splits["established_min_prior"] or tenure_days < 7:
        return "warm"
    return "established"


def time_split(df: pd.DataFrame, cfg: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the (already sorted) frame into train/val/test by position."""
    split = cfg.splits
    n = len(df)
    i1 = int(n * split["train_frac"])
    i2 = int(n * (split["train_frac"] + split["val_frac"]))
    train = df.iloc[:i1].reset_index(drop=True)
    val = df.iloc[i1:i2].reset_index(drop=True)
    test = df.iloc[i2:].reset_index(drop=True)
    return train, val, test


def unseen_customer_split(train: pd.DataFrame, val: pd.DataFrame,
                          test: pd.DataFrame, df: pd.DataFrame,
                          cfg: Settings) -> tuple[pd.DataFrame, pd.DataFrame,
                                                  pd.DataFrame, set]:
    """Remove a disjoint group of customers from the training fold.

    Returns (train_clean, val, test, heldout_customers). Held-out customers
    are sampled with a fixed seed so runs are reproducible. Their rows remain
    in validation/test (the evaluation sets) but never in training.
    """
    frac = cfg.splits["unseen_customer_frac"]
    rng = np.random.RandomState(cfg.seed)
    all_customers = np.array(sorted(df["customer_id"].unique()))
    n_hold = max(2, int(len(all_customers) * frac))
    holdout = set(rng.choice(all_customers, size=n_hold, replace=False).tolist())

    mask = ~train["customer_id"].isin(holdout)
    train_clean = train[mask].reset_index(drop=True)
    return train_clean, val, test, holdout


def first_time_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute cold-start flags from the recomputed history counts
    (available after the feature engine replay)."""
    out = df.copy()
    out["first_time_customer"] = (out["customer_history_count"] == 0).astype(int)
    out["first_time_device"] = (out["device_history_count"] == 0).astype(int)
    out["first_time_merchant"] = (out["merchant_history_count"] == 0).astype(int)
    out["first_time_any"] = (
        (out["customer_history_count"] == 0)
        | (out["device_history_count"] == 0)
        | (out["merchant_history_count"] == 0)
    ).astype(int)
    return out