"""Leakage-safe behavioural feature engineering.

Single implementation, two entry points:

* ``FeatureEngine.replay(df)``  - batch replay used for training/evaluation.
* ``FeatureEngine.prepare_row / commit_row`` - the same code path exposed
  for one-transaction-at-a-time streaming scoring, with state updates
  deferred until *after* scoring.

Correctness rules enforced here
-------------------------------
1. Transactions are processed in strict chronological order.
2. Every feature for row *t* uses ONLY state built from rows *< t*.
3. ``commit_row`` (history update) is never interleaved with feature
   computation for the same transaction - ``prepare_row`` first.
4. Raw customer_id/merchant_id/device_id never become features; they only
   key the internal state.
5. Cold-start baselines shrink the customer's own (thin) history toward a
   *segment* baseline (history stratum x merchant tier) and finally toward a
   global baseline, so missing history is not treated as evidence of fraud.
6. Encoders / bins / segment statistics are fitted on training data only and
   frozen into ``priors``; nothing is recomputed on the full dataset.

Normalisation notes (the "how" behind the scores)
--------------------------------------------------
- ``amount_z_shrunk`` is a *shrunk z-score*: a blend of the customer's own
  log-amount mean/std with the segment/global prior. For a brand-new customer
  (n = 0) it equals the deviation from the segment prior.
- ``novelty_score`` (see novelty_model.py) is an ECDF rank of the anomaly
  model output relative to legitimate traffic, so 0.90 means "more atypical
  than 90% of legitimate transactions".
"""
from __future__ import annotations

import bisect
import math

import numpy as np
import pandas as pd

from src.config import Settings

# --------------------------------------------------------------------------
# Feature catalogs (single source of truth for every consumer)
# --------------------------------------------------------------------------
MODEL_FEATURES = [
    "amount_log", "transaction_hour", "day_of_week", "is_weekend", "is_night",
    "amount_bucket", "amount_z_shrunk", "customer_amount_z", "merchant_amount_z",
    "txn_count_5m", "txn_count_5m_log", "txn_count_30m", "txn_count_30m_log",
    "txn_count_24h", "txn_count_24h_log", "distinct_merchants_24h",
    "distinct_devices_24h", "amount_sum_24h_log",
    "customer_history_count", "merchant_history_count", "device_history_count",
    "customer_history_count_log", "merchant_history_count_log",
    "device_history_count_log",
    "customer_avg_amount_log", "merchant_avg_amount_log",
    "time_since_customer_txn_log", "time_since_device_txn_log",
    "new_device", "new_merchant", "first_time_customer",
    "customer_history_available", "merchant_history_available",
    "device_history_available", "customer_history_conf", "history_stratum",
    "customer_amount_baseline", "segment_amount_baseline",
    "global_amount_baseline", "segment_risk_prior",
]

NOVELTY_FEATURES = [
    "amount_log", "amount_z_shrunk", "transaction_hour", "is_night", "is_weekend",
    "txn_count_5m_log", "txn_count_30m_log", "txn_count_24h_log",
    "new_device", "new_merchant", "first_time_customer",
    "time_since_customer_txn_log", "time_since_device_txn_log",
    "customer_history_count_log", "merchant_history_count_log",
    "device_history_count_log",
    "distinct_devices_24h", "distinct_merchants_24h",
    "customer_history_conf", "amount_sum_24h_log",
]

ROUGH_PASSTHROUGH = [
    "transaction_id", "timestamp", "ts_sec", "customer_id", "merchant_id",
    "device_id", "amount", "transaction_hour", "is_night", "day_of_week",
    "is_weekend", "is_fraud", "new_customer", "velocity_8min", "amount_spike",
]

STRATUM_ORD = {"cold": 0, "warm": 1, "established": 2}
TIER_LEVELS = ["rare", "low", "mid", "high"]


def merchant_tier(n_prior: int) -> str:
    if n_prior <= 0:
        return "rare"
    if n_prior <= 2:
        return "low"
    if n_prior <= 9:
        return "mid"
    return "high"


def history_stratum(n_prior: int) -> str:
    if n_prior <= 0:
        return "cold"
    if n_prior < 5:
        return "warm"
    return "established"


class _Welford:
    """Incremental mean / M2 of a stream of values (population variance)."""

    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def push(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def sd(self):
        return math.sqrt(self.M2 / self.n) if self.n > 1 else 0.0


class FeatureEngine:
    """Point-in-time streaming feature engine (one production code path)."""

    def __init__(self, cfg: Settings, priors: dict | None = None):
        self.cfg = cfg
        self.priors = priors or {}
        self.kappa = float(cfg.features["cold_kappa"])
        self.baseline_min_n = int(cfg.features["customer_baseline_min_n"])
        self.reset_state()

    # ------------------------------------------------------------------ state
    def reset_state(self):
        self.customers: dict = {}
        self.merchants: dict = {}
        self.devices: dict = {}
        self.global_ = {"n": 0, "sum_amt": 0.0, "n_log": 0, "sum_log": 0.0,
                        "mean_amt": None, "mean_log": None}

    def _empty_customer(self):
        c = self.customers
        return {"n": 0, "sqrtn": 0.0, "sum_amt": 0.0, "sum_log": 0.0,
                "wf": _Welford(), "times": [], "pairs": [],
                "first_seen": None, "last_seen": None}

    def _empty_merchant(self):
        return {"n": 0, "sum_amt": 0.0, "wf": _Welford(),
                "first_seen": None, "last_seen": None}

    def _empty_device(self):
        return {"n": 0, "first_seen": None, "last_seen": None}

    # ------------------------------------------------------------- priors fit
    def compute_priors(self, train_df: pd.DataFrame) -> dict:
        """Fit segment/global baselines from TRAINING rows only.

        Stratum and merchant tier are evaluated as-of each training row (from
        a fresh replay of the training fold), then aggregated. The resulting
        priors are frozen and reused for every later transaction.
        """
        seg_data = {}
        stratum_data = {}
        tier_data = {}
        glob_log = []
        glob_frauds = 0
        glob_n = 0

        for row in train_df.itertuples(index=False):
            cid, mid = row.customer_id, row.merchant_id
            log_amt = math.log1p(float(row.amount))
            c = self.customers.get(cid)
            m = self.merchants.get(mid)
            n_c = c["n"] if c else 0
            n_m = m["n"] if m else 0
            s = history_stratum(n_c)
            t = merchant_tier(n_m)
            fraud = int(row.is_fraud) if hasattr(row, "is_fraud") else 0

            for key in ((s, t), (s, "any"), ("any", t), ("any", "any")):
                acc = seg_data.setdefault(key, [])
                acc.append((log_amt, fraud))
            stratum_data.setdefault(s, []).append(fraud)
            tier_data.setdefault(t, []).append(fraud)
            glob_log.append(log_amt)
            glob_frauds += fraud
            glob_n += 1

            self._commit_state(cid, mid, row.device_id if hasattr(row, "device_id") else "",
                               float(row.amount), log_amt, float(getattr(row, "ts_sec", 0.0)))

        # ---- aggregate segment stats --------------------------------------
        def _stats(pairs):
            if not pairs:
                return None
            log_vals = np.array([p[0] for p in pairs], dtype=float)
            frauds = sum(p[1] for p in pairs)
            n = len(log_vals)
            return {
                "mean": float(log_vals.mean()),
                "std": float(log_vals.std(ddof=0)) or 0.05,
                "n": n,
                "risk": float((frauds + 20.0 * (glob_frauds / max(1, glob_n))) / (n + 20.0)),
            }

        segments = {k: _stats(v) for k, v in seg_data.items()}
        glob_mean = float(np.mean(glob_log)) if glob_log else 0.0
        glob_std = float(np.std(glob_log, ddof=0)) or 0.05
        global_prior = {
            "mean": glob_mean, "std": glob_std,
            "n": glob_n, "risk": glob_frauds / max(1, glob_n),
        }
        bins = np.quantile(np.array(glob_log) if glob_log else np.array([0.0]),
                           [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]) \
            if glob_log else np.zeros(7)

        return {
            "segments": segments,
            "global": global_prior,
            "amount_bins": np.asarray(bins),
            "fitted_on": "train",
        }

    # --------------------------------------------------------- scoring path
    def prepare_row(self, txn: dict) -> dict:
        """Compute features for ONE transaction from current state only."""
        ts = float(txn["ts_sec"])
        amt = float(txn["amount"])
        log_amt = math.log1p(amt)
        cid = txn["customer_id"]
        mid = txn["merchant_id"]
        did = txn["device_id"]
        hour = int(txn.get("transaction_hour", 0))

        c = self.customers.get(cid)
        m = self.merchants.get(mid)
        d = self.devices.get(did)
        g = self.global_

        n_c = c["n"] if c else 0
        n_m = m["n"] if m else 0
        n_d = d["n"] if d else 0

        # ----- layer-1: transaction -----------------------------------
        day = (int(ts // 86400) + 3) % 7
        is_weekend = 1 if day >= 5 else 0
        is_night = 1 if (hour < 6 or hour >= 23) else 0
        bins = self.priors.get("amount_bins")
        if bins is None or len(bins) == 0:
            amount_bucket = 0
        else:
            amount_bucket = int(np.digitize(log_amt, bins))

        # ----- layer-2: cold-start-aware deviation ----------------------
        mu_c = c["wf"].mean if c and n_c > 0 else None
        sd_c = c["wf"].sd() if c else 0.0

        seg = self._segment_lookup(history_stratum(n_c), merchant_tier(n_m))
        mu_seg = seg["mean"]
        sd_seg = seg["std"]
        risk_seg = seg["risk"]
        global_mean_log = self.priors.get("global", {}).get(
            "mean", g["mean_log"] if g and g.get("mean_log") is not None else 0.0)
        global_std = self.priors.get("global", {}).get("std", 0.3)

        conf = n_c / (n_c + self.kappa)
        if mu_c is None:
            mu = mu_seg
            sd2 = sd_seg ** 2
        else:
            mu = conf * mu_c + (1 - conf) * mu_seg
            sd2 = (conf * (sd_c ** 2) + (1 - conf) * (sd_seg ** 2)
                   + conf * (1 - conf) * (mu_c - mu_seg) ** 2)
        sd = math.sqrt(max(sd2, 1e-6))
        z = (log_amt - mu) / sd if sd else 0.0

        if n_c >= self.baseline_min_n:
            cust_z = (log_amt - mu_c) / max(sd_c, 1e-6) if sd_c > 1e-6 else 0.0
        else:
            cust_z = z

        if n_m >= self.baseline_min_n and m and m["wf"].sd() > 1e-6:
            merch_z = (log_amt - m["wf"].mean) / m["wf"].sd()
        else:
            merch_z = 0.0

        # ----- layer-3: customer velocity (strictly prior txns) --------
        times = c["times"] if c else ()
        txn_count_5m = len(times) - bisect.bisect_left(times, ts - 300)
        txn_count_30m = len(times) - bisect.bisect_left(times, ts - 1800)
        txn_count_24h = len(times) - bisect.bisect_left(times, ts - 86400)

        distinct_merchants_24h = 0
        distinct_devices_24h = 0
        amount_sum_24h = 0.0
        if c:
            cutoff24 = ts - 86400
            _mids = set()
            _dids = set()
            for pair in reversed(c["pairs"]):
                if pair[0] < cutoff24:
                    break
                _sum_amount = pair[3]
                amount_sum_24h += _sum_amount
                _mids.add(pair[1])
                _dids.add(pair[2])
            distinct_merchants_24h = len(_mids)
            distinct_devices_24h = len(_dids)

        # ----- layer-4: history + cold-start ---------------------------
        customer_avg = mu_c if mu_c is not None else mu_seg
        merchant_avg = m["wf"].mean if m and n_m >= 1 else global_mean_log

        ts_last_c = c["last_seen"] if c else None
        ts_last_d = d["last_seen"] if d else None
        since_c = (ts - ts_last_c) / 3600.0 if ts_last_c is not None else 0.0
        since_d = (ts - ts_last_d) / 3600.0 if ts_last_d is not None else 0.0

        feats = {
            "amount_log": log_amt,
            "transaction_hour": hour,
            "day_of_week": day,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "amount_bucket": amount_bucket,
            "amount_z_shrunk": z,
            "customer_amount_z": cust_z,
            "merchant_amount_z": merch_z,
            "txn_count_5m": txn_count_5m,
            "txn_count_5m_log": math.log1p(txn_count_5m),
            "txn_count_30m": txn_count_30m,
            "txn_count_30m_log": math.log1p(txn_count_30m),
            "txn_count_24h": txn_count_24h,
            "txn_count_24h_log": math.log1p(txn_count_24h),
            "distinct_merchants_24h": distinct_merchants_24h,
            "distinct_devices_24h": distinct_devices_24h,
            "amount_sum_24h_log": math.log1p(amount_sum_24h),
            "customer_history_count": n_c,
            "merchant_history_count": n_m,
            "device_history_count": n_d,
            "customer_history_count_log": math.log1p(n_c),
            "merchant_history_count_log": math.log1p(n_m),
            "device_history_count_log": math.log1p(n_d),
            "customer_avg_amount_log": customer_avg,
            "merchant_avg_amount_log": merchant_avg,
            "time_since_customer_txn_log": math.log1p(since_c),
            "time_since_device_txn_log": math.log1p(since_d),
            "new_device": 1 if n_d == 0 else 0,
            "new_merchant": 1 if n_m == 0 else 0,
            "first_time_customer": 1 if n_c == 0 else 0,
            "customer_history_available": 1 if n_c >= self.baseline_min_n else 0,
            "merchant_history_available": 1 if n_m >= self.baseline_min_n else 0,
            "device_history_available": 1 if n_d >= 1 else 0,
            "customer_history_conf": conf,
            "history_stratum": STRATUM_ORD[history_stratum(n_c)],
            "customer_amount_baseline": mu,
            "segment_amount_baseline": mu_seg,
            "global_amount_baseline": global_mean_log,
            "segment_risk_prior": risk_seg,
        }

        # parity columns (used ONLY for the leakage check report)
        if txn.get("_parity"):
            vel8 = len(times) - bisect.bisect_left(times, ts - 480)
            baseline_amt = (c["sum_amt"] / n_c) if n_c >= self.baseline_min_n else (
                (g["sum_amt"] / g["n"]) if g["n"] else amt)
            spike = 1 if (baseline_amt > 0 and amt >= 2.0 * baseline_amt) else 0
            feats["_parity_velocity_8min"] = vel8
            feats["_parity_amount_spike"] = spike
        return feats

    def _segment_lookup(self, stratum: str, tier: str) -> dict:
        segs = self.priors.get("segments", {})
        glob = self.priors.get("global", {})
        g = self.global_
        fallback = {"mean": glob.get("mean", g.get("mean_log", 0.0) or 0.0),
                    "std": glob.get("std", 0.3),
                    "risk": glob.get("risk", 0.0)}
        for key in ((stratum, tier), (stratum, "any"), ("any", tier)):
            if key in segs and segs[key]:
                return segs[key]
        return fallback

    def commit_row(self, txn: dict):
        ts = float(txn["ts_sec"])
        amt = float(txn["amount"])
        log_amt = math.log1p(amt)
        cid = txn["customer_id"]
        mid = txn["merchant_id"]
        did = txn["device_id"]
        self._commit_state(cid, mid, did, amt, log_amt, ts)

    def _commit_state(self, cid, mid, did, amt, log_amt, ts):
        c = self.customers.get(cid)
        if c is None:
            c = self._empty_customer()
            self.customers[cid] = c
        c["n"] += 1
        c["sum_amt"] += amt
        c["wf"].push(log_amt)
        c["times"].append(ts)
        c["pairs"].append((ts, mid, did, amt))
        if c["first_seen"] is None:
            c["first_seen"] = ts
        c["last_seen"] = ts

        m = self.merchants.get(mid)
        if m is None:
            m = self._empty_merchant()
            self.merchants[mid] = m
        m["n"] += 1
        m["sum_amt"] += amt
        m["wf"].push(log_amt)
        if m["first_seen"] is None:
            m["first_seen"] = ts
        m["last_seen"] = ts

        d = self.devices.get(did)
        if d is None:
            d = self._empty_device()
            self.devices[did] = d
        d["n"] += 1
        if d["first_seen"] is None:
            d["first_seen"] = ts
        d["last_seen"] = ts

        g = self.global_
        g["n"] += 1
        g["sum_amt"] += amt
        g["n_log"] += 1
        g["sum_log"] += log_amt
        g["mean_amt"] = g["sum_amt"] / g["n"]
        g["mean_log"] = g["sum_log"] / g["n"]

    # -------------------------------------------------------------- replay
    def replay(self, df: pd.DataFrame, with_parity: bool = False) -> pd.DataFrame:
        """Process a chronologically sorted frame; return feature DataFrame.

        One code path: identical to calling ``prepare_row`` / ``commit_row``
        per transaction (used by the streaming processor).
        """
        if not df.empty:
            ts_order = df["ts_sec"].to_numpy()
            if np.any(np.diff(ts_order) < 0):
                raise ValueError("FeatureEngine.replay requires a "
                                 "chronologically sorted frame.")

        records = []
        for row in df.itertuples(index=False):
            txn = {
                "ts_sec": float(row.ts_sec),
                "amount": float(row.amount),
                "customer_id": row.customer_id,
                "merchant_id": row.merchant_id,
                "device_id": row.device_id,
                "transaction_hour": int(row.transaction_hour),
                "_parity": with_parity,
            }
            feats = self.prepare_row(txn)
            out = {
                k: (getattr(row, k) if hasattr(row, k) else None)
                for k in ROUGH_PASSTHROUGH
            }
            out.update(feats)
            records.append(out)
            self.commit_row(txn)

        frame = pd.DataFrame(records)
        for col in MODEL_FEATURES:
            if col in frame and not pd.api.types.is_numeric_dtype(frame[col]):
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
        return frame


def feature_cols() -> list[str]:
    return list(MODEL_FEATURES)


def novelty_cols() -> list[str]:
    return list(NOVELTY_FEATURES)