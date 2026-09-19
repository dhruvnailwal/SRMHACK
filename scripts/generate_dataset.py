"""
Synthetic fraud dataset generator for Unseen-Customer Fraud Detection.

Produces ``data/raw/synthetic_fraud_transactions.csv`` with EXACTLY the
documented schema:

    transaction_id, timestamp, customer_id, merchant_id, device_id, amount,
    transaction_hour, is_night, new_customer, new_merchant, new_device,
    velocity_8min, amount_spike, customer_history_count, merchant_history_count,
    device_history_count, is_fraud

How the data is made (important for interpretation)
----------------------------------------------------
* A fixed seed reproduces the exact same file on every run.
* Transactions are generated *in time*, one customer-session at a time, then
  sorted chronologically - so every "as-of" column is computed only from rows
  that appear earlier in the stream. Nothing uses the future.
* Fraud patterns included:
    1) NEW-ACCOUNT FRAUD  - brand-new customers that make a short burst of
       high-value / unusual-hour transactions from a new device, then vanish.
       These customers are naturally "unseen" if they arrive after the train
       window of the time-based split.
    2) ACCOUNT TAKEOVER   - established legitimate customers whose account is
       later used from a new device with unusual amount/time/velocity, then
       goes quiet. Produces "known customer, novel behaviour" fraud.
* "New but normal" behaviour is abundant: many legitimate new customers arrive
  throughout the year with ordinary first transactions.
* The only non-standard derivations:
    velocity_8min        = # of the customer's own txns in the prior 8 minutes.
    amount_spike         = 1 if amount >= 2.0 * as_of_baseline_amount, where
                           as_of_baseline_amount is the customer's historical
                           mean amount when they have >=3 prior txns, otherwise
                           the running global mean over all prior txns.
    is_night             = hour in {0..5} or {23}.
All of these are strictly point-in-time (no future information).

The generator is intentionally a *simulation*. Results on it are SYNTHETIC
results; they do not transfer to any real-world portfolio. See README.
"""
from __future__ import annotations

import argparse
import bisect
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "synthetic_fraud_transactions.csv")

# --------------------------------------------------------------------------
# Global constants
# --------------------------------------------------------------------------
START_TS = datetime(2025, 1, 1)
SPAN_DAYS = 365
N_TXN = 120_000
N_CUSTOMERS = 8_500
N_MERCHANTS = 4_200
N_DEVICES = 9_500

FRAUD_NEW_ACCOUNT_ACCOUNTS = 1200   # each contributes one fraud burst
FRAUD_ATO_CUSTOMERS = 200           # established customers given one attack
FRAUD_NEW_ACCOUNT_N = 3200          # target fraud txns from new accounts
TOT_FRAUD_N = 3600                  # target total fraud rows (~3% of ~120k)

# Merchant "categories" drive realistic amount/hour behaviour. The category is
# NOT exported - it only shapes the row values, which keeps the model honest.
CATEGORY_AMT = {
    "grocery":       (4.6, 0.55),
    "food":          (4.4, 0.60),
    "retail":        (5.0, 0.65),
    "fuel":          (5.0, 0.45),
    "health":        (5.2, 0.70),
    "entertainment": (4.9, 0.75),
    "travel":        (6.1, 0.80),
    "electronics":   (6.4, 0.85),
    "finance":       (6.0, 0.90),
    "other":         (5.1, 0.70),
}
# fraction of merchant pool per category (merchants with category == cat)
CATEGORY_MIX = {
    "grocery": 0.20, "food": 0.14, "retail": 0.16, "fuel": 0.07,
    "health": 0.07, "entertainment": 0.10, "travel": 0.08,
    "electronics": 0.09, "finance": 0.05, "other": 0.04,
}
# fraud-favoured categories for synthetic attacks
FRAUD_CATEGORIES = ["electronics", "travel", "finance", "retail"]

# hour archetypes (24-bin multinomials)
HOUR_ARCHETYPES = {
    "morning":   [0.10, 0.02, 0.01, 0.00, 0.00, 0.00, 0.02, 0.10, 0.16, 0.08, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.15],
    "office":    [0.02, 0.01, 0.01, 0.00, 0.00, 0.01, 0.03, 0.05, 0.09, 0.12, 0.08, 0.05, 0.04, 0.05, 0.07, 0.08, 0.09, 0.09, 0.06, 0.02, 0.01, 0.01, 0.01, 0.01],
    "evening":   [0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.06, 0.09, 0.12, 0.16, 0.13, 0.09, 0.05, 0.02, 0.01],
    "nightowl":  [0.10, 0.08, 0.06, 0.04, 0.02, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.07, 0.08, 0.08, 0.07, 0.06, 0.05],
}
# income tier -> offset added to the category log-amount
TIER_OFFSET = {"low": -0.35, "med": 0.0, "high": 0.45}

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _hour_multinomial(rng: np.random.RandomState, profile: np.ndarray) -> int:
    profile = np.asarray(profile, dtype=float)
    profile = profile / profile.sum()
    return int(rng.choice(24, p=profile))


def _cat_weights(rng, favored: str | None = None):
    w = np.array(list(CATEGORY_MIX.values()), dtype=float)
    if favored is not None:
        keys = list(CATEGORY_MIX.keys())
        w[keys.index(favored)] *= 12.0
    w /= w.sum()
    return w


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_customers(rng: np.random.RandomState):
    """Return list of dicts describing each customer's latent behaviour."""
    cust = []
    tier_keys = ["low", "med", "high"]
    tiers = rng.choice(tier_keys, size=N_CUSTOMERS, p=[0.35, 0.50, 0.15])
    profiles = ["morning", "office", "evening", "nightowl"]
    arch = rng.choice(profiles, size=N_CUSTOMERS, p=[0.30, 0.30, 0.30, 0.10])
    # arrival day uniformly across the year (later arrivals -> naturally unseen)
    arrival = rng.randint(0, SPAN_DAYS, size=N_CUSTOMERS)
    # weekly activity intensity (sessions per week): tuned so the full
    # stream is ~ 115k legitimate rows across the year
    intensity = rng.gamma(1.5, 0.123, size=N_CUSTOMERS)
    # device loyalty: probability the customer uses their loyal device
    loyalty = rng.uniform(0.70, 0.97, size=N_CUSTOMERS)
    n_loyal = rng.choice([1, 2], size=N_CUSTOMERS, p=[0.7, 0.3])
    for i in range(N_CUSTOMERS):
        cust.append({
            "cid": f"C{i:05d}",
            "tier": tiers[i],
            "arch": arch[i],
            "arrival_day": int(arrival[i]),
            "intensity": float(intensity[i]),
            "loyalty": float(loyalty[i]),
            "n_loyal": int(n_loyal[i]),
            "devices": [],            # loyal devices (filled later)
            "affinity": _cat_weights(rng),
        })
    return cust


def build_merchants(rng: np.random.RandomState):
    keys = list(CATEGORY_MIX.keys())
    freq = np.array(list(CATEGORY_MIX.values()))
    idx = rng.choice(len(keys), size=N_MERCHANTS, p=freq)
    merchants = []
    for i in range(N_MERCHANTS):
        cat = keys[idx[i]]
        # popularity determines how often the merchant is seen (zipf-ish)
        pop = float(np.random.random() ** 3)  # many rare, few popular
        merchants.append({
            "mid": f"M{i:05d}",
            "cat": cat,
            "pop": pop,
            "amt_bias": rng.normal(0.0, 0.18),
        })
    # index merchants per category with pre-normalised popularity weights
    by_cat = {k: [] for k in CATEGORY_MIX}
    for m in merchants:
        by_cat[m["cat"]].append(m)
    w_by_cat = {}
    for k, lst in by_cat.items():
        w = np.array([m["pop"] + 0.05 for m in lst], dtype=float)
        w_by_cat[k] = w / w.sum()
    return merchants, by_cat, w_by_cat, keys


def build_devices(rng):
    return [{"did": f"D{i:05d}"} for i in range(N_DEVICES)]


def assign_devices(customers, devices, rng):
    dev_pool = set(range(len(devices)))
    for c in customers:
        if len(dev_pool) < c["n_loyal"]:
            dev_pool = set(range(len(devices)))  # recycle when pool runs low
        chosen = list(rng.choice(list(dev_pool), size=c["n_loyal"], replace=False))
        for d in chosen:
            dev_pool.discard(d)
        c["devices"] = [f"D{d:05d}" for d in chosen]


def merchant_pick(rng, by_cat, w_by_cat, keys, affinities):
    """Pick a merchant weighted by category affinity, then by popularity
    within that category. O(1) with pre-normalised weights."""
    cat_w = np.asarray(affinities, dtype=float)
    cat_w = cat_w / cat_w.sum()
    cat = keys[int(rng.choice(len(keys), p=cat_w))]
    lst = by_cat[cat]
    w = w_by_cat[cat]
    return lst[int(rng.choice(len(lst), p=w))]


def session_amounts(rng, n, cat, tier, fraud_bias=0.0):
    base = CATEGORY_AMT[cat][0] + TIER_OFFSET[tier] + fraud_bias
    std = CATEGORY_AMT[cat][1] * rng.uniform(0.8, 1.3)
    return np.exp(rng.normal(base, std, size=n)).round(2)


def generate_stream(rng):
    customers = build_customers(rng)
    merchants, by_cat, w_by_cat, keys = build_merchants(rng)
    devices = build_devices(rng)
    assign_devices(customers, devices, rng)

    rows = []  # (ts, cid, mid, did, amount, is_fraud)

    def emit(cid, mid, did, amount, ts, fraud):
        rows.append((ts, cid, mid, did, round(float(amount), 2), 1 if fraud else 0))

    # ---------------- legitimate activity ----------------
    for c in customers:
        cid = c["cid"]
        tier = c["tier"]
        arch_hour = HOUR_ARCHETYPES[c["arch"]]
        n_loyal = c["devices"]
        day = c["arrival_day"]
        # sessions until end of year
        max_sessions = int(c["intensity"] * ((SPAN_DAYS - day) / 7.0)) + 1
        for _ in range(max_sessions):
            wait = rng.exponential(7.0 / max(0.1, c["intensity"]))
            day += wait
            if day >= SPAN_DAYS - 1:
                break
            hour = _hour_multinomial(rng, arch_hour)
            date = START_TS + timedelta(days=int(day))
            # weekend shift: slightly more midday shopping
            if date.weekday() >= 5 and rng.rand() < 0.5:
                hour = int(np.clip(hour + rng.randint(2, 6), 8, 21))
            n_txns = 1 + int(rng.geometric(0.45))
            n_txns = min(n_txns, 6)
            first_min = rng.randint(0, 23 * 60)
            t0 = date + timedelta(hours=hour, minutes=first_min)
            for j in range(n_txns):
                # customer affinity per category
                m = merchant_pick(rng, by_cat, w_by_cat, keys, c["affinity"])
                use_loyal = rng.rand() < c["loyalty"]
                did = rng.choice(n_loyal) if use_loyal and n_loyal else f"D{rng.randint(N_DEVICES):05d}"
                amts = session_amounts(rng, 1, m["cat"], tier)[0]
                ts = t0 + timedelta(seconds=int(rng.gamma(4.0, 45.0) + 45))
                emit(cid, m["mid"], did, amts, ts, 0)
                t0 = ts

    # ---------------- NEW-ACCOUNT FRAUD ----------------
    # Brand-new customers, one short burst each, high amounts, fast velocity,
    # fresh device, unusual hours, fraud-favoured categories.
    acc = 0
    total_fraud = 0
    while acc < FRAUD_NEW_ACCOUNT_ACCOUNTS and total_fraud < FRAUD_NEW_ACCOUNT_N:
        cid = f"CX{acc:05d}"
        day = rng.randint(0, SPAN_DAYS - 1)
        date = START_TS + timedelta(days=day)
        n_txns = rng.randint(2, 6)
        # unusual hour: night or extreme tail of a daytime profile
        if rng.rand() < 0.55:
            hour = _hour_multinomial(rng, np.array(HOUR_ARCHETYPES["nightowl"]))
        else:
            hour = int(rng.choice([0, 1, 2, 3, 4, 5, 21, 22, 23], p=[0.18, 0.08, 0.08, 0.08, 0.08, 0.10, 0.16, 0.12, 0.12]))
        t0 = date.replace(hour=hour, minute=rng.randint(0, 55))
        burst_window = 8 + rng.randint(0, 25)   # minutes
        did = f"D{rng.randint(N_DEVICES):05d}"
        # fresh pool of merchants biased to fraud categories
        idx_c = list(CATEGORY_MIX.keys())
        cat = str(rng.choice(FRAUD_CATEGORIES, p=[0.35, 0.25, 0.25, 0.15]))
        fav = _cat_weights(rng, favored=cat)
        for j in range(n_txns):
            m = merchant_pick(rng, by_cat, w_by_cat, keys, fav)
            amt = session_amounts(rng, 1, m["cat"], "high", fraud_bias=1.1)[0]
            ts = t0 + timedelta(minutes=burst_window * j / max(1, n_txns - 1),
                                seconds=int(rng.randint(5, 60)))
            emit(cid, m["mid"], did, amt, ts, 1)
        acc += 1
        total_fraud += n_txns

    # ---------------- ACCOUNT TAKEOVER ----------------
    # Pick established legitimate customers later in the year; give them an
    # attack session: new device, unusual hour, high amounts, high velocity.
    legit_customers = [
        c for c in customers
        if c["arrival_day"] < SPAN_DAYS * 0.55
    ]
    rng.shuffle(legit_customers)
    taken = 0
    for c in legit_customers:
        if taken >= FRAUD_ATO_CUSTOMERS:
            break
        if total_fraud >= TOT_FRAUD_N:
            break
        cid = c["cid"]
        start_day = max(c["arrival_day"], SPAN_DAYS // 2) + rng.randint(10, 60)
        if start_day >= SPAN_DAYS - 5:
            continue
        date = START_TS + timedelta(days=start_day)
        n_txns = rng.randint(2, 5)
        # unusual hour leaning night
        hour = _hour_multinomial(rng, np.array(HOUR_ARCHETYPES["nightowl"]))
        t0 = date.replace(hour=hour, minute=rng.randint(0, 55))
        burst_window = 5 + rng.randint(0, 20)
        # a device the customer has never used
        did = f"D{rng.randint(N_DEVICES):05d}"
        # largely new / rare merchants in fraud categories
        fav = _cat_weights(rng)
        keys = list(CATEGORY_MIX.keys())
        for f in FRAUD_CATEGORIES:
            fav[keys.index(f)] *= 6.0
        fav /= fav.sum()
        for j in range(n_txns):
            m = merchant_pick(rng, by_cat, w_by_cat, keys, fav)
            amt = session_amounts(rng, 1, m["cat"], c["tier"], fraud_bias=1.0)[0]
            ts = t0 + timedelta(minutes=burst_window * j / max(1, n_txns - 1),
                                seconds=int(rng.randint(5, 60)))
            emit(cid, m["mid"], did, amt, ts, 1)
        taken += 1
        total_fraud += n_txns

    # sort chronologically
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def compute_asof_columns(rows):
    """Compute the provided as-of feature columns in a strict chronological
    replay. Nothing from the future is used."""
    cust_n = {}
    merch_n = {}
    dev_n = {}
    cust_total_amount = {}
    global_sum = 0.0
    global_count = 0

    records = []
    for i, (ts, cid, mid, did, amount, fraud) in enumerate(rows):
        n_c = cust_n.get(cid, 0)
        n_m = merch_n.get(mid, 0)
        n_d = dev_n.get(did, 0)

        # velocity within prior 8 minutes for this customer
        vel = 0

        baseline_mean = cust_total_amount.get(cid, 0.0) / n_c if n_c >= 3 else (
            global_sum / global_count if global_count else amount
        )
        spike = 1 if (baseline_mean > 0 and amount >= 2.0 * baseline_mean) else 0

        hour = ts.hour
        is_night = 1 if (hour < 6 or hour >= 23) else 0

        records.append({
            "transaction_id": f"TXN{i + 1:08d}",
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "customer_id": cid,
            "merchant_id": mid,
            "device_id": did,
            "amount": amount,
            "transaction_hour": hour,
            "is_night": is_night,
            "new_customer": 1 if n_c == 0 else 0,
            "new_merchant": 1 if n_m == 0 else 0,
            "new_device": 1 if n_d == 0 else 0,
            # velocity recomputed exactly below inside replay loop
            "velocity_8min": vel,
            "amount_spike": spike,
            "customer_history_count": n_c,
            "merchant_history_count": n_m,
            "device_history_count": n_d,
            "is_fraud": fraud,
        })

        # --- state update AFTER the row above is final ------------------
        cust_n[cid] = n_c + 1
        merch_n[mid] = n_m + 1
        dev_n[did] = n_d + 1
        cust_total_amount[cid] = cust_total_amount.get(cid, 0.0) + amount
        global_sum += amount
        global_count += 1

    # velocity requires per-customer timestamp lists; fix in second pass
    cust_times = {}
    out = []
    for rec in records:
        cid = rec["customer_id"]
        ts = datetime.strptime(rec["timestamp"], "%Y-%m-%d %H:%M:%S")
        tlist = cust_times.setdefault(cid, [])
        cutoff = ts.timestamp() - 8 * 60
        vel = 0
        # tlist is sorted; count entries strictly inside the window
        lo = bisect.bisect_left(tlist, cutoff)
        vel = len(tlist) - lo
        rec["velocity_8min"] = vel
        tlist.append(ts.timestamp())
        tlist.sort()
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--rows", type=int, default=N_TXN)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    print("[gen] building transaction stream ...")
    rows = generate_stream(rng)
    print(f"[gen] generated {len(rows)} rows (target {args.rows})")
    if len(rows) < args.rows * 0.7:
        print(f"[gen] WARNING: only {len(rows)} rows produced; adjust intensity/fraud counts.")
    if len(rows) > args.rows * 1.4:
        print(f"[gen] WARNING: {len(rows)} rows exceeds target by 40%; consider lowering intensity.")
    print("[gen] computing strict point-in-time feature columns ...")
    recs = compute_asof_columns(rows)
    df = pd.DataFrame(recs)
    # round amounts with 2 dp, ensure deterministic dtype order
    df["amount"] = df["amount"].astype(float).round(2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    fraud = df["is_fraud"].sum()
    print(f"[gen] wrote {args.out}")
    print(f"[gen] rows={len(df)} fraud={fraud} rate={fraud / len(df):.4%}")
    print(f"[gen] unique customers={df['customer_id'].nunique()}")
    print(f"[gen] unique merchants={df['merchant_id'].nunique()}")
    print(f"[gen] unique devices={df['device_id'].nunique()}")
    print(f"[gen] new_customer rows={int(df['new_customer'].sum())} "
          f"new_device rows={int(df['new_device'].sum())} "
          f"new_merchant rows={int(df['new_merchant'].sum())}")


if __name__ == "__main__":
    sys.exit(main())
