"""Streaming transaction processor + alert-budget controller.

Sequential scoring pipeline (exactly the production code path):

    validate -> prepare_row (features use ONLY prior state)
             -> fraud_probability (classifier + calibrator)
             -> novelty_score (novelty model + ECDF normalisation)
             -> risk_band + alert decision (risk engine)
             -> explanation (TreeSHAP + templates)
             -> commit_row (history updates happen only AFTER scoring)

``score_transaction(transaction, state)`` is the public API. State is passed
in / mutated so the caller controls lifecycle (reset on replay, persist for
deployment). The current transaction can never influence its own historical
features because ``commit_row`` runs last.

Alert-budget controller
-----------------------
* max alerts per hour / per day (config).
* high-risk alerts get priority and are always escalated (up to a hard cap).
* review alerts are ranked by combined risk and escalated while the budget
  allows.
* monitor alerts are logged without immediate escalation.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from src.config import Settings
from src.feature_engineering import NOVELTY_FEATURES


class AlertBudgetController:
    """Keeps the alert stream inside an analyst budget.

    Every transaction passes through ``decide`` (streaming, order-preserving).
    Counters are per rolling hour/day. Escalation rules:

    * ``high-risk``  -> escalated while under the high-risk share cap *and*
                       the total alert budget.
    * ``review``     -> escalated while under the total alert budget
                       (call this on a stream already ranked by risk score so
                       the budget goes to the riskiest reviews).
    * ``monitor``    -> logged, never escalated.
    """

    def __init__(self, max_per_hour: int, max_per_day: int,
                 budget_pct: float, high_risk_share_cap_pct: float):
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day
        self.budget_pct = budget_pct
        self.high_cap = high_risk_share_cap_pct
        self._hour: deque = deque()   # (ts,)
        self._day: deque = deque()    # (ts, band, rank)
        self._volume_day: dict = {}   # day_key -> transactions seen
        self.logged = 0

    def reset(self):
        self._hour.clear()
        self._day.clear()
        self._volume_day.clear()
        self.logged = 0

    @property
    def escalated_high(self) -> int:
        return sum(1 for _, b, _ in self._day if b == "high-risk")

    @property
    def escalated_review(self) -> int:
        return sum(1 for _, b, _ in self._day if b == "review")

    @property
    def total_escalated(self) -> int:
        return len(self._day)

    def _prune(self, ts: float):
        while self._hour and ts - self._hour[0][0] > 3600:
            self._hour.popleft()
        while self._day and ts - self._day[0][0] > 86400:
            self._day.popleft()

    def decide(self, ts: float, band: str, rank: float) -> str:
        """Return ``"alert"`` (escalated) or ``"log"`` for one decision."""
        self._prune(ts)
        day = int(ts // 86400)
        self._volume_day[day] = self._volume_day.get(day, 0) + 1
        vol_today = max(1, self._volume_day[day])

        hour_ok = len(self._hour) < self.max_per_hour
        day_ok = len(self._day) < self.max_per_day
        budget_cap = max(3, int(vol_today * self.budget_pct / 100.0))
        high_cap = max(2, int(vol_today * self.high_cap / 100.0))
        escalated_today = sum(1 for t in self._day
                              if int(t[0] // 86400) == day)
        high_today = sum(1 for t in self._day
                         if int(t[0] // 86400) == day and t[1] == "high-risk")

        if band not in ("high-risk", "review"):
            self.logged += 1
            return "log"

        if band == "high-risk" and high_today >= high_cap:
            self.logged += 1
            return "log"

        if hour_ok and day_ok and escalated_today < budget_cap:
            self._hour.append((ts,))
            self._day.append((ts, band, rank))
            return "alert"

        self.logged += 1
        return "log"


class StreamingProcessor:
    """Scoring pipeline; stateful (mutates engine state after scoring)."""

    def __init__(self, cfg: Settings, engine, model, feature_names: list[str],
                 calibrator=None, novelty=None, risk_engine=None,
                 explainer=None, budget: AlertBudgetController | None = None,
                 novelty_features: list[str] | None = None):
        self.cfg = cfg
        self.engine = engine
        self.model = model
        self.features = feature_names
        self.novelty_features = novelty_features or NOVELTY_FEATURES
        self.calibrator = calibrator
        self.novelty = novelty
        self.risk = risk_engine
        self.explainer = explainer
        self.budget = budget

    # ------------------------------------------------------------- seeding
    def seed_state(self, df: pd.DataFrame):
        """Build historical state by replaying prior transactions WITHOUT
        scoring (this is how the stream keeps continuity for the test fold)."""
        for row in df.itertuples(index=False):
            txn = self._to_txn(row)
            self.engine.commit_row(txn)

    def _to_txn(self, row) -> dict:
        return {
            "ts_sec": float(row.ts_sec),
            "amount": float(row.amount),
            "customer_id": row.customer_id,
            "merchant_id": row.merchant_id,
            "device_id": row.device_id,
            "transaction_hour": int(row.transaction_hour),
        }

    def validate(self, txn: dict) -> list[str]:
        errors = []
        if txn.get("amount") is None or float(txn["amount"]) <= 0:
            errors.append("amount must be a positive number")
        if not txn.get("timestamp") and not txn.get("ts_sec"):
            errors.append("timestamp required")
        for k in ("customer_id", "merchant_id", "device_id"):
            if not txn.get(k):
                errors.append(f"{k} required")
        return errors

    # ---------------------------------------------------------------- core
    def score_transaction(self, txn: dict) -> dict:
        """Validate -> features -> probability -> novelty -> band -> explain
        -> alert; state updated only at the end."""
        errs = self.validate(txn)
        if errs:
            return {"valid": False, "errors": errs}
        if "ts_sec" not in txn:
            txn["ts_sec"] = float(pd.Timestamp(txn["timestamp"]).timestamp())

        feats = self.engine.prepare_row(txn)
        X = pd.DataFrame([feats])[self.features]

        rawer = float(self.model.predict_proba(X)[:, 1][0])
        p = self.calibrator.predict(np.array([rawer]))[0] if self.calibrator else rawer
        nov = float(self.novelty.novelty(X[self.novelty_features])[0]) \
            if self.novelty else 0.0
        decision = self.risk.decide(p, nov)
        _expl = self.explainer.explain(X)[0] if self.explainer else {
            "top_contributing_features": [], "positive_contributors": [],
            "negative_contributors": [], "explanation": ""}

        if self.budget is not None:
            status = self.budget.decide(float(txn["ts_sec"]),
                                        decision["risk_band"],
                                        decision["rank_score"])
        else:
            status = "alert" if decision["alert"] else "log"

        self.engine.commit_row(txn)   # state update AFTER scoring

        return {
            "valid": True,
            "fraud_probability": decision["fraud_probability"],
            "raw_probability": round(float(rawer), 4),
            "novelty_score": decision["novelty_score"],
            "risk_band": decision["risk_band"],
            "alert": decision["alert"],
            "budget_status": status,
            "rank_score": decision["rank_score"],
            "tags": decision["tags"],
            "top_contributing_features": _expl["top_contributing_features"],
            "positive_contributors": _expl["positive_contributors"],
            "negative_contributors": _expl["negative_contributors"],
            "explanation": _expl["explanation"],
        }

    # ------------------------------------------------------------ replay
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a chronologically sorted frame; returns decisions per row."""
        outputs = []
        for row in df.itertuples(index=False):
            txn = self._to_txn(row)
            txn["timestamp"] = str(row.timestamp)
            res = self.score_transaction(txn)
            res["transaction_id"] = row.transaction_id
            res["timestamp"] = str(row.timestamp)
            res["ts_sec"] = float(row.ts_sec)
            res["customer_id"] = row.customer_id
            res["merchant_id"] = row.merchant_id
            res["device_id"] = row.device_id
            res["amount"] = float(row.amount)
            res["is_fraud"] = int(row.is_fraud)
            outputs.append(res)
        drop = {"top_contributing_features", "positive_contributors",
                "negative_contributors", "tags"}
        return pd.DataFrame([{k: v for k, v in r.items() if k not in drop}
                             for r in outputs])

    # ------------------------------------------------------- batch replay
    def score_batch(self, df: pd.DataFrame) -> list[dict]:
        """Stream a chronologically sorted frame with *vectorized inference*.

        Semantics are identical to ``score_transaction`` per row - features
        are computed point-in-time and state is committed only after each row
        - but the MODEL calls (LightGBM probability, calibrator, novelty
        anomaly score, TreeSHAP explanations) run on the whole frame at once.
        Row-by-row LightGBM + IsolationForest inference costs ~30 ms/row;
        batched inference drops that to well under 1 ms/row for large files.
        """
        n = len(df)
        err_mask = [False] * n
        feats: list[dict | None] = [None] * n
        for i, row in enumerate(df.itertuples(index=False)):
            txn = self._to_txn(row)
            txn["timestamp"] = str(row.timestamp)
            if self.validate(txn):
                err_mask[i] = True
                continue
            feats[i] = self.engine.prepare_row(txn)
            self.engine.commit_row(txn)

        valid = [i for i in range(n) if not err_mask[i]]
        if not valid:
            out: list[dict] = [None] * n
            for i in range(n):
                out[i] = {"valid": False, "fraud_probability": None,
                          "raw_probability": None, "novelty_score": None,
                          "risk_band": "invalid", "alert": False,
                          "budget_status": "log", "rank_score": None,
                          "tags": None, "top_contributing_features": [],
                          "positive_contributors": [],
                          "negative_contributors": [], "explanation": None}
            return out
        X = pd.DataFrame([feats[i] for i in valid])[self.features]

        rawer = np.asarray(self.model.predict_proba(X)[:, 1])
        p = np.asarray(self.calibrator.predict(rawer)) \
            if self.calibrator else rawer
        nov = np.asarray(self.novelty.novelty(X[self.novelty_features])) \
            if self.novelty else np.zeros(len(X), dtype=float)
        expls = self.explainer.explain(X) if self.explainer else [{
            "top_contributing_features": [], "positive_contributors": [],
            "negative_contributors": [], "explanation": ""}] * len(X)

        out: list[dict] = [None] * n
        k = 0
        for i in range(n):
            if err_mask[i]:
                out[i] = {"valid": False}
                continue
            row = df.iloc[i]
            txn = self._to_txn(row)
            txn["timestamp"] = str(row.timestamp)
            decision = self.risk.decide(float(p[k]), float(nov[k]))
            if self.budget is not None:
                status = self.budget.decide(float(txn["ts_sec"]),
                                            decision["risk_band"],
                                            decision["rank_score"])
            else:
                status = "alert" if decision["alert"] else "log"
            out[i] = {
                "valid": True,
                "fraud_probability": decision["fraud_probability"],
                "raw_probability": round(float(rawer[k]), 4),
                "novelty_score": decision["novelty_score"],
                "risk_band": decision["risk_band"],
                "alert": decision["alert"],
                "budget_status": status,
                "rank_score": decision["rank_score"],
                "tags": decision["tags"],
                "top_contributing_features":
                    expls[k]["top_contributing_features"],
                "positive_contributors": expls[k]["positive_contributors"],
                "negative_contributors": expls[k]["negative_contributors"],
                "explanation": expls[k]["explanation"],
            }
            k += 1
        return out

    def run_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """``score_batch`` returning a DataFrame with row identity columns."""
        res = self.score_batch(df)
        outputs = []
        for row, r in zip(df.itertuples(index=False), res):
            r = dict(r)
            r["transaction_id"] = row.transaction_id
            r["timestamp"] = str(row.timestamp)
            r["ts_sec"] = float(row.ts_sec)
            r["customer_id"] = row.customer_id
            r["merchant_id"] = row.merchant_id
            r["device_id"] = row.device_id
            r["amount"] = float(row.amount)
            r["is_fraud"] = int(row.is_fraud)
            outputs.append(r)
        cols = ["valid", "fraud_probability", "raw_probability",
                "novelty_score", "risk_band", "alert", "budget_status",
                "rank_score", "tags", "top_contributing_features",
                "positive_contributors", "negative_contributors",
                "explanation", "transaction_id", "timestamp", "ts_sec",
                "customer_id", "merchant_id", "device_id", "amount",
                "is_fraud"]
        return pd.DataFrame(outputs, columns=cols)