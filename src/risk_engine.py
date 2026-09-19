"""Combined risk decision engine.

Combines the two independent signals - fraud probability and behavioural
novelty - into a risk band and an alert decision using a configurable
decision matrix:

                  novelty low        novelty high
    p low         normal             monitor
    p high        review             high-risk

* ``high-risk``  -> always alert (recommendation for hold / analyst review).
* ``review``     -> alert (probability channel or novelty channel).
* ``monitor``    -> logged, no immediate escalation.
* ``normal``     -> allow.

Thresholds are selected on VALIDATION data (``fit_thresholds``) and stored
with the run; they are documented, never hardcoded, and never tuned on test.

Novelty alone never escalates: the novelty channel to ``review`` requires
``p >= p_floor`` as well, so a new-but-ordinary customer is *not* penalised
for being new.
"""
from __future__ import annotations

import numpy as np

from src.config import Settings

RISK_BANDS = ["normal", "monitor", "review", "high-risk"]
BAND_ORDER = {b: i for i, b in enumerate(RISK_BANDS)}


class RiskEngine:
    def __init__(self, thresholds: dict | None = None):
        self.t = thresholds or self.default_thresholds()

    @staticmethod
    def default_thresholds() -> dict:
        return {"p_review": 0.15, "p_high": 0.45, "novelty_high": 0.90,
                "novelty_very_high": 0.97, "p_floor_for_novelty": 0.02}

    def decide(self, p: float, novelty: float) -> dict:
        """Return the decision object for one transaction."""
        t = self.t
        p = float(np.clip(p, 0.0, 1.0))
        nov = float(np.clip(novelty, 0.0, 1.0))

        high_p = p >= t["p_high"]
        review_p = p >= t["p_review"]
        high_nov = nov >= t["novelty_very_high"] and p >= t["p_floor_for_novelty"]
        monitor = nov >= t["novelty_high"]

        if high_p:
            band = "high-risk"
        elif review_p or high_nov:
            band = "review"
        elif monitor:
            band = "monitor"
        else:
            band = "normal"

        alert = band in ("review", "high-risk")
        rank = self.rank_score(p, nov)
        return {
            "fraud_probability": round(p, 4),
            "novelty_score": round(nov, 4),
            "risk_band": band,
            "alert": bool(alert),
            "rank_score": round(rank, 6),
            "tags": self._tags(band, p, nov),
        }

    def _tags(self, band, p, nov):
        tags = []
        if band == "high-risk":
            tags.append("high_probability" if p >= self.t["p_high"] else "high_novelty")
        if nov >= self.t["novelty_high"] and band != "normal":
            tags.append("novel_pattern")
        return tags

    def rank_score(self, p: float, novelty: float) -> float:
        """Operational priority: probability dominates, novelty breaks ties."""
        return 0.7 * float(p) + 0.3 * float(novelty)


def fit_thresholds(val_p: np.ndarray, val_nov: np.ndarray,
                   cfg: Settings) -> dict:
    """Select probability thresholds from VALIDATION data such that the
    volume of each channel matches the configured budgets.

    - p_high   : the quantile leaving ~high_risk_share_cap_pct above it.
    - p_review : the quantile leaving ~alert_budget_pct above it.
    Novelty thresholds stay configurable constants (they bound the
    monitoring channel, which never alerts by itself).
    """
    budget = float(cfg.risk["alert_budget_pct"]) / 100.0
    high_cap = float(cfg.risk["high_risk_share_cap_pct"]) / 100.0
    p_high = float(np.quantile(val_p, 1.0 - high_cap))
    p_review = float(np.quantile(val_p, max(0.0, 1.0 - budget)))
    thresholds = {
        "p_review": round(max(p_review, 0.001), 4),
        "p_high": round(max(p_high, p_review + 0.001, 0.01), 4),
        "novelty_high": float(cfg.risk["novelty_high"]),
        "novelty_very_high": float(cfg.risk["novelty_very_high"]),
        "p_floor_for_novelty": float(cfg.risk["p_floor_for_novelty"]),
    }
    return thresholds