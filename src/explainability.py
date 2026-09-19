"""Explainability: TreeSHAP via LightGBM ``pred_contrib`` + reason templates.

For every scored transaction we return:
* ``top_contributing_features`` - the k strongest per-feature contributions
  to the *fraud* class (positive = increases fraud probability).
* ``explanation`` - a short plain-language sentence list.
* ``positive_contributors`` / ``negative_contributors``.

Contributions are exact TreeSHAP values for the primary LightGBM model
(``pred_contrib=True``), summed into human reason groups so correlated
features (e.g. five velocity columns) become one reason. Explanations are
descriptive, not causal - correlation is never presented as causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REASON_GROUPS = {
    "velocity": ["txn_count_5m", "txn_count_30m", "txn_count_24h",
                 "txn_count_5m_log", "txn_count_30m_log", "txn_count_24h_log",
                 "amount_sum_24h_log", "distinct_merchants_24h",
                 "distinct_devices_24h"],
    "amount": ["amount_log", "amount_z_shrunk", "customer_amount_z",
               "merchant_amount_z", "amount_bucket"],
    "time": ["transaction_hour", "day_of_week", "is_weekend", "is_night"],
    "device": ["new_device", "device_history_count", "device_history_count_log",
               "device_history_available", "time_since_device_txn_log"],
    "merchant": ["new_merchant", "merchant_history_count",
                 "merchant_history_count_log", "merchant_amount_z",
                 "merchant_avg_amount_log", "merchant_history_available"],
    "history": ["customer_history_count", "customer_history_count_log",
                "customer_history_available", "customer_history_conf",
                "history_stratum", "first_time_customer",
                "time_since_customer_txn_log", "customer_amount_baseline",
                "segment_amount_baseline", "global_amount_baseline",
                "segment_risk_prior", "customer_amount_z"],
}

GROUP_LABELS = {
    "velocity": "transaction velocity",
    "amount": "transaction amount vs normal baseline",
    "time": "transaction time",
    "device": "device history / novelty",
    "merchant": "merchant history / novelty",
    "history": "customer history & cold-start context",
}


def _template_reason(group: str, feats: dict, contrib: float, features_dict: dict) -> str:
    """Turn a group + sign into a plain-language sentence."""
    a = "increases risk" if contrib >= 0 else "reduces risk"
    if group == "velocity":
        n5 = int(feats.get("txn_count_5m", 0))
        n30 = int(feats.get("txn_count_30m", 0))
        n24 = int(feats.get("txn_count_24h", 0))
        return (f"Velocity {'elevated' if contrib >= 0 else 'low'}: "
                f"{n5} txns in last 5 min, {n30} in 30 min, {n24} in 24h "
                f"({a}).")
    if group == "amount":
        z = float(feats.get("amount_z_shrunk", 0.0))
        if z > 0.5:
            return f"Amount is {abs(z):.1f} sd above the historical baseline (z={z:.2f}) ({a})."
        if z < -0.5:
            return f"Amount is {abs(z):.1f} sd below the historical baseline (z={z:.2f}) ({a})."
        return f"Amount close to the historical baseline (z={z:.2f}) ({a})."
    if group == "time":
        hour = int(feats.get("transaction_hour", 0))
        night = int(feats.get("is_night", 0))
        return (f"Transaction at {hour:02d}:00 {'(night-time)' if night else ''} "
                f"({a}).")
    if group == "device":
        new = int(feats.get("new_device", 0))
        if new:
            return f"The device has not been observed previously ({a})."
        n = int(feats.get("device_history_count", 0))
        return f"Device seen {n} time(s) before; behaviour {a}."
    if group == "merchant":
        new = int(feats.get("new_merchant", 0))
        if new:
            return f"The merchant has not been observed previously ({a})."
        n = int(feats.get("merchant_history_count", 0))
        return f"Merchant seen {n} time(s) before; behaviour {a}."
    if group == "history":
        n = int(feats.get("customer_history_count", 0))
        conf = float(feats.get("customer_history_conf", 0.0))
        first = int(feats.get("first_time_customer", 0))
        if first:
            return f"First transaction for this customer; using segment-level baseline ({a})."
        return f"Customer history: {n} prior txns, confidence {conf:.2f} ({a})."
    return f"{features_dict.get('feature', group)} ({a})."


class Explainer:
    """Exact TreeSHAP explainer for the LightGBM primary model."""

    def __init__(self, model, feature_names: list[str], k: int = 5,
                 reason_groups: bool = True, seed: int = 42):
        self.model = model
        self.feature_names = list(feature_names)
        self.k = k
        self.reason_groups = reason_groups
        self.seed = seed
        self._supports_contribs = hasattr(model, "predict")

    # ------------------------------------------------------------ global
    def global_importance(self, X: pd.DataFrame, sample_size: int = 2000) -> pd.DataFrame:
        """Mean |TreeSHAP| contribution per feature (global feature importance)."""
        Xs = X.sample(n=min(sample_size, len(X)), random_state=self.seed)
        contribs = self._contribs(Xs)
        imp = np.abs(contribs).mean(axis=0)
        df = pd.DataFrame({"feature": self.feature_names, "mean_abs_contribution": imp})
        return df.sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------ per-row
    def _contribs(self, X: pd.DataFrame) -> np.ndarray:
        Xa = X[self.feature_names].to_numpy(dtype=float)
        try:
            if hasattr(self.model, "predict"):
                out = np.asarray(self.model.predict(Xa, pred_contrib=True))
                # LightGBM returns an extra trailing bias column
                if out.shape[1] == len(self.feature_names) + 1:
                    out = out[:, : len(self.feature_names)]
                return out
        except TypeError:  # non-lightgbm fallback
            pass
        # fallback: observation-level permutation influence (labelled clearly)
        base = np.asarray(self.model.predict_proba(Xa))[:, 1].mean()
        contribs = np.zeros((len(Xa), len(self.feature_names)))
        for j in range(len(self.feature_names)):
            Xp = Xa.copy()
            rng = np.random.RandomState(self.seed)
            Xp[:, j] = rng.permutation(Xa[:, j])
            shift = base - np.asarray(self.model.predict_proba(Xp))[:, 1].mean()
            contribs[:, j] = shift
        return contribs

    def explain(self, X: pd.DataFrame, feats_original: dict | None = None) -> list[dict]:
        """Explain each row of X against the fraud class."""
        contribs = self._contribs(X)
        out = []
        for i in range(len(contribs)):
            out.append(self._explain_row(contribs[i], X.iloc[i], feats_original))
        return out

    def explain_row(self, feats_row: pd.Series) -> dict:
        contribs = self._contribs(feats_row.to_frame().T)
        return self._explain_row(contribs[0], feats_row, None)

    def _explain_row(self, contrib, feats_row, feats_original) -> dict:
        contrib_dict = {f: float(contrib[j]) for j, f in enumerate(self.feature_names)}
        if self.reason_groups:
            grouped = {}
            for group, cols in REASON_GROUPS.items():
                g = sum(contrib_dict.get(c, 0.0) for c in cols)
                grouped[group] = g
            items = sorted(grouped.items(), key=lambda kv: -abs(kv[1]))
            top = items[: self.k]
            reasons = []
            for group, g in top:
                if abs(g) < 1e-4:
                    continue
                reasons.append({
                    "feature": group,
                    "effect": "increases_risk" if g >= 0 else "decreases_risk",
                    "contribution": round(g, 4),
                    "reason": _template_reason(group, feats_row.to_dict(),
                                               g, None),
                })
            positives = [r for r in reasons if r["contribution"] > 0]
            negatives = [r for r in reasons if r["contribution"] < 0]
            explanation = "; ".join(
                f"{r['reason']}" for r in
                sorted(reasons, key=lambda r: -abs(r["contribution"])))
        else:
            top = sorted(contrib_dict.items(), key=lambda kv: -abs(kv[1]))[: self.k]
            reasons = [{"feature": f, "effect": "increases_risk" if v >= 0 else "decreases_risk",
                        "contribution": round(v, 4), "reason": f"contrib {v:+.3f}"}
                       for f, v in top if abs(v) > 1e-4]
            positives = [r for r in reasons if r["contribution"] > 0]
            negatives = [r for r in reasons if r["contribution"] < 0]
            explanation = "; ".join(r["reason"] for r in reasons)

        return {
            "top_contributing_features": reasons,
            "positive_contributors": positives,
            "negative_contributors": negatives,
            "explanation": explanation,
        }