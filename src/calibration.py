"""Probability calibration.

Converts the raw model score into a well-calibrated ``fraud_probability``.

Two methods are compared on validation data:
* Platt scaling  - logistic regression on the raw score (monotone, smooth).
* Isotonic regression - non-parametric monotone map (more flexible, risks
  overfitting; we fit on a calibration slice and validate on the rest).

Selection is by Brier score on the held-out half of the validation fold.
The output includes calibration curves / reliability-diagram data.
"""
from __future__ import annotations

import numpy as np

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.config import Settings


class Calibrator:
    def __init__(self, method: str = "isotonic", seed: int = 42):
        self.method = method
        self.seed = seed
        self._platt: LogisticRegression | None = None
        self._iso: IsotonicRegression | None = None
        self.brier = None
        self.used_method = method

    def fit(self, raw_score: np.ndarray, y: np.ndarray) -> "Calibrator":
        self._platt = LogisticRegression(C=1.0, max_iter=2000, random_state=self.seed)
        self._platt.fit(raw_score.reshape(-1, 1), y)
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(raw_score, y)
        return self

    def predict(self, raw_score: np.ndarray) -> np.ndarray:
        raw_score = np.asarray(raw_score, dtype=float)
        if self.method == "platt":
            return self._platt.predict_proba(raw_score.reshape(-1, 1))[:, 1]
        return self._iso.predict(raw_score)


def fit_best(raw, y, cfg: Settings | None = None) -> tuple[Calibrator, dict]:
    """Split validation into fit / eval halves, try both methods, return the
    best by Brier along with a comparison report."""
    rng = np.random.RandomState((cfg.seed if cfg else 42))
    n = len(raw)
    idx = rng.permutation(n)
    half = n // 2
    if half < 200:
        half = max(50, int(n * 0.5))
    fit_i = idx[:half]
    eval_i = idx[half:]

    report = {"methods": {}}
    best_cal, best_brier = None, np.inf
    for method in ["platt", "isotonic"]:
        cal = Calibrator(method=method)
        cal.fit(raw[fit_i], y[fit_i])
        p = cal.predict(raw[eval_i])
        b = float(brier_score_loss(y[eval_i], p))
        report["methods"][method] = {"brier": round(b, 5),
                                     "eval_n": int(len(eval_i))}
        if b < best_brier:
            best_brier, best_cal = b, cal
            best_cal.used_method = method
    report["selected"] = best_cal.used_method
    report["selected_brier"] = round(best_brier, 5)
    best_cal.brier = best_brier
    return best_cal, report


def calibration_curve_data(y_true, p_pred, bins: int = 10) -> dict:
    """Equal-width bins: predicted mean vs observed fraction (reliability
    diagram data)."""
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    edges = np.linspace(0.0, 1.0, bins + 1)
    pred_mean, obs_mean, counts, ece = [], [], [], 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p_pred >= lo) & (p_pred <= hi) if i == bins - 1 else \
            (p_pred >= lo) & (p_pred < hi)
        c = int(m.sum())
        counts.append(c)
        if c:
            pm = float(p_pred[m].mean())
            om = float(y_true[m].mean())
            pred_mean.append(pm)
            obs_mean.append(om)
            ece += c / max(1, len(y_true)) * abs(pm - om)
        else:
            pred_mean.append(float("nan"))
            obs_mean.append(float("nan"))
    return {"predicted_mean": pred_mean, "observed_mean": obs_mean,
            "counts": counts, "ece": round(ece, 5),
            "brier": round(float(brier_score_loss(y_true, p_pred)), 5)}