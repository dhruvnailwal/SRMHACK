"""Behavioural novelty (anomaly) detector.

The novelty component answers a DIFFERENT question than the classifier:
*"how unusual is this behaviour compared with the normal traffic it resembles?"*

Approach
--------
1. An anomaly model is trained on legitimate (is_fraud == 0) training rows
   using ONLY behavioural features (never raw identity values).
2. Its raw decision score is converted into ``novelty_score`` in [0, 1] by an
   ECDF (empirical CDF) fitted on legitimate calibration traffic:

       novelty = 1 - ECDF(raw_score)

   Higher raw scores mean *more normal* for IsolationForest, so a transaction
   more atypical than 90% of legitimate traffic gets novelty = 0.90.

3. A high novelty score NEVER fires an alert by itself (handled by the risk
   engine): new-but-normal customers are deliberately protected.

Default model is IsolationForest; LOF / One-Class SVM / a small MLP
autoencoder are available for comparison.
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM


class NoveltyModel:
    """Wrapper unifying anomaly model + ECDF normalisation."""

    def __init__(self, name: str = "isolation_forest", seed: int = 42,
                 noise: float = 0.4):
        self.name = name
        self.seed = seed
        self.noise = noise
        self.model = self._build(name, seed)
        self.ref_scores: np.ndarray | None = None   # sorted ECDF reference
        self.ref_n = 0

    @staticmethod
    def _build(name: str, seed: int):
        if name == "isolation_forest":
            return IsolationForest(
                n_estimators=200, max_samples=4096, contamination="auto",
                random_state=seed, n_jobs=-1)
        if name == "lof":
            return LocalOutlierFactor(n_neighbors=80, novelty=True)
        if name == "one_class_svm":
            return OneClassSVM(kernel="rbf", gamma=0.01, nu=0.05)
        if name == "autoencoder":
            return MLPRegressor(
                hidden_layer_sizes=(32, 8, 32), max_iter=60, random_state=seed,
                early_stopping=True)
        raise ValueError(f"unknown novelty model '{name}'")

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None,
            legit_only: bool = True):
        """Train on legitimate samples (default) or on everything."""
        mask = (y == 0) if (y is not None and legit_only) else np.ones(len(X), bool)
        Xf = X[mask].to_numpy(dtype=float)
        if self.name == "autoencoder":
            # target = reconstruction of input
            self.model.fit(Xf, Xf)
        else:
            self.model.fit(Xf)
        self._fit_encoding = Xf
        return self

    def raw_deviation(self, X: pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
        """Higher value = MORE anomalous regardless of model type.
        Maps every variant's output to a common monotone 'anomaly' axis."""
        Xa = np.asarray(X, dtype=float)
        if self.name == "autoencoder":
            recon = self.model.predict(Xa)
            dev = np.sqrt(((Xa - recon) ** 2).sum(axis=1))      # reconstruction err
        elif self.name == "lof" or self.name == "one_class_svm":
            # predict() returns +1 normal / -1 anomaly; decision fn higher = normal
            d = self.model.decision_function(Xa)
            dev = -d
        else:  # isolation_forest: decision_function higher = normal
            d = self.model.decision_function(Xa)
            dev = -d
        return np.asarray(dev, dtype=float)

    # ------------------------------------------------------------------
    def fit_ecdf(self, X: pd.DataFrame, y: np.ndarray | None = None,
                 legit_only: bool = True):
        """Fit the ECDF reference on legitimate calibration traffic."""
        mask = (y == 0) if (y is not None and legit_only) else np.ones(len(X), bool)
        raw = self.raw_deviation(X[mask])
        raw = np.nan_to_num(raw, nan=0.0, posinf=1e9, neginf=-1e9)
        self.ref_scores = np.sort(raw)
        self.ref_n = len(self.ref_scores)
        return self

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        """ECDF-rank the raw deviation into a novelty score in [0, 1].

        novelty = 1 - ECDF(raw). A value of 0.95 means the deviation exceeds
        ~95% of the legitimate reference traffic.
        """
        raw = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0,
                            posinf=1e9, neginf=-1e9)
        if self.ref_scores is None or self.ref_n == 0:
            raise RuntimeError("NoveltyModel.fit_ecdf() must be called before "
                               "normalize()")
        ranks = np.searchsorted(self.ref_scores, raw, side="right")
        return 1.0 - (ranks / self.ref_n)

    def novelty(self, X: pd.DataFrame) -> np.ndarray:
        return self.normalize(self.raw_deviation(X))


def fit_compare_variants(Xtr_l, ytr, Xval_l, yval, names, seed=42):
    """Fit several novelty variants and return their cold-cohort novelty AUROC
    (does novelty rank fraud above legit first-timers?)."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for n in names:
        try:
            nm = NoveltyModel(n, seed=seed)
            nm.fit(Xtr_l, ytr, legit_only=True)
            nm.fit_ecdf(Xval_l, yval, legit_only=True)
            nov = nm.novelty(Xval_l)
            a = roc_auc_score(yval, nov) if len(np.unique(yval)) > 1 else float("nan")
            out[n] = {"novelty_auroc_val": a}
        except Exception as e:  # pragma: no cover
            out[n] = {"novelty_auroc_val": None, "error": str(e)}
    return out