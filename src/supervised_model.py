"""Supervised fraud classifiers.

Trains and compares three tabular models (Logistic Regression, Random Forest,
LightGBM), handles class imbalance via *positive-class weighting* (balanced
class weights for LR/RF, ``scale_pos_weight`` for LightGBM) and selects the
primary model by PR-AUC (average precision) on the validation fold - never
by accuracy.

The primary model outputs ``fraud_probability`` = calibrated positive-class
probability in [0, 1].
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Settings

MODEL_ARTIFACTS_DIR = None  # set here; resolved relative to project root


def _artifacts_dir(cfg: Settings) -> str:
    d = cfg.models_dir()
    os.makedirs(d, exist_ok=True)
    return d


def imbalanced_pos_weight(y: np.ndarray) -> float:
    """Heuristic positive-class weight for binary objectives."""
    pos = int(np.sum(y))
    neg = int(len(y) - pos)
    return float(np.sqrt(max(1, neg) / max(1, pos)))


def build_logistic(cfg: Settings, Xtr: pd.DataFrame, ytr: np.ndarray):
    p = cfg.models["logistic_regression"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=float(p.get("C", 0.5)), max_iter=int(p.get("max_iter", 2000)),
            class_weight="balanced", solver="lbfgs", random_state=cfg.seed)),
    ])


def build_random_forest(cfg: Settings):
    p = cfg.models["random_forest"]
    return RandomForestClassifier(
        n_estimators=int(p.get("n_estimators", 300)),
        max_depth=int(p.get("max_depth", 18)),
        min_samples_leaf=int(p.get("min_samples_leaf", 30)),
        n_jobs=int(p.get("n_jobs", -1)),
        class_weight="balanced",
        random_state=cfg.seed,
    )


def build_lightgbm(cfg: Settings, ytr: np.ndarray):
    p = cfg.models["lightgbm"]
    spw = p.get("scale_pos_weight", "auto")
    if spw == "auto":
        spw = imbalanced_pos_weight(ytr)
    params = {
        "objective": "binary",
        "n_estimators": int(p.get("n_estimators", 600)),
        "learning_rate": float(p.get("learning_rate", 0.08)),
        "num_leaves": int(p.get("num_leaves", 63)),
        "min_child_samples": int(p.get("min_child_samples", 80)),
        "subsample": float(p.get("subsample", 0.9)),
        "subsample_freq": int(p.get("subsample_freq", 1)),
        "colsample_bytree": float(p.get("colsample_bytree", 0.8)),
        "reg_lambda": float(p.get("reg_lambda", 5.0)),
        "scale_pos_weight": float(spw),
        "verbosity": int(p.get("verbosity", -1)),
        "random_state": cfg.seed,
        "n_jobs": -1,
    }
    return pipeline_lgb(params)


def pipeline_lgb(params: dict):
    return LGBMClassifier(**params)


def train_and_compare(Xtr: pd.DataFrame, ytr: np.ndarray,
                      Xval: pd.DataFrame, yval: np.ndarray,
                      cfg: Settings) -> dict:
    """Train the configured models, return comparison + fitted artifacts."""
    results = {}
    candidates = cfg.models.get("compare", ["lightgbm"])

    for name in candidates:
        print(f"[model] training {name} ...")
        if name == "logistic_regression":
            model = build_logistic(cfg, Xtr, ytr)
            if Xval is not None and len(Xval):
                model.fit(Xtr, ytr)
                pv = model.predict_proba(Xval)[:, 1]
            else:
                model.fit(Xtr, ytr)
                pv = model.predict_proba(Xtr)[:, 1]
        elif name == "random_forest":
            model = build_random_forest(cfg)
            model.fit(Xtr, ytr)
            pv = model.predict_proba(Xval)[:, 1]
        elif name == "lightgbm":
            if LGBMClassifier is None:
                print("    !! lightgbm not installed; skipping")
                continue
            model = build_lightgbm(cfg, ytr)
            model.fit(
                Xtr, ytr,
                eval_set=[(Xtr, ytr), (Xval, yval)],
                eval_metric=["average_precision", "auc"],
                callbacks=[_early_stopping()],
            )
            pv = model.predict_proba(Xval)[:, 1]
        else:  # pragma: no cover
            raise ValueError(f"unknown model candidate '{name}'")

        pr = float(average_precision_score(yval, pv))
        roc = float(roc_auc_score(yval, pv))
        results[name] = {"model": model, "pr_auc_val": pr, "roc_auc_val": roc,
                         "features": list(Xtr.columns)}
        print(f"    -> PR-AUC(val)={pr:.4f}  ROC-AUC(val)={roc:.4f}")

    if not results:
        raise RuntimeError("No model candidates trained successfully.")

    best = max(results, key=lambda k: results[k]["pr_auc_val"])
    results["__best__"] = best
    return results


def _early_stopping():
    try:
        from lightgbm import early_stopping
        return early_stopping(stopping_rounds=80, verbose=False)
    except Exception:  # pragma: no cover
        return None


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1]).astype(float)


def save_model_bundle(cfg: Settings, results: dict, names: list[str]) -> str:
    """Persist trained models + the comparison table."""
    out_dir = _artifacts_dir(cfg)
    best = results.get("__best__")
    for name in names:
        if name in results:
            joblib.dump(results[name]["model"],
                        os.path.join(out_dir, f"model_{name}.joblib"))
    comp = {k: {"pr_auc_val": v["pr_auc_val"], "roc_auc_val": v["roc_auc_val"]}
            for k, v in results.items() if k != "__best__"}
    summary = {"best": best, "comparison": comp}
    with open(os.path.join(out_dir, "model_comparison.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return out_dir