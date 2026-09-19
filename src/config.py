"""Lightweight configuration loader.

Reads ``config.yaml`` (project root) and exposes a ``Settings`` object.
Everything stays *configurable rather than hardcoded*: paths, column names,
split fractions, model hyperparameters, risk thresholds and budgets all come
from here (with sane defaults if the YAML is missing).
"""
from __future__ import annotations

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")

DEFAULTS = {
    "project": {"name": "Unseen-Customer Fraud Detection", "seed": 42},
    "data": {"raw_file": "data/raw/synthetic_fraud_transactions.csv",
             "processed_dir": "data/processed"},
    "splits": {"train_frac": 0.60, "val_frac": 0.20, "test_frac": 0.20,
               "unseen_customer_frac": 0.20,
               "cold_max_prior": 0, "warm_min_prior": 1, "warm_max_prior": 4,
               "established_min_prior": 5},
    "features": {"parity_velocity_window_min": 8, "cold_kappa": 5.0,
                 "customer_baseline_min_n": 3, "novelty_model": "isolation_forest",
                 "novelty_noise": 0.4, "novelty_normalize_on": "calibration"},
    "models": {
        "compare": ["logistic_regression", "random_forest", "lightgbm"],
        "primary": "lightgbm",
        "lightgbm": {"scale_pos_weight": "auto", "num_leaves": 63,
                     "min_child_samples": 80, "learning_rate": 0.08,
                     "n_estimators": 600, "subsample": 0.9,
                     "subsample_freq": 1, "colsample_bytree": 0.8,
                     "reg_lambda": 5.0, "verbosity": -1},
        "random_forest": {"n_estimators": 300, "max_depth": 18,
                          "min_samples_leaf": 30, "n_jobs": -1},
        "logistic_regression": {"C": 0.5, "max_iter": 2000},
    },
    "calibration": {"method": "isotonic", "bins": 10},
    "risk": {"p_review": 0.15, "p_high": 0.45, "novelty_high": 0.90,
             "novelty_very_high": 0.97, "p_floor_for_novelty": 0.02,
             "alert_budget_pct": 2.0, "high_risk_share_cap_pct": 0.7,
             "max_alerts_per_hour": 250, "max_alerts_per_day": 3000},
    "evaluation": {"alert_budget_pct_grid": [0.5, 1.0, 2.0, 3.0, 5.0],
                   "shap_sample_size": 2000, "explanation_reason_groups": True},
    "dashboard": {"host": "localhost", "port": 8501},
    "api": {"host": "0.0.0.0", "port": 8000},
}


def _deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Settings:
    """Simple attribute-style settings object built from YAML + defaults."""

    def __init__(self, config_path: str = CONFIG_PATH):
        raw = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        cfg = _deep_merge(DEFAULTS, raw)
        self.raw = cfg
        self.project = cfg["project"]
        self.data = dict(cfg["data"])
        self.splits = dict(cfg["splits"])
        self.features = dict(cfg["features"])
        self.models = cfg["models"]
        self.calibration = dict(cfg["calibration"])
        self.risk = dict(cfg["risk"])
        self.evaluation = dict(cfg["evaluation"])
        self.dashboard = dict(cfg["dashboard"])
        self.api = dict(cfg["api"])
        self.name = self.project["name"]
        self.seed = int(self.project["seed"])

    # ---- path helpers -------------------------------------------------
    def raw_path(self):
        p = self.data["raw_file"]
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    def processed_dir(self):
        p = self.data["processed_dir"]
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    def models_dir(self):
        return os.path.join(ROOT, "models")

    def reports_dir(self):
        return os.path.join(ROOT, "reports")

    def figures_dir(self):
        return os.path.join(self.reports_dir(), "figures")

    def artifacts_dir(self):
        return os.path.join(self.reports_dir(), "artifacts")


def get_settings(config_path: str = CONFIG_PATH) -> Settings:
    return Settings(config_path)


if __name__ == "__main__":
    s = Settings()
    print("name:", s.name)
    print("raw_path:", s.raw_path())
    print("primary model:", s.models["primary"])