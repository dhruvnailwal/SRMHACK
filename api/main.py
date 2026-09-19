"""FastAPI service exposing the trained fraud-risk model.

Run from the project root:
    uvicorn api.main:app --reload

Endpoints
---------
* ``GET  /health``        - service status, model + thresholds metadata
* ``GET  /leaderboard``   - validation model comparison
* ``POST /predict``       - score ONE transaction, return risk decision
* ``POST /predict/batch`` - score MANY transactions in one chronological
                            pass (fresh processor per batch), return decisions

Scoring semantics
-----------------
The service uses the *same production code path* as the pipeline's streaming
processor: features are computed point-in-time by the feature engine, the
classifier + calibrator produce ``fraud_probability``, the novelty model
produces ``novelty_score``, and the risk engine issues the band/alert
decision. Explanations are exact TreeSHAP contributions grouped into human
reason categories.

State is built by replaying every transaction the service has scored since
startup (the feature engine state lives in memory). For deployments you
should instead seed the engine from the saved stream state:
``python run_pipeline.py`` persists ``data/processed/val_scores.parquet``
etc.; see ``seed_state = True`` support below (set ``ESTATE_SEED`` env var to
a parquet path to pre-seed history).

A single transaction with no surrounding history is scored at *cold start*:
all customer/device/merchant baselines fall back to segment/global priors.
This is the documented cold-start behaviour - history is not invented.
"""
from __future__ import annotations

import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import get_settings                 # noqa: E402
from src.feature_engineering import (FeatureEngine,  # noqa: E402
                                     MODEL_FEATURES)
from src.risk_engine import RiskEngine               # noqa: E402
from src.explainability import Explainer             # noqa: E402
from src.streaming_processor import StreamingProcessor  # noqa: E402

cfg = get_settings()
APP_NAME = "Unseen-Customer Fraud Detection API"


class Payload(BaseModel):
    timestamp: str
    amount: float = Field(gt=0)
    customer_id: str = Field(min_length=1)
    merchant_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    transaction_hour: int | None = Field(default=None, ge=0, le=23)
    include_explanation: bool = True


class Decision(BaseModel):
    valid: bool = True
    fraud_probability: float
    raw_probability: float
    novelty_score: float
    risk_band: str
    alert: bool
    rank_score: float
    tags: list[str]
    top_contributing_features: list[dict]
    explanation: str


def _build_processor(seed_path: str | None = None):
    models_dir = cfg.models_dir()
    model = joblib.load(os.path.join(models_dir, "model_primary.joblib"))
    calibrator = joblib.load(os.path.join(models_dir, "calibrator.joblib"))
    novelty = joblib.load(os.path.join(models_dir, "novelty.joblib"))
    priors = joblib.load(os.path.join(models_dir, "priors.joblib"))
    thresholds = joblib.load(os.path.join(models_dir, "thresholds.joblib")) \
        if os.path.exists(os.path.join(models_dir, "thresholds.joblib")) \
        else _read_thresholds(models_dir)
    engine = FeatureEngine(cfg, priors)
    proc = StreamingProcessor(
        cfg, engine, model, feature_names=MODEL_FEATURES,
        calibrator=calibrator, novelty=novelty,
        risk_engine=RiskEngine(thresholds),
        explainer=Explainer(model, MODEL_FEATURES, reason_groups=True,
                            seed=cfg.seed))
    if seed_path and os.path.exists(seed_path):
        df = pd.read_parquet(seed_path)
        proc.seed_state(df)
        print(f"[api] seeded state with {len(df):,} transactions",
              file=sys.stdout)
    return proc


def _read_thresholds(models_dir: str) -> dict:
    import json
    with open(os.path.join(models_dir, "thresholds.json")) as fh:
        return json.load(fh)


def _load_leaderboard() -> dict:
    import json
    with open(os.path.join(cfg.models_dir(), "model_comparison.json")) as fh:
        return json.load(fh)


app = FastAPI(title=APP_NAME, version="1.0.0",
              description="Fraud-probability + behavioural-novelty risk "
                          "scoring service")
_proc = _build_processor(os.environ.get("ESTATE_SEED"))
_leaderboard = _load_leaderboard()


@app.get("/health")
def health():
    return {"status": "ok", "app": APP_NAME,
            "model": _leaderboard.get("best"),
            "features": len(MODEL_FEATURES),
            "seeded_transactions": _proc.engine.global_["n"]}


@app.get("/leaderboard")
def leaderboard():
    return _leaderboard


class BatchRequest(BaseModel):
    transactions: list[Payload]


@app.post("/predict/batch", response_model=list[Decision])
def predict_batch(req: BatchRequest):
    """Score many transactions in a single chronological replay.

    A *fresh* processor is built for the request so the file's own history
    builds the state (identical semantics to the pipeline's streaming
    replay). Transactions are processed in timestamp order for point-in-time
    integrity; the returned list preserves the request order.
    """
    if not req.transactions:
        return []
    proc = _build_processor()
    order = sorted(range(len(req.transactions)),
                   key=lambda i: pd.to_datetime(
                       req.transactions[i].timestamp, utc=True).timestamp())
    out: list[dict] = [None] * len(req.transactions)
    for i in order:
        p = req.transactions[i]
        ts = pd.to_datetime(p.timestamp, utc=True)
        hour = p.transaction_hour if p.transaction_hour is not None \
            else int(ts.hour)
        txn = {"timestamp": p.timestamp, "ts_sec": float(ts.timestamp()),
               "amount": float(p.amount),
               "customer_id": p.customer_id,
               "merchant_id": p.merchant_id,
               "device_id": p.device_id,
               "transaction_hour": hour}
        res = proc.score_transaction(txn)
        if not res.get("valid"):
            raise HTTPException(
                status_code=400, detail={"row_index": i,
                                         **dict(res.get("errors", {}))})
        if not p.include_explanation:
            res["top_contributing_features"] = []
            res["explanation"] = ""
        out[i] = Decision(**{k: res.get(k) for k in Decision.model_fields})
    return out


@app.post("/predict", response_model=Decision)
def predict(p: Payload):
    ts = pd.to_datetime(p.timestamp, utc=True)
    hour = p.transaction_hour if p.transaction_hour is not None \
        else int(ts.hour)
    txn = {"timestamp": p.timestamp, "ts_sec": float(ts.timestamp()),
           "amount": float(p.amount),
           "customer_id": p.customer_id,
           "merchant_id": p.merchant_id,
           "device_id": p.device_id,
           "transaction_hour": hour}
    res = _proc.score_transaction(txn)
    if not res.get("valid"):
        raise HTTPException(status_code=400, detail=res.get("errors"))
    if not p.include_explanation:
        res["top_contributing_features"] = []
        res["explanation"] = ""
    return Decision(**{k: res.get(k) for k in Decision.model_fields})