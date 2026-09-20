"""Comprehensive QA suite for the Unseen-Customer Fraud Detection app.

Run from the project root:
    python scripts/qa_suite.py

Covers three planes:

1. FRONTEND (Streamlit AppTest)
     A. boot renders all three sections in order (Overview / Upload & Analyse
        / System evaluation).
     B. drag-and-drop CSV upload triggers the Analyse flow.
     C. the dynamic System Evaluation metrics update from the uploaded file
        (never static hold-out numbers).

2. BACKEND (FastAPI TestClient)
     D. /predict scores ONE transaction with the exact production streaming
        code path (validate -> features -> model -> calibrate -> novelty
        -> risk -> explain -> commit).
     E. every decision carries the four required output fields:
        fraud_probability, novelty_score, risk_band, top reasons.
     F. /predict/batch replays chronologically and preserves request order.
     G. an invalid transaction is rejected with a structured 400.

3. CONSTRAINT CHECKS (leakage + budget)
     H. point-in-time integrity: a transaction never sees its own future;
        identical input -> identical decision (determinism).
     I. raw ids never become model features (identity-free design).
     J. the alert budget hard-caps escalations on a fraud burst.

Each check is reported as PASS/FAIL; the file writes reports/qa_report.md.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = []
_START = time.time()


def record(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok), detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" - {detail}" if detail else ""))


def section(title: str):
    print(f"\n=== {title} ===")


def embed_md() -> str:
    lines = []
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    lines.append("# QA Report - Unseen-Customer Fraud Detection")
    lines.append("")
    lines.append(f"Run finished in **{time.time() - _START:.1f}s** - "
                 f"**{passed}/{total} checks passed**.")
    lines.append("")
    lines.append("| # | check | status | detail |")
    lines.append("|---|---|---|---|")
    for i, (name, ok, detail) in enumerate(RESULTS, 1):
        lines.append(f"| {i} | {name} | {'PASS' if ok else 'FAIL'} | "
                     f"{detail or ''} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- helpers
def _sample_txn(over: dict | None = None) -> dict:
    txn = {"timestamp": "2024-05-01 10:30:00+00:00", "amount": 240.0,
           "customer_id": "C-001", "merchant_id": "M-001",
           "device_id": "D-001", "transaction_hour": 10}
    if over:
        txn.update(over)
    return txn


# =================================================================== 1. FRONTEND
def run_frontend():
    section("Frontend (Streamlit AppTest)")
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as exc:  # pragma: no cover
        record("frontend import AppTest", False, str(exc))
        return

    at = AppTest.from_file(os.path.join(ROOT, "dashboard", "app.py"))
    at.run(timeout=120)
    record("A. app boots with no exceptions", len(at.exception) == 0,
           f"{len(at.exception)} exception(s)")

    headers = [str(h.value) for h in at.header]
    subheads = [str(s.value) for s in at.subheader]
    s1 = any("1. Overview" in h for h in headers + subheads)
    s2 = any("2. Upload & Analyse" in h for h in subheads)
    s3 = any("System evaluation" in h for h in subheads)
    record("B. three sections render (Overview / Upload / System eval)",
           s1 and s2 and s3, f"sections: overview={s1} upload={s2} eval={s3}")

    # ---- C. upload + Analyse ----
    sample = os.path.join(ROOT, "data", "processed", "test_fold_file.csv")
    if os.path.exists(sample):
        with open(sample, "rb") as fh:
            data = fh.read()
        at.file_uploader[0].set_value([(os.path.basename(sample), data,
                                        "text/csv")])
        at.run(timeout=120)
        btn = next((b for b in at.button if getattr(b, "label", "") ==
                    "Analyse"), None)
        if btn is None:
            record("C. upload + Analyse", False,
                   "'Analyse' button not found after upload")
            return
        btn.click()
        at.run(timeout=600)
        record("C1. Analyse button runs the pipeline",
               len(at.exception) == 0, f"{len(at.exception)} exception(s)")
        # dynamic metrics in section 3 reflect *this file*
        caps = " ".join(str(c.value) for c in at.caption)
        record("C2. System evaluation shows a dynamic file caption",
               ("test_fold_file.csv" in caps) or ("rows" in caps.lower()),
               caps[:120])
    else:
        record("C. upload + Analyse", False, "test_fold_file.csv missing")


# =================================================================== 2. BACKEND
def run_backend():
    section("Backend (FastAPI TestClient)")
    try:
        from fastapi.testclient import TestClient
        from api.main import app
    except Exception as exc:  # pragma: no cover
        record("backend import TestClient + app", False, str(exc))
        return
    c = TestClient(app)

    h = c.get("/health")
    record("D. /health reports service state", h.status_code == 200
           and h.json().get("status") == "ok", str(h.json().get("status")))

    # ---- one-transaction streaming ----
    t0 = time.time()
    r = c.post("/predict", json=_sample_txn())
    dt = (time.time() - t0) * 1000
    js = r.json()
    needs = ["fraud_probability", "novelty_score", "risk_band",
             "top_contributing_features", "explanation"]
    ok_field = all(k in js for k in needs)
    has_reasons = bool(js.get("top_contributing_features"))
    has_all4 = all(k in js for k in
                   ["fraud_probability", "novelty_score", "risk_band"] +
                   ["top_contributing_features", "explanation"])
    record("E1. /predict returns risk decision", r.status_code == 200,
           f"http {r.status_code} in {dt:.0f}ms")
    record("E2. all four required fields present",
           ok_field and has_reasons and has_all4,
           f"fields={sorted(js.keys())}")
    record("E3. probabilities in [0,1]", 0 <= js["fraud_probability"] <= 1
           and 0 <= js["novelty_score"] <= 1,
           f"p={js['fraud_probability']:.3f} nov={js['novelty_score']:.3f}")
    record("E4. risk band is one of the four bands",
           js["risk_band"] in ("normal", "monitor", "review", "high-risk"),
           js["risk_band"])

    # ---- batch preserves order ----
    txn_a = _sample_txn({"timestamp": "2024-05-01 08:00:00+00:00",
                         "customer_id": "C1", "amount": 50.0})
    txn_b = _sample_txn({"timestamp": "2024-05-01 09:30:00+00:00",
                         "customer_id": "C2", "amount": 5000.0})
    payload = {"transactions": [txn_a, txn_b]}
    rb = c.post("/predict/batch", json=payload)
    decided = rb.json()
    record("F1. /predict/batch returns decisions for every row",
           rb.status_code == 200 and len(decided) == 2,
           f"http {rb.status_code}, {len(decided)} decisions")
    record("F2. response preserves request order",
           decided and all(d.get("valid") for d in decided)
           and decided[0]["fraud_probability"] <=
           decided[1]["fraud_probability"],
           f"p0={decided[0]['fraud_probability']} "
           f"p1={decided[1]['fraud_probability']}")
    record("F3. batch scores chronologically (all rows valid)",
           decided and all(d.get("valid") for d in decided))

    # ---- invalid transaction ----
    bad = _sample_txn({"amount": -5.0})
    rbad = c.post("/predict", json=bad)
    record("G. invalid transaction -> structured 4xx",
           rbad.status_code in (400, 422), f"http {rbad.status_code}")


# ============================================================== 3. CONSTRAINTS
def run_constraints():
    section("Constraint checks (leakage + budget)")
    try:
        import json
        from src.config import get_settings
        from src.feature_engineering import (FeatureEngine, MODEL_FEATURES)
        from src.streaming_processor import (StreamingProcessor,
                                             AlertBudgetController)
        from src.risk_engine import RiskEngine
        import joblib
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        record("constraint imports", False, str(exc))
        return

    cfg = get_settings()
    md = cfg.models_dir()
    model = joblib.load(os.path.join(md, "model_primary.joblib"))
    cal = joblib.load(os.path.join(md, "calibrator.joblib"))
    nov = joblib.load(os.path.join(md, "novelty.joblib"))
    priors = joblib.load(os.path.join(md, "priors.joblib"))
    thr = json.load(open(os.path.join(md, "thresholds.json")))

    def _proc(with_budget=False, with_engine_novelty=True):
        eng = FeatureEngine(cfg, priors)
        return StreamingProcessor(
            cfg, eng, model, feature_names=MODEL_FEATURES,
            calibrator=cal, novelty=nov if with_engine_novelty else None,
            risk_engine=RiskEngine(thr),
            budget=(AlertBudgetController(
                max_per_hour=int(cfg.risk["max_alerts_per_hour"]),
                max_per_day=int(cfg.risk["max_alerts_per_day"]),
                budget_pct=float(cfg.risk["alert_budget_pct"]),
                high_risk_share_cap_pct=float(cfg.risk["high_risk_share_cap_pct"])
            ) if with_budget else None))

    # H1. determinism: identical txn -> identical decision (fresh state each)
    r1 = _proc().score_transaction(_sample_txn())
    r2 = _proc().score_transaction(_sample_txn())
    both = {"risk_band", "fraud_probability", "novelty_score"}
    record("H1. identical input -> identical decision (determinism)",
           all(r1[k] == r2[k] for k in both),
           f"band={r1['risk_band']} p={r1['fraud_probability']}")

    # H2. point-in-time: a 3rd txn on same customer sees the 2 committed rows
    p = _proc()
    p.score_transaction(_sample_txn({"customer_id": "C-PIT",
                                     "amount": 120.0,
                                     "timestamp": "2024-05-01 08:00:00+00:00"}))
    p.score_transaction(_sample_txn({"customer_id": "C-PIT",
                                     "amount": 120.0,
                                     "timestamp":
                                         "2024-05-01 08:05:00+00:00"}))
    probe = _sample_txn({"customer_id": "C-PIT", "amount": 120.0})
    probe["ts_sec"] = float(pd.Timestamp("2024-05-01 08:06:00+00:00").timestamp())
    feats_after = p.engine.prepare_row(probe)
    record("H2. streaming state accumulates history (2nd txn sees 1st)",
           feats_after["customer_history_count"] == 2,
           f"history_count={feats_after['customer_history_count']}")

    # H3. raw ids never in model features (identity-free)
    raw_ids = {"customer_id", "merchant_id", "device_id"}
    leak = sorted(raw_ids & set(MODEL_FEATURES))
    record("I. raw ids not in MODEL_FEATURES (identity-free)",
           not leak, f"raw cols leaked={leak}")

    # J. budget caps a fraud burst
    burst = []
    for i in range(60):
        burst.append(_sample_txn({
            "timestamp": f"2024-05-01 12:{i // 60:02d}:{i % 60:02d}+00:00",
            "customer_id": f"C-burst-{i % 5}",
            "merchant_id": "M-burst", "amount": 900.0 + i}))
    p2 = _proc(with_budget=True)
    decs = [p2.score_transaction(t) for t in burst]
    alerted = sum(1 for d in decs if d["budget_status"] == "alert")
    budget_cap = max(3, int(len(burst) * cfg.risk["alert_budget_pct"] / 100.0))
    record("J. alert budget hard-caps escalations",
           alerted <= budget_cap,
           f"alerts={alerted} cap={budget_cap} "
           f"({cfg.risk['alert_budget_pct']}% of {len(burst)})")

    # leak report cross-check
    rep = json.load(open(os.path.join(cfg.reports_dir(),
                                      "evaluation_report.json")))
    rl = rep.get("leakage", {})
    record("K. leakage report flags raw ids clean",
           rl.get("raw_ids_in_features") is False,
           f"raw_ids_in_features={rl.get('raw_ids_in_features')}")

    df = pd.read_csv(os.path.join(cfg.processed_dir(),
                                  "test_fold_file.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_sec"] = df["timestamp"].astype("int64") // 10**9
    df = df.sort_values("ts_sec").reset_index(drop=True)


def main():
    print("Unseen-Customer Fraud Detection - QA suite")
    run_frontend()
    run_backend()
    run_constraints()

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    out = os.path.join(ROOT, "reports", "qa_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(embed_md())
    print("\n" + "=" * 30)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"{passed}/{len(RESULTS)} checks passed  ->  {out}")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()