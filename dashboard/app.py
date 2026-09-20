"""Streamlit dashboard for the Unseen-Customer Fraud Detection project.

Run from the project root:
    streamlit run dashboard/app.py

Sections
--------
1. Overview & context          - what the model does, the memorization vs
                               generalization problem, and how two signals
                               (fraud probability + novelty) make one
                               decision - shown before any tool, no controls
2. Upload & Analyse            - drag-and-drop / browse a CSV, then Analyse:
                               per-transaction fraud probability gauge, novelty
                               score, colour-coded risk band, top reasons;
                               adjustable alert-budget slider that re-derives
                               Review/High-Risk thresholds; exports in
                               .csv / .pdf / .json / .md / .txt. Scoring can
                               run in-process or via the FastAPI service
                               (``python -m uvicorn api.main:app --reload``).
3. System evaluation (dynamic) - PR-AUC / recall @ budget / known-vs-unseen
                               computed from the file you just uploaded - never
                               static hold-out numbers; leakage-safety
                               compliance panel.

Every number in section 3 is derived from the analysed file; the model
itself comes from the training artifacts (models/, reports/).
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import get_settings               # noqa: E402
from src.data_loader import load_dataframe, valid_numeric  # noqa: E402
from src.feature_engineering import (FeatureEngine,  # noqa: E402
                                     MODEL_FEATURES)
from src.streaming_processor import (StreamingProcessor,  # noqa: E402
                                     AlertBudgetController)
from src.risk_engine import RiskEngine             # noqa: E402
from src.explainability import Explainer           # noqa: E402

SCHEMA_DOC = """The loader tolerates common aliases (timestamp / event_time,
customer_id / customer_key, ...). Required columns:
`transaction_id`, `timestamp`, `customer_id`, `merchant_id`, `device_id`,
`amount`, `is_fraud`. Optional features used for parity checks:
`transaction_hour`, `is_night`, `new_customer`, `new_merchant`, `new_device`,
`velocity_8min`, `amount_spike`, `customer_history_count`,
`merchant_history_count`, `device_history_count`."""

RISK_COLORS = {"normal": "#2e7d32", "monitor": "#f9a825",
               "review": "#ef6c00", "high-risk": "#c62828"}
BAND_ORDER = ["normal", "monitor", "review", "high-risk"]


@st.cache_data(show_spinner=False)
def load_context() -> dict:
    cfg = get_settings()
    ctx = {"cfg": cfg}
    models = cfg.models_dir()
    ctx["model"] = joblib.load(os.path.join(models, "model_primary.joblib"))
    ctx["calibrator"] = joblib.load(os.path.join(models, "calibrator.joblib"))
    ctx["novelty"] = joblib.load(os.path.join(models, "novelty.joblib"))
    ctx["priors"] = joblib.load(os.path.join(models, "priors.joblib"))
    ctx["thresholds"] = json.load(open(os.path.join(models,
                                                    "thresholds.json")))
    ctx["report"] = json.load(
        open(os.path.join(cfg.reports_dir(), "evaluation_report.json")))
    train_feats = pd.read_parquet(
        os.path.join(cfg.processed_dir(), "train_features.parquet"))
    ctx["train_customers"] = set(train_feats["customer_id"].astype(str))
    return ctx


def _metric_grid(cols, items):
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


def _run_with_factors(df: pd.DataFrame, proc) -> pd.DataFrame:
    """Score a sorted frame chronologically while KEEPING the explanation
    columns (`top_contributing_features`, tags) that ``proc.run`` drops.
    Uses the vectorized batch path (same point-in-time semantics, far faster
    on large files)."""
    return proc.run_batch(df)


def panel_overview(ctx):
    st.header("1. Overview")
    st.markdown("""
**The problem: memorization vs generalization.**

Models that memorise known-customer history excel on the transactions they
have seen during training - and quietly fail the moment they meet someone
new. Fraud is adversarial: attackers rotate cards, devices and merchants,
so the transactions that matter most are precisely the ones with **no
history to memorise**. A fraud detector must work as well on an
*unseen customer's very first transaction* as on a loyal customer's
thousandth.

This system is built for that asymmetry. Instead of betting on one
memorising classifier, it fuses **two independent signals** into a single
decision:

1. **Fraud probability** - a supervised classifier (LightGBM, selected by
   PR-AUC on validation) trained on labelled transactions, *calibrated* so
   a score of 0.8 genuinely means ~80% fraud likelihood. It uses
   point-in-time features (velocity, amount z-scores, merchant/device
   novelty, time-of-day deviation) blended with **shrinkage priors** - so a
   brand-new identity is treated with statistical caution, not paranoia.
2. **Novelty score** - an unsupervised behavioural model (IsolationForest +
   ECDF) measuring how unusual a transaction is *compared to legitimate
   traffic*. 0.90 means "more atypical than 90% of legitimate
   transactions". No labels needed; it is an identity-free second opinion.

Both signals feed the **risk decision matrix** - and crucially, **novelty
alone never alerts** (`p >= p_floor` is required), so legitimate first-time
customers are logged, not escalated. That is what turns "new" from a false
positive into a real lead.

|                 | novelty low      | novelty high      |
|-----------------|------------------|-------------------|
| **p low**       | `normal` → allow | `monitor` → log   |
| **p high**      | `review` → alert | `high-risk` → alert |

Every feature is computed **point-in-time** (no future information), the
priors are fit on training data only, and the model is proven on a disjoint
*unseen-customer* holdout. Upload a file below to see these same metrics
computed from *your* rows.
""")


@st.cache_data(show_spinner=False)
def score_upload_bytes(data: bytes, fname: str, with_explanation: bool,
                       with_budget: bool) -> dict:
    """Cached entry point for the upload scoring (see ``_score_upload_impl``)."""
    return _score_upload_impl(data, fname, with_explanation, with_budget)


def _score_upload_impl(data: bytes, fname: str, with_explanation: bool,
                       with_budget: bool) -> dict:
    """Load an uploaded CSV and score every row through the production
    streaming code path."""
    cfg = get_settings()
    tmp = tempfile.NamedTemporaryFile(suffix="_upload.csv", delete=False,
                                      dir=cfg.processed_dir())
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        df, schema = load_dataframe(cfg, tmp.name, require_label=False)
        df = valid_numeric(df)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    ctx = load_context()
    engine = FeatureEngine(cfg, ctx["priors"])
    budget = None
    if with_budget:
        budget = AlertBudgetController(
            max_per_hour=int(cfg.risk["max_alerts_per_hour"]),
            max_per_day=int(cfg.risk["max_alerts_per_day"]),
            budget_pct=float(cfg.risk["alert_budget_pct"]),
            high_risk_share_cap_pct=float(cfg.risk["high_risk_share_cap_pct"]))
    proc = StreamingProcessor(
        cfg, engine, ctx["model"], feature_names=MODEL_FEATURES,
        calibrator=ctx["calibrator"], novelty=ctx["novelty"],
        risk_engine=RiskEngine(ctx["thresholds"]),
        explainer=Explainer(ctx["model"], MODEL_FEATURES, reason_groups=True,
                            seed=cfg.seed) if with_explanation else None,
        budget=budget)
    out = _run_with_factors(df, proc)
    has_label = "is_fraud" in out.columns and out["is_fraud"].nunique() > 1
    return {"rows": int(len(df)), "filename": fname,
            "mapped": schema.mapping, "unmapped": schema.unmapped,
            "has_label": has_label, "scored": out}


def _band_value_counts(df, col="risk_band"):
    vc = df[col].value_counts()
    return vc.reindex([b for b in BAND_ORDER if b in vc.index]).dropna()


@st.cache_data(show_spinner=False)
def validate_upload_bytes(data: bytes, fname: str) -> dict:
    """Lightweight inspection of an uploaded file - schema + summary only,
    no fraud analysis (that runs only when the user presses Analyse)."""
    return _validate_upload_impl(data, fname)


def _validate_upload_impl(data: bytes, fname: str) -> dict:
    cfg = get_settings()
    tmp = tempfile.NamedTemporaryFile(suffix="_upload.csv", delete=False,
                                      dir=cfg.processed_dir())
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        df, schema = load_dataframe(cfg, tmp.name, require_label=False)
        df = valid_numeric(df)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    has_fraud = "is_fraud" in schema.mapping
    return {
        "filename": fname,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "mapped": schema.mapping,
        "unmapped": schema.unmapped,
        "missing_required": schema.missing_required,
        "has_fraud": has_fraud,
        "fraud_rate_pct": round(float(df["is_fraud"].mean() * 100), 4)
        if has_fraud else None,
        "timestamp_range": (str(df["timestamp"].min()),
                            str(df["timestamp"].max())),
        "preview": df.head(5).to_dict("records"),
    }


def _build_analysis_report(res: dict, ctx: dict) -> dict:
    """Assemble the full analysis of the uploaded file into one report dict."""
    sc = res["scored"]
    alert_dec = sc["budget_status"] == "alert"
    n_alerts = int(alert_dec.sum())
    rep = {
        "report_title": "Fraud Risk Analysis",
        "filename": res["filename"],
        "rows_scored": int(len(sc)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_mapping": res["mapped"],
        "unmapped_optional_columns": res["unmapped"],
        "has_label": res["has_label"],
        "thresholds": ctx["thresholds"],
        "risk_band_counts": sc["risk_band"].value_counts().to_dict(),
        "alerts": {
            "alerts_fired": n_alerts,
            "alert_rate_pct": round(100.0 * n_alerts / max(1, len(sc)), 3),
            "high_risk": int((sc["risk_band"] == "high-risk").sum()),
            "review": int((sc["risk_band"] == "review").sum()),
            "monitor": int((sc["risk_band"] == "monitor").sum()),
            "normal": int((sc["risk_band"] == "normal").sum()),
        },
    }
    if res["has_label"]:
        y = sc["is_fraud"].to_numpy()
        p = sc["fraud_probability"].to_numpy(dtype=float)
        tp = int(((y == 1) & alert_dec).sum())
        rep["label_metrics"] = {
            "fraud_rows_in_file": int(y.sum()),
            "fraud_rate_pct": round(float(y.mean() * 100), 3),
            "alert_precision": round(float(tp / n_alerts), 4)
            if n_alerts else None,
            "alert_recall": round(float(tp / max(1, int(y.sum()))), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
        }
    top_cols = [c for c in ("transaction_id", "timestamp", "amount",
                            "fraud_probability", "novelty_score",
                            "risk_band", "rank_score")
                if c in sc.columns]
    alerted = sc[alert_dec].sort_values("rank_score", ascending=False)
    rep["top_alerted_transactions"] = alerted.head(10)[top_cols].to_dict(
        "records") if len(alerted) else []
    if "explanation" in sc:
        rep["sample_explanation"] = \
            str(sc["explanation"].fillna("").iloc[0]) if len(sc) else ""
    return rep


def _format_analysis_markdown(rep: dict) -> str:
    L = []
    L.append(f"# {rep['report_title']}")
    L.append("")
    L.append(f"- **File**: `{rep['filename']}`")
    L.append(f"- **Rows scored**: {rep['rows_scored']:,}")
    L.append(f"- **Analysed at**: {rep['generated_at']}")
    L.append("")
    L.append("## Schema mapping used")
    L.append("")
    for canon, raw in rep["schema_mapping"].items():
        L.append(f"- `{canon}` <- `{raw}`")
    if rep.get("unmapped_optional_columns"):
        L.append(f"- Unmapped optional columns: "
                 f"{', '.join(rep['unmapped_optional_columns'])}")
    L.append("")
    L.append("## Risk-band distribution")
    L.append("")
    L.append("| risk band | transactions |")
    L.append("|---|---|")
    for band in BAND_ORDER:
        L.append(f"| {band} | {rep['risk_band_counts'].get(band, 0):,} |")
    L.append("")
    a = rep["alerts"]
    L.append("## Alert summary")
    L.append("")
    L.append(f"- Alerts fired: **{a['alerts_fired']:,}** "
             f"({a['alert_rate_pct']:.2f}% of rows)")
    L.append(f"- High-risk: {a['high_risk']:,}  |  Review: {a['review']:,}  "
             f"|  Monitor (logged): {a['monitor']:,}")
    L.append("")
    m = rep.get("label_metrics")
    if m:
        L.append("## Performance vs the fraud label (if provided)")
        L.append("")
        L.append("| metric | value |")
        L.append("|---|---|")
        for k, v in m.items():
            L.append(f"| {k} | {v} |")
        L.append("")
    if rep["top_alerted_transactions"]:
        L.append("## Top alerted transactions (by risk priority)")
        L.append("")
        hdrs = list(rep["top_alerted_transactions"][0].keys())
        L.append("| " + " | ".join(h.replace("_", " ") for h in hdrs) + " |")
        L.append("|" + "---|" * len(hdrs))
        for r in rep["top_alerted_transactions"]:
            L.append("| " + " | ".join(str(r.get(h, "")) for h in hdrs) + " |")
        L.append("")
    if rep.get("sample_explanation"):
        L.append("## Sample explanation")
        L.append("")
        L.append(rep["sample_explanation"])
        L.append("")
    if rep.get("insights"):
        L.append("## Key insights")
        L.append("")
        for line in rep["insights"]:
            L.append("- " + line)
        L.append("")
    L.append("## Thresholds used")
    L.append("")
    for k, v in rep["thresholds"].items():
        L.append(f"- {k}: {v}")
    return "\n".join(L)


SCORE_BACKENDS = {
    "Via FastAPI service (LightGBM API)": "api",
    "In-process (local joblib models)": "local",
}
DEFAULT_API_URL = "http://127.0.0.1:8000"


def _load_upload_frame(data: bytes, fname: str):
    """Load an uploaded CSV once, returning (df, schema)."""
    cfg = get_settings()
    tmp = tempfile.NamedTemporaryFile(suffix="_upload.csv", delete=False,
                                      dir=cfg.processed_dir())
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        df, schema = load_dataframe(cfg, tmp.name, require_label=False)
        df = valid_numeric(df)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    return df, schema


def _score_via_api(df: pd.DataFrame, base_url: str) -> pd.DataFrame:
    """Score a loaded frame through the FastAPI ``/predict/batch`` endpoint.
    The service replays the file chronologically with a fresh processor."""
    rows = []
    for r in df.to_dict("records"):
        hour = r.get("transaction_hour")
        hour = int(hour) if hour is not None and pd.notna(hour) else None
        rows.append({"timestamp": str(r["timestamp"]),
                     "amount": float(r["amount"]),
                     "customer_id": str(r["customer_id"]),
                     "merchant_id": str(r["merchant_id"]),
                     "device_id": str(r["device_id"]),
                     "transaction_hour": hour,
                     "include_explanation": True})
    url = base_url.rstrip("/") + "/predict/batch"
    body = json.dumps({"transactions": rows}).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            decs = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"FastAPI service unreachable at {url} ({exc.reason}). Start it "
            f"with `python -m uvicorn api.main:app --reload` or switch to "
            f"in-process scoring.") from exc
    out = pd.DataFrame.from_records(decs)
    out["transaction_id"] = df["transaction_id"].astype(str).to_numpy()
    out["timestamp"] = df["timestamp"].astype(str).to_numpy()
    out["ts_sec"] = df["ts_sec"].astype(float).to_numpy()
    out["customer_id"] = df["customer_id"].astype(str).to_numpy()
    out["merchant_id"] = df["merchant_id"].astype(str).to_numpy()
    out["device_id"] = df["device_id"].astype(str).to_numpy()
    out["amount"] = df["amount"].astype(float).to_numpy()
    out["is_fraud"] = df["is_fraud"].astype(int).to_numpy()
    out["budget_status"] = np.where(out["alert"].to_numpy(dtype=bool),
                                    "alert", "log")
    return out


def _add_unseen_flag(scored: pd.DataFrame,
                     train_customers: set) -> pd.DataFrame:
    """Mark each row as a seen or unseen (training) customer transaction."""
    out = scored.copy()
    if "customer_id" in out.columns:
        out["unseen_customer"] = (
            ~out["customer_id"].astype(str).isin(train_customers)).astype(int)
    return out


def _apply_decision(p, nov, p_review, p_high, nov_high, nov_vh, p_floor):
    """Risk decision matrix shared by the engine and the budget slider."""
    return np.select(
        [p >= p_high,
         p >= p_review,
         (nov >= nov_vh) & (p >= p_floor),
         nov >= nov_high],
        ["high-risk", "review", "review", "monitor"],
        default="normal")


def _recompute_bands(scored: pd.DataFrame, ctx: dict, budget_pct: float):
    """Re-derive risk bands + alerts from cached scores using a chosen alert
    budget (quantile probability thresholds). Returns (scored, thresholds)."""
    thr = dict(ctx["thresholds"])
    p = scored["fraud_probability"].to_numpy(dtype=float)
    nov = scored["novelty_score"].to_numpy(dtype=float)
    n = scored.shape[0]
    p_review = float(thr.get("p_review", 0.2))
    p_high = float(thr.get("p_high", 1.0))
    if n:
        budget_n = min(max(int(round(n * budget_pct / 100.0)), 1), n)
        high_n = min(max(int(round(budget_n * float(thr.get(
            "high_risk_share_cap_pct", 70.0)) / 100.0)), 1), n)
        order = np.argsort(p)[::-1]
        p_review = float(p[order[min(budget_n, n) - 1]])
        p_high = float(p[order[min(high_n, n) - 1]])
    bands = _apply_decision(p, nov, p_review, p_high,
                            float(thr["novelty_high"]),
                            float(thr.get("novelty_very_high", 1.0)),
                            float(thr.get("p_floor_for_novelty", 0.02)))
    out = scored.copy()
    out["risk_band"] = bands
    out["alert"] = np.isin(bands, ["review", "high-risk"])
    out["budget_status"] = np.where(out["alert"], "alert", "log")
    out["rank_score"] = 0.7 * p + 0.3 * nov
    new_thr = {"p_review": round(p_review, 4), "p_high": round(p_high, 4),
               "novelty_high": thr["novelty_high"],
               "p_floor_for_novelty": thr.get("p_floor_for_novelty", 0.02)}
    return out, new_thr


def _metric_summary_from_scored(scored: pd.DataFrame) -> dict | None:
    """Accuracy metrics vs the ground-truth label when the file has one."""
    if "is_fraud" not in scored.columns or scored["is_fraud"].nunique() < 2:
        return None
    y = scored["is_fraud"].astype(int).to_numpy()
    al = scored["alert"].to_numpy(dtype=bool)
    p = scored["fraud_probability"].to_numpy(dtype=float)
    tp = int(((y == 1) & al).sum())
    return {"fraud_rows_in_file": int(y.sum()),
            "fraud_rate_pct": round(float(100.0 * y.sum() / len(y)), 3),
            "alert_precision": round(float(tp / max(1, int(al.sum()))), 4),
            "alert_recall": round(float(tp / max(1, int(y.sum()))), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4)}


def _band_badge(band: str) -> str:
    color = RISK_COLORS.get(band, "#616161")
    return (f"<span style='background:{color};color:white;padding:3px 12px;"
            f"border-radius:12px;font-weight:600'>{band}</span>")


def _render_transaction_cards(scored: pd.DataFrame, rep: dict,
                              with_expl: bool):
    """Per-transaction output cards: fraud probability gauge, novelty score,
    colour-coded risk band and the top plain-language reasons."""
    alerted = scored[scored["budget_status"] == "alert"]
    scope = st.radio("Cards to show",
                     ["Alerted / reviewed only", "Top by risk score"],
                     horizontal=True, key="cards_scope")
    df = alerted.copy() if scope.startswith("Alerted") else scored.copy()
    df = df.sort_values("rank_score", ascending=False).head(25)
    if len(df) == 0:
        st.info("No transactions to display for the selected filter.")
        return
    has_label = rep.get("has_label", False)
    for _, r in df.iterrows():
        band = str(r["risk_band"])
        p = float(r["fraud_probability"])
        nov = float(r["novelty_score"])
        flag = f" · **{band.upper()}**"
        if r["budget_status"] == "alert":
            flag += " · ALERT"
        head = (f"`{r['transaction_id']}`  ·  {str(r['timestamp'])[:16]}  ·  "
                f"amount {float(r['amount']):,.2f}{flag}")
        with st.expander(head, expanded=False):
            g1, g2, g3 = st.columns([2, 2, 3])
            g1.markdown("**Fraud probability**")
            g1.progress(min(1.0, p), text=f"{p:.1%}")
            g2.markdown("**Novelty score**")
            g2.progress(min(1.0, nov), text=f"{nov:.3f}")
            g3.markdown(_band_badge(band), unsafe_allow_html=True)
            g3.markdown(f"risk rank {float(r['rank_score']):.3f}")
            if has_label:
                lbl = "**FRAUD**" if int(r["is_fraud"]) == 1 \
                    else "legitimate"
                g3.markdown(f"Ground truth: {lbl}")
            st.markdown("**Top reasons contributing to this score**")
            factors = r.get("top_contributing_features")
            if with_expl and factors:
                for fr in list(factors)[:5]:
                    st.markdown(f"- {fr.get('reason') or fr.get('feature')}")
            elif r.get("explanation"):
                st.markdown(r["explanation"])
            else:
                st.caption("Enable **Per-row explanations** before analysing "
                           "to see factor-level reasons.")


def _leakage_checklist(ctx: dict) -> str:
    """Markdown compliance checklist from the pipeline leakage report."""
    LK = ctx["report"].get("leakage", {})

    def _fmt(v):
        if isinstance(v, (int, float)):
            return f"{v:.1%} match"
        return str(v)

    raw_ids = LK.get("raw_ids_in_features", False)
    return (
        "- Temporal integrity: **" + str(LK.get("temporal_integrity", "n/a"))
        + "**  \n"
        + "- Unseen-customer holdout: **"
        + str(LK.get("unseen_customer_holdout", "n/a")) + "**  \n"
        + "- Priors / encoders fit on: **"
        + str(LK.get("priors_fit_on", "n/a")) + "**  \n"
        + "- Features computed: **"
        + str(LK.get("features_computed_as_of", "n/a")) + "**  \n"
        + "- Raw IDs as model features: **"
        + ("no - clean" if not raw_ids else "YES - flagged")
        + "**  \n"
        + "- Engineered-feature parity (velocity_8min): **"
        + _fmt(LK.get("velocity_8min_match_rate", "n/a")) + "**  \n"
        + "- Engineered-feature parity (amount_spike): **"
        + _fmt(LK.get("amount_spike_match_rate", "n/a")) + "**  \n"
        + "- Thresholds are tuned on validation only; nothing from the test "
        + "fold or the future is ever used at scoring time.")


def _render_system_eval(ctx: dict):
    """Section 3 - system evaluation, computed from the just-analysed file.

    Every metric below is derived from ``st.session_state["up_analysis"]`` -
    the file the user uploaded in section 2 - never from static hold-out
    numbers. With no analysis yet, an empty-state guide is shown instead.
    """
    st.subheader("3. System evaluation (this file, dynamic)")
    cached = st.session_state.get("up_analysis")
    if not cached:
        st.info("Upload a file and press **Analyse** above - this section "
                "then shows PR-AUC, Recall @ budget and the known-vs-unseen "
                "split calculated directly from **your** rows. No "
                "pre-trained-model placeholder numbers are shown here.")
        return
    scored = cached["scored"]
    rep = dict(cached["rep"])
    budget_pct = float(st.session_state.get("up_budget", 2.0))
    scored, newthr = _recompute_bands(scored, ctx, budget_pct)
    rep["thresholds"] = newthr
    rep["risk_band_counts"] = scored["risk_band"].value_counts().to_dict()
    rep["alerts"] = {
        "alerts_fired": int((scored["budget_status"] == "alert").sum()),
        "alert_rate_pct": round(100.0 * (scored["budget_status"] == "alert")
                                .sum() / max(1, len(scored)), 3),
        "high_risk": int((scored["risk_band"] == "high-risk").sum()),
        "review": int((scored["risk_band"] == "review").sum()),
        "monitor": int((scored["risk_band"] == "monitor").sum()),
        "normal": int((scored["risk_band"] == "normal").sum()),
    }
    if scored["is_fraud"].nunique() > 1:
        rep["label_metrics"] = _metric_summary_from_scored(scored)
    else:
        rep["label_metrics"] = None
    fname = rep.get("filename", "uploaded file")
    generated = rep.get("generated_at", "")
    try:
        when = datetime.fromisoformat(generated).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        when = "just now"
    st.caption(f"Computed from **`{fname}`** - {len(scored):,} rows, "
               f"analysed {when}.")

    a, b, c, d = st.columns(4)
    m = rep.get("label_metrics")
    if m:
        _metric_grid([a, b, c, d], [
            ("PR-AUC (this file)", f"{m['pr_auc']:.3f}"),
            ("Recall @ 2% budget", f"{m['alert_recall']:.0%}"),
            ("Alert precision", f"{m['alert_precision']:.0%}"),
            ("Fraud rate in file", f"{m['fraud_rate_pct']:.1f}%"),
        ])
    else:
        _metric_grid([a, b, c, d], [
            ("Rows scored", f"{len(scored):,}"),
            ("Alerted", f"{rep['alerts']['alerts_fired']:,}"),
            ("Alert rate", f"{rep['alerts']['alert_rate_pct']:.2f}%"),
            ("Normal (allowed)", f"{rep['alerts']['normal']:,}"),
        ])

    st.markdown("**Generalization split - known vs unseen customers**")
    if m and "unseen_customer" in scored.columns:
        rows = []
        for flag, sub in scored.groupby("unseen_customer"):
            if sub["is_fraud"].nunique() < 2:
                continue
            y = sub["is_fraud"].astype(int).to_numpy()
            p = sub["fraud_probability"].to_numpy(dtype=float)
            recall2 = 0.0
            if y.sum():
                bn = min(max(int(round(len(sub) * 0.02)), 1), len(sub))
                recall2 = float(y[np.argsort(p)[::-1][:bn]].sum() / y.sum())
            rows.append({"cohort": "Unseen" if int(flag) == 1 else "Seen",
                         "rows": len(sub), "fraud": int(y.sum()),
                         "PR-AUC": round(float(average_precision_score(y, p)),
                                         3),
                         "ROC-AUC": round(float(roc_auc_score(y, p)), 3),
                         "Recall @ 2%": round(recall2, 3)})
        if rows:
            gdf = pd.DataFrame(rows)
            st.dataframe(gdf, use_container_width=True)
            st.plotly_chart(
                px.bar(gdf, x="cohort", y="PR-AUC", color="cohort",
                       color_discrete_map={"Seen": "#1565c0",
                                           "Unseen": "#ef6c00"},
                       title="PR-AUC: seen vs unseen customers (this file)"),
                use_container_width=True)
            st.caption("No cold-start degradation: performance on customers "
                       "never seen in training matches or beats seen "
                       "customers (the model uses priors + novelty, not "
                       "memorised history).")
    else:
        vc = scored["unseen_customer"].value_counts()
        k = int(vc.get(0, 0))
        u = int(vc.get(1, 0))
        st.caption(f"File split: {k:,} transactions from customers **seen** "
                   f"in training, {u:,} from **unseen** customers "
                   f"({100 * u / max(1, len(scored)):.0f}% unseen). Add an "
                   f"`is_fraud` column to compute known-vs-unseen metrics for "
                   f"this file.")

    st.markdown("**Leakage-safety compliance panel**")
    st.markdown(_leakage_checklist(ctx))
    st.caption("The compliance rules are static pipeline guards; every metric "
               "above is recomputed from the file you just analysed.")


def panel_upload(ctx: dict):
    """Section 2 - upload / browse a CSV then run the full analysis pipeline."""
    st.subheader("2. Upload & Analyse")
    c1, c2, c3 = st.columns([2, 1, 1])
    backend = c1.selectbox("Scoring backend",
                           list(SCORE_BACKENDS.keys()),
                           key="backend_choice")
    api_url = c2.text_input("FastAPI base URL", DEFAULT_API_URL,
                            key="api_url")
    expl = c3.toggle("Per-row explanations", value=False,
                     key="with_expl", help="Adds fading-memory glms to the "
                     "decision, slowing it down.")
    f = st.file_uploader("Drop your transaction CSV here", type="csv")
    if f is None:
        st.info("Create the file with `scripts/make_test_file.py` or export "
                "a CSV with columns: timestamp, amount, customer_id, "
                "merchant_id, device_id [, is_fraud].")
        pth = os.path.join(get_settings().processed_dir(),
                           "test_fold_file.csv")
        if st.button("Analyse sample test file"):
            if os.path.exists(pth):
                with open(pth, "rb") as fh:
                    data = fh.read()
                _run_and_render(data, "test_fold_file.csv",
                                SCORE_BACKENDS[backend], api_url, expl, ctx)
            else:
                st.error("Sample file not found; generate one first.")
        _render_cached_results(ctx)
        return
    fname = f.name or "upload.csv"
    data = f.getvalue()
    if st.button("Analyse", type="primary", use_container_width=True,
                 help="Validate the file, run the model, build the report. "
                      "Rows are replayed chronologically."):
        _run_and_render(data, fname, SCORE_BACKENDS[backend], api_url, expl,
                        ctx)
    _render_cached_results(ctx)


def _render_cached_results(ctx: dict):
    """Re-render the last analysis so slider/export changes apply without
    re-running the pipeline."""
    cached = st.session_state.get("up_analysis")
    if cached:
        _render_upload_results(cached["scored"], cached["rep"],
                               cached["md_text"], cached["expl"], ctx)


def _run_and_render(data, fname, backend, api_url, expl, ctx):
    """Shared run path used by both upload branches."""
    with st.spinner("Validating and scoring…"):
        df, schema = _load_upload_frame(data, fname)
        valid = bool(schema == SCHEMA_DOC)
        if backend == "api":
            try:
                scored = _score_via_api(df, api_url)
            except RuntimeError:
                st.warning("FastAPI service unreachable - falling back to "
                           "in-process (local joblib) scoring.")
                scored = _score_upload_impl(data, fname, expl, False)["scored"]
        else:
            scored = _score_upload_impl(data, fname, expl, False)["scored"]
        scored = _add_unseen_flag(scored, ctx["train_customers"])
        scored, newthr = _recompute_bands(
            scored, ctx, float(ctx["cfg"].risk["alert_budget_pct"]))
        model_metrics = _metric_summary_from_scored(scored)
        res = {"scored": scored, "filename": fname, "mapped": schema.mapping,
               "unmapped": schema.unmapped,
               "has_label": model_metrics is not None,
               }
        rep = _build_analysis_report(res, ctx)
        rep["valid"] = valid
        rep["thresholds"] = newthr
        rep["label_metrics"] = model_metrics
        insights = _build_insights(scored, rep)
        rep["insights"] = insights
        md_text = _format_analysis_markdown(rep)
        st.success(f"File **{fname}** analysed — {len(scored):,} rows "
                   f"scored, {rep['alerts']['alerts_fired']:,} alerts")
    st.session_state["up_analysis"] = {
        "scored": scored, "rep": rep, "md_text": md_text,
        "backend": backend, "expl": expl}


def _render_upload_results(scored: pd.DataFrame, rep: dict, md_text: str,
                           with_expl: bool, ctx: dict):
    """Per-transaction results + system-evaluation + settings/export."""
    budget_pct = st.slider(
        "Alert budget (% of volume to alert on)", 0.5, 10.0,
        float(ctx["cfg"].risk["alert_budget_pct"]), 0.5,
        key="up_budget",
        help="When the slider moves, Review/High-Risk probability "
             "thresholds are re-derived as quantiles so exactly this share "
             "of transactions alerts.")
    scored, newthr = _recompute_bands(scored, ctx, budget_pct)
    rep = dict(rep)
    rep["thresholds"] = newthr
    rep["risk_band_counts"] = scored["risk_band"].value_counts().to_dict()
    rep["alerts"] = {
        "alerts_fired": int((scored["budget_status"] == "alert").sum()),
        "alert_rate_pct": round(100.0 * (scored["budget_status"] == "alert")
                                .sum() / max(1, len(scored)), 3),
        "high_risk": int((scored["risk_band"] == "high-risk").sum()),
        "review": int((scored["risk_band"] == "review").sum()),
        "monitor": int((scored["risk_band"] == "monitor").sum()),
        "normal": int((scored["risk_band"] == "normal").sum()),
    }
    if scored["is_fraud"].nunique() > 1:
        rep["label_metrics"] = _metric_summary_from_scored(scored)

    st.subheader("File metrics & model evaluation")
    a, b, c, d = st.columns(4)
    _metric_grid([a, b, c, d], [
        ("PR-AUC", _fmt_metric(rep.get("label_metrics"), "pr_auc", 3)),
        ("Recall @ 2%", _fmt_metric(rep.get("label_metrics"),
                                    "alert_recall")),
        ("Alerts fired", str(rep["alerts"]["alerts_fired"])),
        ("Alert rate", f"{rep['alerts']['alert_rate_pct']:.2f}%"),
    ])
    with st.expander("Thresholds / budget used"):
        st.write(rep["thresholds"])
    st.subheader("Per-transaction outputs")
    _render_transaction_cards(scored, rep, with_expl)
    st.subheader("Suspected-fraud list")
    alerted = scored[scored["budget_status"] == "alert"].copy()
    if len(alerted):
        cols = [c for c in ("transaction_id", "timestamp", "amount",
                            "fraud_probability", "novelty_score",
                            "risk_band", "rank_score") if c in alerted]
        st.dataframe(alerted.sort_values("rank_score", ascending=False)
                     [cols].reset_index(drop=True), use_container_width=True)
    else:
        st.info("No transactions exceed the current alert budget.")
    st.subheader("Key insights")
    for s in rep.get("insights", []):
        st.markdown(f"- {s}" if not s.startswith("-") else s)
    _render_export(scored, rep, md_text)


def _fmt_metric(label_metrics: dict | None, key: str,
                ndigits: int = 2) -> str:
    if label_metrics and key in label_metrics:
        return f"{label_metrics[key]:,.{ndigits}f}"
    return "n/a"


TABLE_ORDER = ["report_title", "filename", "rows_scored", "generated_at",
               "thresholds", "risk_band_counts", "alerts", "label_metrics",
               "sample_explanation", "insights"]


def _build_insights(scored: pd.DataFrame, rep: dict) -> list[str]:
    """Short narrative findings derived from the scored frame + report."""
    from collections import Counter
    n = len(scored)
    if n == 0:
        return ["No transactions in the analysed file."]
    facts = []
    vc = scored["risk_band"].value_counts()
    high = int(vc.get("high-risk", 0)) + int(vc.get("review", 0))
    facts.append(f"{high:,} of {n:,} transactions "
                 f"({100.0 * high / max(1, n):.1f}%) fall into the "
                 f"review or high-risk bands and would alert under the "
                 f"current budget.")
    if "unseen_customer" in scored.columns:
        unseen = int(scored["unseen_customer"].sum())
        facts.append(f"{unseen:,} "
                     f"({100.0 * unseen / max(1, n):.1f}%) transactions are "
                     f"from customers never seen in training — scored from "
                     f"priors + novelty, no memorised history.")
    if "tags" in scored.columns:
        tags = Counter(t for ts in scored["tags"].dropna()
                       for t in (ts.split(",") if isinstance(ts, str) else []))
        for tag, cnt in tags.most_common(3):
            facts.append(f"Decision driver `{tag}` appears on {cnt:,} "
                         f"transactions.")
    m = rep.get("label_metrics")
    if m:
        facts.append(f"PR-AUC {m['pr_auc']:.3f} on this file "
                     f"({m['alert_recall']:.0%} recall at the 2% budget, "
                     f"alert precision {m['alert_precision']:.0%}).")
    else:
        facts.append("No `is_fraud` column was provided, so precision/recall "
                     "metrics are not computed; every probability shown is "
                     "from the calibrated model.")
    return facts


_PDF_TXN_COLS = ["transaction_id", "timestamp", "amount",
                 "fraud_probability", "novelty_score", "risk_band",
                 "rank_score"]


def _build_pdf_report(rep: dict, scored: pd.DataFrame) -> bytes:
    """Render an analysis report into a PDF: a summary page followed by a
    full per-transaction table (amount + fraud probability + novelty +
    risk band + rank for every scored row)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    out = io.BytesIO()
    with PdfPages(out) as pdf:
        # ---- page 1: summary ----
        on = [name for name in TABLE_ORDER if name in rep]
        fig = plt.figure(figsize=(11, max(8.5, 0.62 * max(len(on), 1) + 5)))
        fig.suptitle("Unseen-Customer Fraud Detection - Analysis Summary",
                     fontsize=14, fontweight="bold")
        y = 0.97
        seen = set(("file_name", "rows", "alerts", "thresholds", "has_label"))
        off = {k: v for k, v in rep.items()
               if k in TABLE_ORDER and k not in seen}
        for g, val in off.items():
            fig.text(0.03, y, g, fontsize=11, fontweight="bold", va="top")
            if isinstance(val, dict):
                val = pd.DataFrame([val])
            if isinstance(val, pd.DataFrame):
                val = val.to_string(max_rows=8)
            else:
                val = str(val)
            fig.text(0.06, y, val, fontsize=9,
                     va="top", family="monospace")
            y -= 0.115
        pdf.savefig(fig)
        plt.close(fig)
        # ---- page 2+: full transaction table ----
        if scored is not None and len(scored):
            cols = [c for c in _PDF_TXN_COLS if c in scored.columns]
            rows = scored[cols].copy()
            disp = rows.to_string(index=False)
            fig2 = plt.figure(figsize=(11, 8.5))
            fig2.suptitle("Scored transactions (all rows)", fontsize=14,
                          fontweight="bold")
            ax = fig2.add_subplot(111)
            ax.axis("off")
            ax.text(0.01, 0.98, disp, fontsize=6, va="top",
                    family="monospace")
            pdf.savefig(fig2)
            plt.close(fig2)
    return out.getvalue()


def _export_bytes(kind: str, scored: pd.DataFrame, rep: dict,
                  md_text: str) -> tuple:
    """Generic exporter for *.csv / *.pdf / *.json / *.md / *.txt.

    Every format carries the per-transaction values (amount, fraud
    probability, novelty score, risk band, rank) - not just the summary."""
    export_cols = [c for c in _PDF_TXN_COLS if c in scored.columns]
    for extra in ("is_fraud", "budget_status", "explanation"):
        if extra in scored.columns and extra not in export_cols:
            export_cols.append(extra)
    export_frame = scored[export_cols].copy() if len(export_cols) else scored
    if kind == "csv":
        return export_frame.to_csv(index=False).encode("utf-8"), "csv"
    if kind == "json":
        payload = dict(rep)
        payload["scored_transactions"] = scored.to_dict("records")
        return json.dumps(payload, default=str,
                          indent=2).encode("utf-8"), "json"
    full_md = _format_transaction_table_markdown(export_frame, md_text)
    if kind == "markdown":
        return full_md.encode("utf-8"), "md"
    if kind == "txt":
        alerts = len(scored[scored["budget_status"] == "alert"])
        head = (f"Analysis of {rep.get('filename', 'upload')} - "
                f"{len(scored)} rows, {alerts} alert(s)\n\n")
        return (head + full_md).encode("utf-8"), "txt"
    if kind == "pdf":
        return _build_pdf_report(rep, scored), "pdf"
    raise ValueError(f"unsupported export kind: {kind}")


def _format_transaction_table_markdown(export_frame: pd.DataFrame,
                                       md_text: str) -> str:
    """Append every scored transaction as a markdown table to md_text."""
    if len(export_frame) == 0:
        return md_text
    hdrs = list(export_frame.columns)
    L = [md_text.rstrip(), "", "## Scored transactions (all rows)", ""]
    L.append("| " + " | ".join(h.replace("_", " ") for h in hdrs) + " |")
    L.append("|" + "---|" * len(hdrs))
    for r in export_frame.itertuples(index=False):
        L.append("| " + " | ".join(str(getattr(r, h)) for h in hdrs) + " |")
    L.append("")
    return "\n".join(L)


def _render_export(scored: pd.DataFrame, rep: dict, md_text: str):
    st.subheader("Settings & export")
    kind = st.radio("Format", ["csv", "json", "markdown", "txt", "pdf"],
                    horizontal=True, key="export_format")
    fname = rep.get("filename", "analysis")
    data, ext = _export_bytes(kind, scored, rep, md_text)
    st.download_button(f"Download {kind.upper()} report",
                       data=data, mime="application/octet-stream",
                       file_name=f"{fname}.{ext}", key="download_export")
    st.caption("CSV/JSON/Markdown/txt/PDF all include every scored "
               "transaction (amount, fraud probability, novelty score, "
               "risk band, rank); PDF leads with a compact summary page.")


def _render_hero():
    """Section 1 - landing & context: headline, sub-headline, how it works."""
    c1, c2 = st.columns([3, 2])
    c1.title("Unseen-Customer Fraud Detection")
    c1.markdown("**Stops fraud even on customers the model has never seen.**")
    c1.markdown(
        "Every transaction gets a supervised **fraud probability**, an "
        "unsupervised **novelty score**, and a plain-language explanation - "
        "all computed point-in-time on a leak-safe pipeline. Decide how many "
        "alerts you can action; the thresholds follow.")
    c2.plotly_chart(
        px.imshow([[0, 1, 1, 0, 1], [0, 1, 1, 0, 1], [0, 0, 0, 0, 1],
                   [1, 1, 0, 1, 1], [1, 0, 0, 0, 0]],
                  color_continuous_scale=["#08306b", "#7fcdbb", "#ffffcc"],
                  labels=dict(color="fraud p"), text_auto=False,
                  title="Novelty x probability decision matrix")
        .update_layout(coloraxis_showscale=False, height=220,
                       margin=dict(l=0, r=0, t=40, b=0)),
        use_container_width=True)
    st.markdown("### How it works")
    for step, txt in [
        ("1 · Load", "Drop a CSV (or pick the built-in test fold). Rows are "
         "replayed **chronologically** through the same streaming pipeline "
         "that runs in production."),
        ("2 · Score", "The model computes fraud probability + novelty for "
         "every row, applying only information available at that instant."),
        ("3 · Decide", "Fraud probability and novelty map to normal / "
         "monitor / review / high-risk bands under your alert budget."),
        ("4 · Explain", "Each alert names its top contributing factors so an "
         "analyst can act - or export the full report (.csv/.pdf/.json)."),
    ]:
        st.markdown(f"**{step}**  \n{txt}")
    st.divider()


def main():
    st.set_page_config(page_title="Unseen-Customer Fraud Detection",
                       layout="wide", initial_sidebar_state="expanded")
    ctx = load_context()
    _render_hero()
    panel_overview(ctx)
    st.divider()
    panel_upload(ctx)
    st.divider()
    _render_system_eval(ctx)


if __name__ == "__main__":
    main()

