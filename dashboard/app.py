"""Streamlit dashboard for the Unseen-Customer Fraud Detection project.

Run from the project root:
    streamlit run dashboard/app.py

Sections
--------
1. Landing & context       - what the model does, how it works, hero
2. Figure gallery          - the 15 evaluation figures rendered as PNGs
3. Evaluation              - cohort tables, first-time FP analysis, leakage
4. System evaluation & spl - PR-AUC / recall@budget, known vs unseen,
                             leakage-safety compliance panel
5. Upload File & Analyse   - drag-and-drop / browse a CSV, then Analyse:
                             per-transaction fraud probability gauge, novelty
                             score, color-coded risk band, top reasons;
                             system-evaluation metrics on the file (incl.
                             known vs unseen split); adjustable alert-budget
                             slider that re-derives Review/High-Risk
                             thresholds; exports in .csv / .pdf / .json /
                             .md / .txt. Scoring can run in-process or via
                             the FastAPI service (``python -m uvicorn
                             api.main:app --reload``).

Every number shown here is read from the artifacts produced by
``python run_pipeline.py`` (models/, data/processed/, reports/).
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

FIG_CAPTIONS = {
    "01_class_distribution.png": "Fraud vs non-fraud class distribution.",
    "02_amount_distribution.png": "Amount distribution (log scale) by class.",
    "03_fraud_rate_by_hour.png": "Fraud rate across the hour of day.",
    "04_new_vs_known_customers.png": "Fraud rate for new vs known customers.",
    "05_new_vs_known_devices.png": "Fraud rate for new vs known devices.",
    "06_fraud_probability.png": "Calibrated fraud probability by class.",
    "07_novelty_distribution.png": "Novelty score distribution by class.",
    "08_prob_vs_novelty.png": "Probability vs novelty decision landscape.",
    "09_risk_band_distribution.png": "Volume per risk band on the test fold.",
    "10_precision_recall_curve.png": "Precision-recall curve (PR-AUC).",
    "11_confusion_matrix.png": "Confusion matrix of the alert decision.",
    "12_shap_feature_importance.png": "Global TreeSHAP feature importance.",
    "13_alert_volume_over_time.png": "Daily alert volume on the test fold.",
    "14_known_vs_unseen.png": "Performance: known vs unseen customers.",
    "15_fp_first_time.png": "False-positive rate on first-time transactions.",
}


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
    ctx["model_comparison"] = json.load(
        open(os.path.join(models, "model_comparison.json")))
    ctx["model_name"] = ctx["model_comparison"]["best"]
    ctx["feature_names"] = json.load(
        open(os.path.join(models, "feature_names.json")))
    ctx["report"] = json.load(
        open(os.path.join(cfg.reports_dir(), "evaluation_report.json")))
    ctx["scored"] = pd.read_parquet(
        os.path.join(cfg.processed_dir(), "test_scores.parquet"))
    ctx["raw"] = pd.read_csv(cfg.raw_path(), parse_dates=["timestamp"])
    train_feats = pd.read_parquet(
        os.path.join(cfg.processed_dir(), "train_features.parquet"))
    ctx["train_customers"] = set(train_feats["customer_id"].astype(str))
    return ctx


def _metric_grid(cols, items):
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


def _run_with_factors(df: pd.DataFrame, proc) -> pd.DataFrame:
    """Score a sorted frame row-by-row while KEEPING the explanation
    columns (`top_contributing_features`, tags) that ``proc.run`` drops."""
    outputs = []
    for row in df.itertuples(index=False):
        txn = {"ts_sec": float(row.ts_sec), "amount": float(row.amount),
               "customer_id": row.customer_id,
               "merchant_id": row.merchant_id,
               "device_id": row.device_id,
               "transaction_hour": int(row.transaction_hour),
               "timestamp": str(row.timestamp)}
        res = proc.score_transaction(txn)
        res["transaction_id"] = row.transaction_id
        res["timestamp"] = str(row.timestamp)
        res["ts_sec"] = float(row.ts_sec)
        res["customer_id"] = row.customer_id
        res["merchant_id"] = row.merchant_id
        res["device_id"] = row.device_id
        res["amount"] = float(row.amount)
        res["is_fraud"] = int(row.is_fraud)
        outputs.append(res)
    return pd.DataFrame(outputs)


def panel_overview(ctx):
    st.header("Overview")
    st.markdown("""
**What this system does.** The model detects fraudulent transactions by
combining **two independent signals**:

1. **Fraud probability** – a supervised classifier (LightGBM, selected by
   PR-AUC on validation) trained on labelled transactions, whose output is
   *calibrated* so a score of 0.8 genuinely means ~80% fraud likelihood.
2. **Novelty score** – an unsupervised behavioural model (IsolationForest
   + ECDF) that measures how unusual a transaction is *compared to
   legitimate traffic*. 0.90 means "more atypical than 90% of legitimate
   transactions".

Both signals feed a **risk decision matrix**:

|                 | novelty low      | novelty high     |
|-----------------|------------------|------------------|
| **p low**       | `normal` → allow | `monitor` → log  |
| **p high**      | `review` → alert | `high-risk` → alert |

Every feature is computed **point-in-time** (no future information), priors
are fit on training data only, and the system is evaluated on a disjoint
*unseen-customer* holdout. Alerts run through a **budget controller** so the
analyst workload stays bounded, and each decision comes with a
plain-language explanation of its contributing factors.
""")

    rep = ctx["report"]
    m, s = ctx["model_comparison"], ctx["scored"]

    c1, c2, c3, c4 = st.columns(4)
    _metric_grid([c1, c2, c3, c4], [
        ("PR-AUC (test)", f"{rep['classification']['pr_auc']:.3f}"),
        ("ROC-AUC (test)", f"{rep['classification']['roc_auc']:.3f}"),
        ("ECE (test)", f"{rep['classification']['ece']:.4f}"),
        ("Alert rate", f"{rep['alerts']['alert_rate_pct']:.2f}%"),
    ])
    c5, c6, c7, c8 = st.columns(4)
    _metric_grid([c5, c6, c7, c8], [
        ("Alert precision", f"{rep['alerts'].get('alert_precision', 0):.3f}"),
        ("Alert recall", f"{rep['alerts'].get('alert_recall', 0):.3f}"),
        ("Alerts fired", f"{rep['alerts']['alerts_generated']:,}"),
        ("Test rows", f"{len(s):,}"),
    ])

    st.subheader("Model comparison (validation PR-AUC)")
    comp = pd.DataFrame(
        {"model": list(m["comparison"].keys()),
         "PR-AUC (val)": [v["pr_auc_val"] for v in m["comparison"].values()],
         "ROC-AUC (val)": [v["roc_auc_val"] for v in m["comparison"].values()]})
    st.dataframe(comp.sort_values("PR-AUC (val)", ascending=False),
                 use_container_width=True)
    st.caption(f"Primary model: **{m['best']}** (selected by PR-AUC on "
               f"validation, never tuned on test).")

    st.subheader("Data-integrity / parity")
    st.json(rep["parity"])
    st.subheader("Leakage checks")
    st.json(rep["leakage"])


def panel_figures(ctx):
    st.header("Figure gallery")
    fig_dir = ctx["cfg"].figures_dir()
    files = [f for f in sorted(os.listdir(fig_dir)) if f.endswith(".png")]
    sel = st.selectbox("Figure", files,
                       format_func=lambda f: f.split("_", 1)[1][:-4])
    st.image(os.path.join(fig_dir, sel), width=820)
    st.caption(FIG_CAPTIONS.get(sel, ""))


def panel_evaluation(ctx):
    rep = ctx["report"]
    st.header("Evaluation")

    st.subheader("Cohorts (test fold)")
    rows = []
    for name, b in rep["cohorts"].items():
        rows.append({"cohort": name, "n": b["n_rows"], "fraud": b["n_fraud"],
                     "PR-AUC": b.get("pr_auc"), "ROC-AUC": b.get("roc_auc"),
                     "recall@2%": b.get("recall@budget2"),
                     "novelty AUROC": b.get("novelty_auroc"),
                     "FNR@0.90%": b.get("false_novelty_rate"),
                     "fraud_rate": b["fraud_rate"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("False positives on legitimate first-time transactions")
    st.json(rep["fp_first_time"])

    st.subheader("Feature availability / cold-start rate")
    st.json(rep["feature_availability"])

    st.subheader("Classification metrics")
    st.json({k: rep["classification"][k] for k in
             ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier",
              "ece", "confusion_matrix")})


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
        df, schema = load_dataframe(cfg, tmp.name)
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
        df, schema = load_dataframe(cfg, tmp.name)
        df = valid_numeric(df)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    has_fraud = "is_fraud" in df.columns
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
    if "label_metrics" in rep:
        m = rep["label_metrics"]
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


def panel_upload(ctx):
    st.header("Upload & score your own data")
    st.caption("Drag & drop or browse a transaction CSV. The file is only "
               "inspected on upload; **nothing is analysed until you press "
               "Analyse**. Results stay in memory and can be downloaded as "
               "Markdown, TXT, JSON or CSV.")

    with st.expander("Expected columns (documented schema)"):
        st.markdown(SCHEMA_DOC)
        template = ctx["raw"].head(3).to_csv(index=False)
        st.download_button("Download a 3-row template", data=template,
                           file_name="template.csv", mime="text/csv")

    uploaded = st.file_uploader("Transaction file (CSV)", type=["csv"],
                                key="up_csv", help=SCHEMA_DOC)
    if uploaded is None:
        st.info("Drag a CSV here or click **Browse files** to inspect and "
                "analyse it with the trained model.")
        return

    # ---- inspection only (no fraud analysis) -----------------------------
    with st.spinner("Inspecting file ..."):
        info = validate_upload_bytes(uploaded.getvalue(), uploaded.name)
    st.success(f"File **{info['filename']}** inspected: "
               f"{info['rows']:,} rows, {len(info['columns'])} columns")
    st.caption(f"Timestamp range: {info['timestamp_range'][0]} → "
               f"{info['timestamp_range'][1]}")
    if info["missing_required"]:
        st.error("Missing required columns: " + ", ".join(
            info["missing_required"]))
    with st.expander("Schema mapping + preview"):
        st.json({"mapped": info["mapped"],
                 "unmapped_optional": info["unmapped"]})
        st.write("First 5 rows:")
        st.dataframe(pd.DataFrame(info["preview"]), use_container_width=True)

    c1, c2 = st.columns(2)
    with_expl = c1.toggle("Per-row explanations (slower)", value=False)
    with_budget = c2.toggle("Apply alert budget (2% of volume)", value=True)

    # ---- analysis happens ONLY on this click -----------------------------
    if not st.button("Analyse file", type="primary"):
        st.info("Press **Analyse file** to run the fraud-detection analysis "
                "on this file now.")
        return

    with st.spinner("Analysing every transaction (features → fraud "
                    "probability → novelty → risk) ..."):
        try:
            res = score_upload_bytes(uploaded.getvalue(), uploaded.name,
                                     with_expl, with_budget)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            st.error(f"Could not analyse the file: {exc}")
            return

    scored = res["scored"]
    rep = _build_analysis_report(res, ctx)
    rep["insights"] = _build_insights(scored, rep)
    md_text = _format_analysis_markdown(rep)
    st.success(f"**Analysis complete** – {rep['rows_scored']:,} rows from "
               f"`{res['filename']}`")

    _render_upload_results(scored, rep, md_text, ctx, with_expl)


def _render_upload_results(scored: pd.DataFrame, rep: dict, md_text: str,
                           ctx: dict, with_expl: bool):
    """Display the complete results of an analysis:
    evaluation metrics + readings, suspected-fraud list, visualisations,
    probability/novelty scores, and key insights + contributing factors."""
    alert_dec = scored["budget_status"] == "alert"
    n_alerts = int(alert_dec.sum())

    # ---------------------------------------------------------------- 1. metrics
    st.subheader("1. Complete model evaluation metrics & readings")
    c1, c2, c3, c4 = st.columns(4)
    _metric_grid([c1, c2, c3, c4], [
        ("Rows scored", f"{len(scored):,}"),
        ("Alerts fired", f"{n_alerts:,}"),
        ("High-risk", int((scored["risk_band"] == "high-risk").sum())),
        ("Review", int((scored["risk_band"] == "review").sum())),
    ])
    row2 = []
    if "label_metrics" in rep:
        m = rep["label_metrics"]
        row2 = [("PR-AUC (vs label)", f"{m['pr_auc']:.3f}"),
                ("ROC-AUC (vs label)", f"{m['roc_auc']:.3f}"),
                ("Alert precision", f"{m['alert_precision']}"),
                ("Alert recall", f"{m['alert_recall']}")]
    else:
        row2 = [("Alert rate (of rows)", f"{rep['alerts']['alert_rate_pct']:.2f}%"),
                ("Novelty-only alerts", "-"), ("Monitor (logged)", "-"),
                ("Normal (allowed)", f"{rep['alerts']['normal']:,}")]
    a, b, c, d = st.columns(4)
    _metric_grid([a, b, c, d], row2)

    with st.expander("Model readings (thresholds & settings)"):
        st.markdown(f"**Model**: `{ctx['model_name']}`  ·  "
                    f"**Calibrated fraud probability** + **novelty score** "
                    f"combined by the risk engine.")
        st.json(rep["thresholds"])
        st.caption("p_review: probability to enter `review`; p_high: to enter "
                   "`high-risk`; novelty_high: novelty alone → `monitor`; "
                   "p_floor_for_novelty: minimum probability for the "
                   "novelty→review channel.")

    # ------------------------------------------------------ 2. suspected fraud
    st.subheader("2. Suspected fraud accounts & transactions")
    alerted = scored[alert_dec]
    if len(alerted) == 0:
        st.info("No transactions were flagged within the configured alert "
                "budget in this file.")
    else:
        tab_acc, tab_txn = st.tabs(["Suspected accounts", "Suspected "
                                    "transactions"])
        with tab_acc:
            acc = (alerted.groupby("customer_id")
                   .agg(n_alerts=("transaction_id", "count"),
                        flagged_amount=("amount", "sum"),
                        max_probability=("fraud_probability", "max"),
                        max_novelty=("novelty_score", "max"),
                        top_band=("risk_band", lambda s: s.mode().iloc[0]))
                   .sort_values("n_alerts", ascending=False).head(15)
                   .reset_index())
            st.dataframe(acc, use_container_width=True)
            st.caption("Accounts with the most flagged transactions, the "
                       "total amount involved and their highest "
                       "probability / novelty readings.")
        with tab_txn:
            tcols = ["transaction_id", "timestamp", "amount",
                     "fraud_probability", "novelty_score", "risk_band",
                     "rank_score"]
            tcols = [c for c in tcols if c in alerted.columns]
            if rep.get("has_label"):
                tcols = tcols + ["is_fraud"]
            txn = (alerted.sort_values("rank_score", ascending=False)
                   .head(25)[tcols])
            st.dataframe(txn, use_container_width=True)
            if rep.get("has_label"):
                st.caption("`is_fraud` = ground-truth label of the uploaded "
                           "file (when provided).")

    # ------------------------------------------------- 3. graphs & visualisations
    st.subheader("3. Graphs & visualizations")
    vc = _band_value_counts(scored)
    fig_bands = px.bar(x=vc.index, y=vc.values, color=vc.index,
                       color_discrete_map=RISK_COLORS,
                       labels={"x": "Risk band", "y": "Transactions"},
                       title="Risk-band distribution of the analysed file")
    fig_bands.update_layout(showlegend=False)
    sc_sort = scored.sort_values("rank_score", ascending=False)
    fig_scatter = px.scatter(
        sc_sort, x="fraud_probability", y="novelty_score",
        color="risk_band", color_discrete_map=RISK_COLORS,
        hover_data=["transaction_id", "amount", "budget_status"],
        title="Fraud probability vs novelty score (colored by risk band)")
    fig_p = px.histogram(
        scored, x="fraud_probability", nbins=50,
        color="is_fraud" if rep.get("has_label") else None,
        labels={"fraud_probability": "fraud_probability",
                "count": "transactions"},
        title="Fraud probability distribution")
    fig_n = px.histogram(
        scored, x="novelty_score", nbins=50,
        color="is_fraud" if rep.get("has_label") else None,
        labels={"novelty_score": "novelty_score", "count": "transactions"},
        title="Novelty score distribution")
    alert_dates = pd.to_datetime(alerted["ts_sec"], unit="s").dt.date \
        if len(alerted) else pd.Series(dtype=object)
    if len(alert_dates):
        daily = alert_dates.value_counts().sort_index()
        fig_vol = px.line(x=daily.index.astype(str), y=daily.values,
                          labels={"x": "Date", "y": "Alerts"},
                          title="Alert volume over time")
        fig_vol.update_traces(line=dict(color="#c62828", width=2))
    else:
        fig_vol = None
    st.plotly_chart(fig_bands, use_container_width=True)
    st.plotly_chart(fig_scatter, use_container_width=True)
    cA, cB = st.columns(2)
    cA.plotly_chart(fig_p, use_container_width=True)
    cB.plotly_chart(fig_n, use_container_width=True)
    if fig_vol is not None:
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.caption("No alerted transactions in this file, so the alert-volume "
                   "chart is empty.")

    # ---------------------------------------- 4. fraud probability & novelty scores
    st.subheader("4. Fraud probability & novelty scores")
    pa, pb, pc, pd = st.columns(4)
    _metric_grid([pa, pb, pc, pd], [
        ("Mean fraud probability", f"{scored['fraud_probability'].mean():.4f}"),
        ("Max fraud probability", f"{scored['fraud_probability'].max():.4f}"),
        ("Mean novelty score", f"{scored['novelty_score'].mean():.3f}"),
        ("Max novelty score", f"{scored['novelty_score'].max():.3f}"),
    ])
    if len(alerted):
        st.write("Top-10 riskiest transactions with their scores:")
        scols = ["transaction_id", "amount", "fraud_probability",
                 "raw_probability", "novelty_score", "rank_score"]
        scols = [c for c in scols if c in alerted.columns]
        st.dataframe(alerted.sort_values("rank_score", ascending=False)
                     .head(10)[scols], use_container_width=True)

    # ------------------------------------------- 5. key insights & factors
    st.subheader("5. Key insights & contributing factors")
    insights = _build_insights(scored, rep)
    for line in insights:
        st.markdown(f"- {line}")
    if with_expl and len(alerted):
        factors = Counter()
        for _, r in alerted.iterrows():
            for f in r.get("top_contributing_features") or []:
                if f.get("contribution", 0) > 0:
                    factors[f["feature"]] += 1
        if factors:
            st.write("**Top risk factors across alerted transactions** "
                     "(how often each factor pushed a transaction toward "
                     "fraud):")
            fdf = pd.DataFrame(factors.most_common(6),
                               columns=["factor", "alerts affected"])
            fig_f = px.bar(fdf, x="factor", y="alerts affected",
                           color="factor", color_discrete_map=RISK_COLORS)
            fig_f.update_layout(showlegend=False)
            st.plotly_chart(fig_f, use_container_width=True)
        st.write("**Explanations for the top-3 riskiest alerts:**")
        top3 = alerted.sort_values("rank_score", ascending=False).head(3)
        for _, r in top3.iterrows():
            st.markdown(f"- `{r['transaction_id']}`  ·  p="
                        f"{r['fraud_probability']:.3f}  ·  novel="
                        f"{r['novelty_score']:.2f}  ·  "
                        f"**{r['risk_band']}** — "
                        f"{r['explanation']}")

    # ---------------------------------------------------------------- download
    st.subheader("Download analysis")
    preview = scored[[c for c in ["transaction_id", "timestamp", "amount",
                                  "fraud_probability", "raw_probability",
                                  "novelty_score", "risk_band", "alert",
                                  "budget_status", "rank_score", "is_fraud"]
                      if c in scored.columns]].copy()
    if with_expl and "explanation" in scored:
        preview = preview.copy()
        preview["explanation"] = scored["explanation"]
    c_md, c_txt, c_js, c_csv = st.columns(4)
    stem = rep["filename"].rsplit(".", 1)[0]
    c_md.download_button("Markdown (.md)",
                         data=md_text.encode("utf-8"),
                         file_name=f"analysis_{stem}.md",
                         mime="text/markdown")
    c_txt.download_button("Plain text (.txt)",
                          data=md_text.encode("utf-8"),
                          file_name=f"analysis_{stem}.txt",
                          mime="text/plain")
    c_js.download_button("JSON (.json)",
                         data=json.dumps(rep, indent=2,
                                         default=str).encode("utf-8"),
                         file_name=f"analysis_{stem}.json",
                         mime="application/json")
    c_csv.download_button("Scored data (.csv)",
                          data=preview.to_csv(index=False).encode("utf-8"),
                          file_name="scored_output.csv",
                          mime="text/csv")


def _build_insights(scored: pd.DataFrame, rep: dict) -> list[str]:
    """Derive a short set of plain-language insights from the scored file."""
    from collections import Counter
    alert_dec = scored["budget_status"] == "alert"
    n_alerts = int(alert_dec.sum())
    n = len(scored)
    out = []
    if n == 0:
        return ["No rows to analyse."]
    out.append(f"{n:,} transactions were analysed; "
               f"**{n_alerts:,} ({100.0*n_alerts/n:.1f}%)** fired an alert "
               f"inside the configured alert budget.")
    nov_high = float(rep["thresholds"]["novelty_high"])
    nov_only = int((alert_dec & (scored["novelty_score"] >= nov_high)
                    & (scored["fraud_probability"] <
                       float(rep["thresholds"]["p_review"]))).sum())
    if nov_only:
        out.append(f"{nov_only} alerted transactions look risky *mostly "
                   f"because their behaviour is novel* (novelty ≥ {nov_high}) "
                   f"while the fraud probability alone was below the review "
                   f"threshold – the novelty channel is doing its job.")
    if n_alerts:
        amt_al = scored.loc[alert_dec, "amount"].median()
        amt_na = scored.loc[~alert_dec, "amount"].median()
        ratio = amt_al / amt_na if amt_na else float("nan")
        out.append(f"Alerted transactions are ~**{ratio:.1f}×** the median "
                   f"amount of allowed ones (median {amt_al:,.2f} vs "
                   f"{amt_na:,.2f}).")
        hours = (scored.loc[alert_dec, "ts_sec"] // 3600) % 24
        peak = int(hours.mode().iloc[0])
        out.append(f"Peak alerting hour is **{peak:02d}:00**.")
        top_c = scored.loc[alert_dec, "customer_id"].value_counts().idxmax()
        top_v = int(scored.loc[alert_dec, "customer_id"].value_counts().max())
        out.append(f"Customer **{top_c}** triggered the most alerts "
                   f"({top_v}).")
    if rep.get("has_label"):
        m = rep["label_metrics"]
        out.append(f"The file carries ground-truth labels: the model caught "
                   f"**{m['alert_recall']:.0%}** of the {m['fraud_rows_in_file']:,} "
                   f"frauds in the file (precision {m['alert_precision']:.0%}, "
                   f"PR-AUC {m['pr_auc']:.3f}).")
    return out


def main():
    st.set_page_config(page_title="Unseen-Customer Fraud Detection",
                       page_icon="🛡️", layout="wide")
    st.title("Unseen-Customer Fraud Detection")
    st.caption("Data-driven risk scoring combining calibrated fraud "
               "probability with behavioural novelty")

    with st.spinner("Loading artifacts ..."):
        ctx = load_context()

    pages = {"Overview": panel_overview, "Figure gallery": panel_figures,
             "Evaluation": panel_evaluation,
             "Upload File & Analyse": panel_upload}
    sel = st.sidebar.radio("Section", list(pages.keys()))
    if sel == "Upload File & Analyse":
        st.sidebar.caption("Your file is only analysed in memory; the data "
                           "stays on your machine.")
    pages[sel](ctx)


if __name__ == "__main__":
    main()