"""Evaluation: metrics, cohort reports, reliability and the final report.

Every headline number is reported on the *test* fold and separately for the
cohorts that matter for this project's thesis:

* known customers  vs  unseen customers
* first-time customers / devices / merchants (false-positive analysis)

We also report reliability (Brier, expected calibration error, reliability
data), novelty diagnostics (novelty AUROC on the cold cohort, false-novelty
rate) and operational metrics (alert volume, recall/precision at fixed alert
budgets). Aggregate metrics never hide poor unseen-customer performance.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix,
                             brier_score_loss)

from src.config import Settings


# ------------------------------------------------------------------ helpers
def add_cohort_flags(scores: pd.DataFrame, train_customers: set,
                     feats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach evaluation cohort flags to a scored stream."""
    out = scores.copy()
    out["unseen_customer"] = (~out["customer_id"].isin(train_customers)).astype(int)
    out["known_customer"] = 1 - out["unseen_customer"]
    if feats is not None:
        cols = [c for c in ["customer_history_count", "device_history_count",
                            "merchant_history_count"] if c in feats.columns]
        if cols:
            sub = feats[["transaction_id"] + cols].copy()
            out = out.merge(sub, on="transaction_id", how="left",
                            suffixes=("", "_feat"))
            out["first_time_customer"] = (out["customer_history_count"] == 0).astype(int)
            out["first_time_device"] = (out["device_history_count"] == 0).astype(int)
            out["first_time_merchant"] = (out["merchant_history_count"] == 0).astype(int)
            out["first_time_any"] = (
                (out["first_time_customer"] == 1) | (out["first_time_device"] == 1)
                | (out["first_time_merchant"] == 1)
            ).astype(int)
    return out


def recall_precision_at_budget(y, score, budget_pct, n_total):
    """Fixed alert budget: alert the top ``budget_pct``% by score."""
    k = max(1, int(round(n_total * budget_pct / 100.0)))
    order = np.argsort(-np.asarray(score), kind="mergesort")
    alert_mask = np.zeros(len(y), dtype=bool)
    alert_mask[order[:k]] = True
    tp = int(((y == 1) & alert_mask).sum())
    n_alerts = int(alert_mask.sum())
    n_fraud = max(1, int((y == 1).sum()))
    return {"budget_pct": budget_pct, "alerts": n_alerts,
            "recall": tp / n_fraud,
            "precision": tp / n_alerts if n_alerts else 0.0,
            "tp": tp, "fp": n_alerts - tp}


def _cohort_block(df, p, y, n_all):
    block = {"n_rows": int(len(df)), "n_fraud": int(y.sum()),
             "fraud_rate": round(float(y.mean()) if len(y) else 0.0, 4)}
    if len(np.unique(y)) >= 2:
        block["pr_auc"] = round(float(average_precision_score(y, p)), 4)
        block["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
        block["brier"] = round(float(brier_score_loss(y, p)), 4)
        block["novelty_auroc"] = round(
            float(roc_auc_score(y, df["novelty_score"].to_numpy())), 4)
    else:
        for k in ("pr_auc", "roc_auc", "brier", "novelty_auroc"):
            block[k] = None
    r = recall_precision_at_budget(y, p, 2.0, n_all)
    block["recall@budget2"] = round(r["recall"], 4)
    block["precision@budget2"] = round(r["precision"], 4)
    nov = df["novelty_score"].to_numpy()
    legit = y == 0
    block["false_novelty_rate"] = (round(float((nov[legit] >= 0.90).mean()), 4)
                                   if legit.sum() else None)
    return block


def cohort_report(scores: pd.DataFrame, train_customers: set,
                  feats: pd.DataFrame | None = None) -> dict:
    df = add_cohort_flags(scores, train_customers, feats)
    n_all = len(df)
    out = {}
    idx = {"all": (slice(None), "all"),
           "known_customers": (df["known_customer"] == 1, "known_customers"),
           "unseen_customers": (df["unseen_customer"] == 1, "unseen_customers"),
           "first_time_customers": (df["first_time_customer"] == 1, "first_time_customers"),
           "new_devices": (df["first_time_device"] == 1, "new_devices"),
           "new_merchants": (df["first_time_merchant"] == 1, "new_merchants")}
    for key, (mask, name) in idx.items():
        sub = df[mask]
        if len(sub) == 0:
            continue
        out[key] = _cohort_block(sub, sub["fraud_probability"].to_numpy(),
                                 sub["is_fraud"].to_numpy(), n_all)
        out[key]["cohort"] = name
    return out


def alert_metrics(scores: pd.DataFrame, budget_col="budget_status") -> dict:
    warned = scores[scores[budget_col] == "alert"] if budget_col in scores \
        else scores[scores["alert"] == True].copy()  # noqa: E712
    n = len(scores)
    y = scores["is_fraud"].to_numpy()
    out = {"total_transactions": int(n),
           "alerts_generated": int(len(warned)),
           "alert_rate_pct": round(100.0 * len(warned) / max(1, n), 4),
           "high_risk_alerts": int((warned["risk_band"] == "high-risk").sum())
           if len(warned) else 0,
           "review_alerts": int((warned["risk_band"] == "review").sum())
           if len(warned) else 0,
           "monitor_logged": int((scores["risk_band"] == "monitor").sum())}
    if len(warned):
        hit = warned["is_fraud"].to_numpy()
        tp = int(hit.sum())
        out["alert_precision"] = round(float(hit.mean()), 4)
        out["alert_recall"] = round(tp / max(1, int(y.sum())), 4)
        out["alerts_per_caught_fraud"] = round(len(warned) / max(1, tp), 2)
    return out


def false_positive_first_time(scores: pd.DataFrame) -> dict:
    out = {}
    for col in ["first_time_customer", "first_time_device",
                "first_time_merchant", "first_time_any"]:
        if col not in scores:
            continue
        sub = scores[(scores[col] == 1) & (scores["is_fraud"] == 0)]
        alerted = sub["budget_status"] == "alert" if "budget_status" in sub \
            else sub["alert"]
        out[col] = {"legit_first_time_rows": int(len(sub)),
                    "fp_alerts": int(alerted.sum()),
                    "false_positive_rate": round(
                        float(alerted.mean()) if len(sub) else 0.0, 4)}
    return out


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return round(float(ece), 5)


def pr_curve_data(y, p) -> dict:
    prec, rec, _ = precision_recall_curve(y, p)
    return {"precision": prec.tolist(), "recall": rec.tolist(),
            "pr_auc": round(float(average_precision_score(y, p)), 4)}


def classification_block(y, p, decision) -> dict:
    out = {}
    pr = precision_recall_curve(y, p)
    out["pr_auc"] = round(float(average_precision_score(y, p)), 4)
    out["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
    cm = confusion_matrix(y, decision)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out.update({"precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(f1, 4),
                "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
                "brier": round(float(brier_score_loss(y, p)), 4),
                "ece": expected_calibration_error(y, p)})
    return out


def write_evaluation_report(report: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "evaluation_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    md = format_metrics_markdown(report)
    with open(os.path.join(out_dir, "02_evaluation_report.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)
    return md


def format_metrics_markdown(report: dict) -> str:
    L = []
    L.append("# Evaluation Report")
    L.append("")
    L.append(f"Run info: seed={report.get('seed')}  "
             f"primary model={report.get('primary_model')}  "
             f"calibration={report.get('calibration_method')}")
    L.append("")

    L.append("## Classification (test fold)")
    L.append("")
    c = report.get("classification", {})
    L.append("| metric | value |")
    L.append("|---|---|")
    for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "ece"):
        if k in c:
            L.append(f"| {k} | {c[k]} |")
    L.append("")

    L.append("## Reliability")
    L.append("")
    L.append(f"- Brier score: **{c.get('brier')}**")
    L.append(f"- Expected calibration error (ECE): **{c.get('ece')}**")
    L.append("")

    L.append("## Alert operations")
    L.append("")
    a = report.get("alerts", {})
    L.append("| metric | value |")
    L.append("|---|---|")
    for k, v in a.items():
        L.append(f"| {k} | {v} |")
    L.append("")

    L.append("## Cohorts")
    L.append("")
    L.append("| cohort | n_rows | n_fraud | fraud_rate | PR-AUC | ROC-AUC | "
             "recall@2% | precision@2% | novelty_AUROC | FNR@0.9 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, b in report.get("cohorts", {}).items():
        L.append("| {} | {} | {} | {:.3f} | {} | {} | {} | {} | {} | {} |".format(
            name, b["n_rows"], b["n_fraud"], b["fraud_rate"],
            b.get("pr_auc"), b.get("roc_auc"),
            b.get("recall@budget2"), b.get("precision@budget2"),
            b.get("novelty_auroc"), b.get("false_novelty_rate")))
    L.append("")

    L.append("## First-time false positives")
    L.append("")
    for k, v in report.get("fp_first_time", {}).items():
        L.append(f"- {k}: legit rows={v['legit_first_time_rows']:,}, "
                 f"FP alerts={v['fp_alerts']}, FP rate={v['false_positive_rate']}")
    L.append("")

    L.append("## Leakage & data-quality checks")
    L.append("")
    for k, v in report.get("leakage", {}).items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L.append("## Feature availability (cold-start rate)")
    L.append("")
    for k, v in report.get("feature_availability", {}).items():
        L.append(f"- {k}: {v}")
    return "\n".join(L)