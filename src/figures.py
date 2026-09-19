"""All 15 required visualisations, saved to ``reports/figures``.

Professional titles, axis labels, legends and readable formatting throughout.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import Settings

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12, "axes.labelsize": 10,
    "figure.facecolor": "white",
})

RISK_COLORS = {"normal": "#2e7d32", "monitor": "#f9a825",
               "review": "#ef6c00", "high-risk": "#c62828"}
BAND_ORDER = ["normal", "monitor", "review", "high-risk"]


def _save(fig, name, cfg: Settings):
    d = cfg.figures_dir()
    os.makedirs(d, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(d, name), dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_all_figures(df_eda: pd.DataFrame, scores: pd.DataFrame,
                     pr_data: dict, confusion: np.ndarray,
                     shap_imp: pd.DataFrame, alert_series,
                     cohort_summary: pd.DataFrame, cfg: Settings) -> list[str]:
    written = []
    sns.set_style("whitegrid")

    # 1. Fraud vs non-fraud -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df_eda["is_fraud"].value_counts().sort_index()
    bars = ax.bar(["non-fraud", "fraud"], counts.values,
                  color=["#90caf9", "#e57373"])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom")
    ax.set_title("1. Fraud vs Non-Fraud Distribution")
    ax.set_xlabel("Class"); ax.set_ylabel("Transactions")
    _save(fig, "01_class_distribution.png", cfg)
    written.append("01_class_distribution.png")

    # 2. Amount distribution -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, c in [(0, "#90caf9"), (1, "#e57373")]:
        mask = df_eda["is_fraud"] == lbl
        ax.hist(np.log1p(df_eda.loc[mask, "amount"]), bins=60, alpha=0.55,
                label="fraud" if lbl else "non-fraud", color=c)
    ax.set_title("2. Transaction Amount Distribution (log scale)")
    ax.set_xlabel("log(amount+1)"); ax.set_ylabel("count"); ax.legend()
    _save(fig, "02_amount_distribution.png", cfg)
    written.append("02_amount_distribution.png")

    # 3. Fraud rate by hour --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    g = df_eda.groupby("transaction_hour")["is_fraud"].agg(["count", "mean"])
    ax.bar(g.index, g["mean"] * 100, color="#5c6bc0")
    ax.set_title("3. Fraud Rate by Transaction Hour")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Fraud rate (%)")
    ax.set_ylim(0, max(g["mean"] * 100) * 1.15)
    _save(fig, "03_fraud_rate_by_hour.png", cfg)
    written.append("03_fraud_rate_by_hour.png")

    # 4. New vs known customers ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    g = df_eda.groupby("new_customer")["is_fraud"].agg(["count", "mean"])
    bars = ax.bar(["known customer", "new customer"], g["mean"] * 100,
                  color=["#66bb6a", "#ef5350"])
    for b, row in zip(bars, g.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{row[1]['count']:,}\n{row[1]['mean']*100:.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("4. Fraud Rate for New vs Known Customers")
    ax.set_ylabel("Fraud rate (%)")
    _save(fig, "04_new_vs_known_customers.png", cfg)
    written.append("04_new_vs_known_customers.png")

    # 5. New vs known devices ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    g = df_eda.groupby("new_device")["is_fraud"].agg(["count", "mean"])
    bars = ax.bar(["known device", "new device"], g["mean"] * 100,
                  color=["#66bb6a", "#ef5350"])
    for b, row in zip(bars, g.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{row[1]['count']:,}\n{row[1]['mean']*100:.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("5. Fraud Rate for New vs Known Devices")
    ax.set_ylabel("Fraud rate (%)")
    _save(fig, "05_new_vs_known_devices.png", cfg)
    written.append("05_new_vs_known_devices.png")

    # 6. Fraud probability distribution -------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, c in [(0, "#90caf9"), (1, "#e57373")]:
        ax.hist(scores.loc[scores["is_fraud"] == lbl, "fraud_probability"],
                bins=40, alpha=0.55, label="fraud" if lbl else "non-fraud",
                color=c)
    ax.set_title("6. Fraud Probability Distribution (calibrated)")
    ax.set_xlabel("fraud_probability"); ax.set_ylabel("count"); ax.legend()
    _save(fig, "06_fraud_probability.png", cfg)
    written.append("06_fraud_probability.png")

    # 7. Novelty score distribution -----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, c in [(0, "#90caf9"), (1, "#e57373")]:
        ax.hist(scores.loc[scores["is_fraud"] == lbl, "novelty_score"],
                bins=40, alpha=0.55, label="fraud" if lbl else "non-fraud",
                color=c)
    ax.set_title("7. Novelty Score Distribution")
    ax.set_xlabel("novelty_score (0-1, ECDF-normalised)")
    ax.set_ylabel("count"); ax.legend()
    _save(fig, "07_novelty_distribution.png", cfg)
    written.append("07_novelty_distribution.png")

    # 8. Probability vs novelty scatter -------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(scores["fraud_probability"], scores["novelty_score"],
                    c=scores["is_fraud"], cmap="RdYlGn_r", alpha=0.35,
                    s=8, rasterized=True)
    ax.set_title("8. Fraud Probability vs Novelty Score")
    ax.set_xlabel("fraud_probability"); ax.set_ylabel("novelty_score")
    ax.axvline(0.15, c="gray", ls="--", lw=0.8, label="review threshold")
    ax.axhline(0.90, c="gray", ls=":", lw=0.8, label="novelty_high")
    ax.legend(); fig.colorbar(sc, label="is_fraud")
    _save(fig, "08_prob_vs_novelty.png", cfg)
    written.append("08_prob_vs_novelty.png")

    # 9. Risk-band distribution ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    order = [b for b in BAND_ORDER if (scores["risk_band"] == b).any()]
    counts = [int((scores["risk_band"] == b).sum()) for b in order]
    bars = ax.bar(order, counts, color=[RISK_COLORS[b] for b in order])
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom")
    ax.set_title("9. Risk-Band Distribution")
    ax.set_xlabel("Risk band"); ax.set_ylabel("Transactions")
    _save(fig, "09_risk_band_distribution.png", cfg)
    written.append("09_risk_band_distribution.png")

    # 10. Precision-recall curve ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pr_data["recall"], pr_data["precision"], color="#1565c0", lw=2)
    ax.set_title(f"10. Precision-Recall Curve (PR-AUC={pr_data['pr_auc']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    _save(fig, "10_precision_recall_curve.png", cfg)
    written.append("10_precision_recall_curve.png")

    # 11. Confusion matrix ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                xticklabels=["non-fraud", "fraud"],
                yticklabels=["non-fraud", "fraud"], ax=ax)
    ax.set_title("11. Confusion Matrix (alert decision)"); 
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    _save(fig, "11_confusion_matrix.png", cfg)
    written.append("11_confusion_matrix.png")

    # 12. SHAP feature importance ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 8))
    d = shap_imp.head(20).iloc[::-1]
    ax.barh(d["feature"], d["mean_abs_contribution"], color="#5c6bc0")
    ax.set_title("12. SHAP Feature Importance (mean |TreeSHAP| contribution)")
    ax.set_xlabel("mean |contribution|")
    _save(fig, "12_shap_feature_importance.png", cfg)
    written.append("12_shap_feature_importance.png")

    # 13. Alert volume over time ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    if alert_series is not None and len(alert_series):
        ax.plot(alert_series.index, alert_series.values, color="#c62828", lw=1.5)
        ax.fill_between(alert_series.index, alert_series.values, alpha=0.15,
                        color="#c62828")
        ax.set_title("13. Alert Volume over Time (daily)")
        ax.set_xlabel("Date"); ax.set_ylabel("Alerts per day")
    else:
        ax.text(0.5, 0.5, "no alert series available", ha="center", va="center")
    _save(fig, "13_alert_volume_over_time.png", cfg)
    written.append("13_alert_volume_over_time.png")

    # 14. Known vs unseen-customer performance --------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    if "cohort" in cohort_summary.columns:
        wide = cohort_summary.set_index("cohort")
        cols = [c for c in ["pr_auc", "recall@budget2", "roc_auc"] if c in wide]
        if cols:
            wide[cols].plot(kind="bar", ax=ax)
            ax.set_title("14. Performance: Known vs Unseen Customers")
            ax.set_ylabel("Metric value"); ax.set_xticklabels(ax.get_xticklabels(),
                                                              rotation=25)
            ax.legend(loc="best")
    _save(fig, "14_known_vs_unseen.png", cfg)
    written.append("14_known_vs_unseen.png")

    # 15. False-positive analysis for first-time customers --------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    if "first_time_kind" in cohort_summary.columns:
        ft = cohort_summary[cohort_summary["first_time_kind"].notna()]
        if len(ft):
            labs = ft["first_time_kind"].tolist()
            vals = ft["fpr_first_time"].fillna(0).tolist()
            bars = ax.bar(labs, [v * 100 for v in vals], color="#8e24aa")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v*100:.2f}%", ha="center", va="bottom", fontsize=9)
            ax.set_title("15. False-Positive Rate on Legitimate First-Time "
                         "Transactions")
            ax.set_ylabel("Alerted legit first-time txns (%)")
    else:
        ax.text(0.5, 0.5, "no first-time cohort data", ha="center", va="center")
    _save(fig, "15_fp_first_time.png", cfg)
    written.append("15_fp_first_time.png")

    return written