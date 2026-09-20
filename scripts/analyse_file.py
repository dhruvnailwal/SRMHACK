"""Run the full fraud-analysis pipeline on a transaction CSV locally - no
browser, no FastAPI server needed (models are loaded from models/).

Usage:
    python scripts/analyse_file.py data/processed/test_fold_file.csv
    python scripts/analyse_file.py my__data.csv --output analysis_out
    python scripts/analyse_file.py data.csv --budget 5.0 --explain --outdir out/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dashboard.app import (  # noqa: E402
    _add_unseen_flag,
    _build_analysis_report,
    _build_insights,
    _export_bytes,
    _format_analysis_markdown,
    _metric_summary_from_scored,
    _recompute_bands,
    _score_upload_impl,
    load_context,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("csv", help="Path to the transaction CSV to analyse")
    ap.add_argument("--explain", action="store_true",
                    help="Add per-row plain-language explanations (slower)")
    ap.add_argument("--budget", type=float, default=None,
                    help="Alert budget %% of volume (default from config)")
    ap.add_argument("--outdir", default=None,
                    help="Directory for report files "
                         "(default: alongside the CSV, name_analysis/)")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = open(csv_path, "rb").read()
    fname = os.path.basename(csv_path)

    ctx = load_context()

    print(f"Scoring {fname} ({len(pd.read_csv(csv_path)):,} rows) "
          f"locally with {'explanations' if args.explain else 'no explanations'}...")
    res = _score_upload_impl(data, fname, args.explain, False)
    scored = res["scored"]

    scored = _add_unseen_flag(scored, ctx["train_customers"])
    budget = (args.budget if args.budget is not None
              else float(ctx["cfg"].risk["alert_budget_pct"]))
    scored, newthr = _recompute_bands(scored, ctx, budget)

    model_metrics = _metric_summary_from_scored(scored)
    rep = _build_analysis_report(res, ctx)
    rep["valid"] = True
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
    rep["label_metrics"] = model_metrics
    rep["insights"] = _build_insights(scored, rep)
    md_text = _format_analysis_markdown(rep)

    outdir = args.outdir or os.path.join(os.path.dirname(csv_path),
                                         fname.replace(".csv", "") + "_analysis")
    os.makedirs(outdir, exist_ok=True)

    for kind in ("csv", "json", "markdown", "txt"):
        data_b, ext = _export_bytes(kind, scored, rep, md_text)
        with open(os.path.join(outdir, f"{fname[:-4]}.{ext}"), "wb") as fh:
            fh.write(data_b)
    pdf_b, _ = _export_bytes("pdf", scored, rep, md_text)
    with open(os.path.join(outdir, f"{fname[:-4]}.pdf"), "wb") as fh:
        fh.write(pdf_b)

    print("\nReport written to", outdir)
    print("  " + ", ".join(
        f"{fname[:-4]}.{ext}" for ext in ("csv", "json", "md", "txt", "pdf")))

    print("\n==== SUMMARY ====")
    print(f"Rows scored     : {len(scored):,}")
    print(f"Alert budget    : {budget:.1f}%")
    print(f"Alerts fired    : {int((scored['budget_status'] == 'alert').sum()):,} "
          f"({rep['alerts']['alert_rate_pct']:.2f}%)")
    print(f"Risk bands      : "
          + " | ".join(f"{b}: {rep['alerts'][b]:,}"
                       for b in ("normal", "monitor", "review", "high_risk")))
    print(f"Thresholds      : {json.dumps(newthr)}")
    if model_metrics:
        print("\nMetrics vs fraud label:")
        for k, v in model_metrics.items():
            print(f"  {k:>18}: {v}")
    vc = scored["unseen_customer"].value_counts()
    print(f"\nKnown customers : {int(vc.get(0, 0)):,} txn")
    print(f"Unseen customers: {int(vc.get(1, 0)):,} txn")
    print("\nInsights:")
    for line in rep["insights"]:
        print("  - " + line)


if __name__ == "__main__":
    main()