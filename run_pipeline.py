"""End-to-end training and evaluation pipeline.

Usage:
    python run_pipeline.py            # full run
    python run_pipeline.py --limit 20000   # fast smoke run (for developing)

Stages
------
1. load + schema-map + validate ...................... reports/01_data_validation.md
2. time split (60/20/20) + unseen-customer holdout
3. priors (segment/global baselines) fit on TRAIN only
4. full leakage-safe feature replay (with parity check)
5. supervised model comparison (LR / RF / LightGBM) -> primary by PR-AUC(val)
6. probability calibration (Platt vs Isotonic, Brier)
7. behavioural novelty model + ECDF (legit train / legit val)
8. risk thresholds selected on VALIDATION (budget-aware)
9. streaming replay of val + test (seeded state, one code path)
10. evaluation report: metrics, cohorts, reliability, leakage, figures 01-15
11. persist artifacts (models, thresholds, priors, scores, demo examples)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import get_settings
from src.data_loader import load_dataframe, valid_numeric
from src.data_validation import write_validation_report
from src.preprocessing import time_split, unseen_customer_split
from src.feature_engineering import (FeatureEngine, MODEL_FEATURES,
                                     NOVELTY_FEATURES)
from src.supervised_model import train_and_compare, save_model_bundle
from src.calibration import fit_best
from src.novelty_model import NoveltyModel
from src.risk_engine import RiskEngine, fit_thresholds
from src.explainability import Explainer
from src.streaming_processor import (StreamingProcessor,
                                     AlertBudgetController)
from src import evaluation as ev
from src import figures as figs


def _seed(cfg):
    np.random.seed(cfg.seed)


def _parity_metrics(combined: pd.DataFrame) -> dict:
    if "_parity_velocity_8min" not in combined:
        return {"note": "parity columns not produced (with_parity=False)"}
    m = combined.dropna(subset=["velocity_8min", "_parity_velocity_8min"])
    mr = float(m["velocity_8min"].eq(m["_parity_velocity_8min"]).mean())
    s = combined.dropna(subset=["amount_spike", "_parity_amount_spike"])
    ms = float(s["amount_spike"].eq(s["_parity_amount_spike"]).mean())
    return {"rows_checked": int(len(combined)),
            "velocity_8min_match_rate": round(mr, 4),
            "amount_spike_match_rate": round(ms, 4)}


def _feature_availability(feats: pd.DataFrame) -> dict:
    av = {}
    for col in ["customer_history_available", "first_time_customer",
                "new_device", "new_merchant"]:
        if col in feats:
            av[col + "_rate"] = round(float(feats[col].mean()), 4)
    return av


def _build_processor(cfg, model, priors, calibrator, novelty, risk,
                     budget=None, explainer=None):
    engine = FeatureEngine(cfg, priors)
    return StreamingProcessor(cfg, engine, model, feature_names=MODEL_FEATURES,
                              calibrator=calibrator, novelty=novelty,
                              risk_engine=risk, explainer=explainer,
                              budget=budget)


def _cohort_table(cohorts: dict, fp_pairs: list[dict]) -> pd.DataFrame:
    rows = []
    for name in ["all", "known_customers", "unseen_customers"]:
        b = cohorts.get(name)
        if not b:
            continue
        rows.append({"cohort": name, "pr_auc": b.get("pr_auc"),
                     "roc_auc": b.get("roc_auc"),
                     "recall@budget2": b.get("recall@budget2"),
                     "first_time_kind": None, "fpr_first_time": None})
    for p in fp_pairs:
        rows.append({"cohort": p["key"], "pr_auc": None, "roc_auc": None,
                     "recall@budget2": None,
                     "first_time_kind": p["key"],
                     "fpr_first_time": p["rate"]})
    return pd.DataFrame(rows)


def _alert_series(scores: pd.DataFrame) -> pd.Series | None:
    s = scores[scores["budget_status"] == "alert"] \
        if "budget_status" in scores else scores[scores["alert"]]
    if not len(s):
        return None
    dates = pd.to_datetime(s["ts_sec"], unit="s").dt.date
    return s["ts_sec"].groupby(dates).count().sort_index()


def _demo_examples(raw_df: pd.DataFrame, test_scores: pd.DataFrame,
                   cfg, priors, model, calibrator, novelty, risk):
    """Rescore two illustrative test rows with a fully-seeded stream,
    returning their complete decisions incl. explanations."""
    expl = Explainer(model, MODEL_FEATURES, k=5, reason_groups=True,
                     seed=cfg.seed)
    out = []
    candidates = [
        ("normal_first_time",
         test_scores[(test_scores["is_fraud"] == 0)
                     & (test_scores["risk_band"] == "normal")]),
        ("suspicious_first_time",
         test_scores[(test_scores["is_fraud"] == 1)
                     & (test_scores["alert"] == True)]),  # noqa: E712
    ]
    for name, pool in candidates:
        if pool.empty:
            out.append({"name": name, "note": "no representative row found"})
            continue
        row = pool.sort_values("rank_score").iloc[0] \
            if name == "normal_first_time" else \
            pool.sort_values("rank_score", ascending=False).iloc[0]
        tid = row["transaction_id"]
        raw = raw_df[raw_df["transaction_id"] == tid].iloc[0]
        proc = _build_processor(cfg, model, priors, calibrator, novelty, risk,
                                explainer=expl)
        prior = raw_df[raw_df["ts_sec"] < float(raw["ts_sec"])]
        for p in prior.itertuples(index=False):
            proc.engine.commit_row({
                "ts_sec": float(p.ts_sec), "amount": float(p.amount),
                "customer_id": p.customer_id, "merchant_id": p.merchant_id,
                "device_id": p.device_id,
                "transaction_hour": int(p.transaction_hour)})
        txn = {"ts_sec": float(raw["ts_sec"]), "amount": float(raw["amount"]),
               "customer_id": raw["customer_id"],
               "merchant_id": raw["merchant_id"],
               "device_id": raw["device_id"],
               "transaction_hour": int(raw["transaction_hour"]),
               "timestamp": str(raw["timestamp"])}
        out.append({"name": name,
                    "transaction_id": tid,
                    "is_fraud": int(raw["is_fraud"]),
                    "amount": round(float(raw["amount"]), 2),
                    "transaction_hour": int(raw["transaction_hour"]),
                    "new_customer": int(raw.get("new_customer", -1)),
                    "new_device": int(raw.get("new_device", -1)),
                    "decision": proc.score_transaction(txn)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="optional: only process the first N transactions "
                         "(development smoke runs)")
    opts = ap.parse_args(argv)

    cfg = get_settings()
    _seed(cfg)
    print(f"== {cfg.name} ==")
    print(f"seed={cfg.seed}  primary={cfg.models['primary']}")

    # ---------------- 1. load + validate --------------------------------
    print("[1/10] loading + validating raw data ...")
    df, schema = load_dataframe(cfg)
    df = valid_numeric(df)
    if opts.limit:
        df = df.iloc[: opts.limit].reset_index(drop=True)
    write_validation_report(df, cfg.reports_dir())
    print(f"      {len(df):,} rows, {df['is_fraud'].mean()*100:.2f}% fraud")

    # ---------------- 2. splits ----------------------------------------
    print("[2/10] time split + unseen-customer holdout ...")
    train, val, test = time_split(df, cfg)
    train, val, test, heldout = unseen_customer_split(
        train, val, test, df, cfg)
    train_customers = set(train["customer_id"].unique())
    print(f"      train={len(train):,} val={len(val):,} test={len(test):,} "
          f"heldout_customers={len(heldout):,}")

    # ---------------- 3. priors on train --------------------------------
    print("[3/10] fitting segment/global priors on TRAIN only ...")
    priors_engine = FeatureEngine(cfg)
    priors = priors_engine.compute_priors(train)

    # ---------------- 4. full feature replay + parity --------------------
    print("[4/10] leakage-safe feature replay (all folds, one path) ...")
    combined = pd.concat([train, val, test], ignore_index=True)
    combined = combined.sort_values(["ts_sec", "transaction_id"]) \
                       .reset_index(drop=True)
    replay_engine = FeatureEngine(cfg, priors)
    feats_all = replay_engine.replay(combined, with_parity=True)
    parity = _parity_metrics(feats_all)
    print(f"      parity: {parity}")

    train_feats = feats_all[feats_all["transaction_id"].isin(
        set(train["transaction_id"]))].reset_index(drop=True)
    val_feats = feats_all[feats_all["transaction_id"].isin(
        set(val["transaction_id"]))].reset_index(drop=True)
    test_feats = feats_all[feats_all["transaction_id"].isin(
        set(test["transaction_id"]))].reset_index(drop=True)
    for name, f in [("train", train_feats), ("val", val_feats),
                    ("test", test_feats)]:
        f.to_parquet(os.path.join(cfg.processed_dir(),
                                  f"{name}_features.parquet"))
    print(f"      feats train={len(train_feats):,} val={len(val_feats):,} "
          f"test={len(test_feats):,}")

    # ---------------- 5. model comparison --------------------------------
    print("[5/10] training + comparing supervised models ...")
    Xtr, ytr = train_feats[MODEL_FEATURES], train_feats["is_fraud"].to_numpy()
    Xval, yval = val_feats[MODEL_FEATURES], val_feats["is_fraud"].to_numpy()
    results = train_and_compare(Xtr, ytr, Xval, yval, cfg)
    best = results["__best__"]
    model = results[best]["model"]
    print(f"      primary model: {best}  (PR-AUC(val)="
          f"{results[best]['pr_auc_val']:.4f})")
    save_model_bundle(cfg, results, list(results.keys() - {"__best__"}))

    # ---------------- 6. calibration -------------------------------------
    print("[6/10] calibrating fraud probabilities ...")
    raw_val = model.predict_proba(Xval)[:, 1]
    calibrator, cal_report = fit_best(np.asarray(raw_val), yval, cfg)
    print(f"      selected: {cal_report['selected']} "
          f"(brier={cal_report['selected_brier']})")

    # ---------------- 7. novelty model -----------------------------------
    print("[7/10] fitting behavioural novelty + ECDF ...")
    nov_name = cfg.features["novelty_model"]
    novelty = NoveltyModel(nov_name, seed=cfg.seed,
                           noise=cfg.features["novelty_noise"])
    Xtr_l = train_feats.loc[ytr == 0, NOVELTY_FEATURES]
    novelty.fit(Xtr_l, np.zeros(len(Xtr_l)), legit_only=True)
    legit_val = yval == 0
    novelty.fit_ecdf(Xval.loc[legit_val, NOVELTY_FEATURES],
                     np.zeros(int(legit_val.sum())), legit_only=True)
    print(f"      novelty model: {nov_name}  (ECDF ref n={novelty.ref_n})")

    # ---------------- 8. risk thresholds on validation -------------------
    print("[8/10] selecting risk thresholds on validation ...")
    val_p = calibrator.predict(raw_val).astype(float)
    val_nov = novelty.novelty(Xval[NOVELTY_FEATURES]).astype(float)
    thresholds = fit_thresholds(val_p, val_nov, cfg)
    print(f"      p_review={thresholds['p_review']} "
          f"p_high={thresholds['p_high']}")

    # ---------------- 9. streaming replay val + test ---------------------
    print("[9/10] streaming replay (seeded) over val + test ...")
    val_proc = _build_processor(
        cfg, model, priors, calibrator, novelty,
        RiskEngine(), budget=None, explainer=None)
    val_proc.seed_state(train)
    print(f"      scoring validation fold ...")
    val_scores = val_proc.run(val)
    val_scores.to_parquet(os.path.join(cfg.processed_dir(),
                                       "val_scores.parquet"))

    fitted = RiskEngine(thresholds)
    test_proc = _build_processor(
        cfg, model, priors, calibrator, novelty, fitted,
        budget=AlertBudgetController(
            max_per_hour=int(cfg.risk["max_alerts_per_hour"]),
            max_per_day=int(cfg.risk["max_alerts_per_day"]),
            budget_pct=float(cfg.risk["alert_budget_pct"]),
            high_risk_share_cap_pct=float(cfg.risk["high_risk_share_cap_pct"])),
        explainer=None)
    prior_stream = pd.concat([train, val], ignore_index=True)
    test_proc.seed_state(prior_stream)
    print(f"      scoring test fold ...")
    test_scores = test_proc.run(test)

    # ---------------- 10. evaluation + figures ---------------------------
    print("[10/10] evaluation report + figures ...")
    scored = ev.add_cohort_flags(test_scores, train_customers, test_feats)
    scored.to_parquet(os.path.join(cfg.processed_dir(),
                                   "test_scores.parquet"))

    yt = scored["is_fraud"].to_numpy()
    pt = scored["fraud_probability"].to_numpy(dtype=float)
    alert_dec = (scored["budget_status"] == "alert").to_numpy()

    pr_data = ev.pr_curve_data(yt, pt)
    classif = ev.classification_block(yt, pt, alert_dec)
    cohorts = ev.cohort_report(test_scores, train_customers, test_feats)
    alerts = ev.alert_metrics(scored)
    fp_ft = ev.false_positive_first_time(scored)
    cohorts_rows = _cohort_table(
        cohorts, [{"key": k, "rate": v["false_positive_rate"]}
                  for k, v in fp_ft.items()])

    from sklearn.metrics import confusion_matrix as _cm
    confusion = _cm(yt, alert_dec)

    explainer = Explainer(model, MODEL_FEATURES, k=5,
                          reason_groups=bool(cfg.evaluation
                                             ["explanation_reason_groups"]),
                          seed=cfg.seed)
    shap_imp = explainer.global_importance(
        pd.concat([Xval, test_feats[MODEL_FEATURES]], ignore_index=True),
        sample_size=int(cfg.evaluation["shap_sample_size"]))

    figures_written = figs.make_all_figures(
        df, scored, pr_data, confusion, shap_imp,
        _alert_series(scored), cohorts_rows, cfg)

    report = {
        "seed": cfg.seed,
        "primary_model": best,
        "calibration_method": calibrator.used_method,
        "model_comparison": {k: {"pr_auc_val": v["pr_auc_val"],
                                 "roc_auc_val": v["roc_auc_val"]}
                             for k, v in results.items() if k != "__best__"},
        "calibration_report": cal_report,
        "thresholds": thresholds,
        "classification": classif,
        "alerts": alerts,
        "cohorts": {k: {kk: vv for kk, vv in v.items() if kk != "cohort"}
                    for k, v in cohorts.items()},
        "fp_first_time": fp_ft,
        "parity": parity,
        "feature_availability": _feature_availability(test_feats),
        "leakage": {
            "temporal_integrity": "time-based splits, no shuffle",
            "unseen_customer_holdout": f"{len(heldout):,} customers removed "
                                       "from train only",
            "priors_fit_on": "train",
            "features_computed_as_of": "point-in-time, commit after scoring",
            "raw_ids_in_features": False,
            "velocity_8min_match_rate": parity.get("velocity_8min_match_rate"),
            "amount_spike_match_rate": parity.get("amount_spike_match_rate"),
        },
        "figures": figures_written,
    }
    md = ev.write_evaluation_report(report, cfg.reports_dir())

    # ---- persist artifacts ----
    print("      saving artifacts ...")
    os.makedirs(cfg.models_dir(), exist_ok=True)
    joblib.dump(model, os.path.join(cfg.models_dir(), "model_primary.joblib"))
    joblib.dump(calibrator, os.path.join(cfg.models_dir(),
                                         "calibrator.joblib"))
    joblib.dump(novelty, os.path.join(cfg.models_dir(), "novelty.joblib"))
    joblib.dump(priors, os.path.join(cfg.models_dir(), "priors.joblib"))
    feat_names = {"MODEL_FEATURES": MODEL_FEATURES,
                  "NOVELTY_FEATURES": NOVELTY_FEATURES}
    with open(os.path.join(cfg.models_dir(), "feature_names.json"), "w") as fh:
        json.dump(feat_names, fh, indent=2)
    with open(os.path.join(cfg.models_dir(), "thresholds.json"), "w") as fh:
        json.dump(thresholds, fh, indent=2)

    demos = _demo_examples(df, scored, cfg, priors, model,
                           calibrator, novelty, fitted)
    with open(os.path.join(cfg.reports_dir(), "artifacts",
                           "demo_examples.json"), "w") as fh:
        json.dump(demos, fh, indent=2, default=str)

    print()
    print("== DONE ==")
    print(f"  primary model     : {best} (PR-AUC val "
          f"{results[best]['pr_auc_val']:.4f})")
    print(f"  PR-AUC(test)      : {classif['pr_auc']:.4f}")
    print(f"  ROC-AUC(test)     : {classif['roc_auc']:.4f}")
    print(f"  ECE(test)         : {classif['ece']:.5f}")
    print(f"  Brier(test)       : {classif['brier']:.5f}")
    print(f"  alerts / rate     : {alerts['alerts_generated']:,} "
          f"({alerts['alert_rate_pct']:.2f}% of txns)")
    print(f"  alert precision   : {alerts.get('alert_precision')}")
    print(f"  alert recall      : {alerts.get('alert_recall')}")
    print(f"  figures           : {len(figures_written)} written")
    print(f"  evaluation report : reports/02_evaluation_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())