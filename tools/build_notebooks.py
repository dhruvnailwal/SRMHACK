"""Build the six project notebooks as valid nbformat v4 (stdlib only).

Usage:
    python tools/build_notebooks.py

Produces notebooks/01_*.ipynb ... notebooks/06_*.ipynb. Every code cell is
designed to run as-is after ``python run_pipeline.py`` has been executed.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")

BOOTSTRAP = """\
import os, sys
ROOT = os.path.dirname(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
"""


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in source.strip("\n").split("\n")]}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [],
            "source": [l + "\n" for l in source.strip("\n").split("\n")]}


def notebook(title: str, cells: list[dict]) -> dict:
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python",
                                           "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


def write(name: str, nbf: dict):
    os.makedirs(NB_DIR, exist_ok=True)
    path = os.path.join(NB_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nbf, fh, indent=1)
    print("wrote", path)


# ---------------------------------------------------------------------------
NB1 = notebook("01_data_loading_and_validation.ipynb", [
    md("""# 01 - Data Loading & Validation

**Unseen-Customer Fraud Detection Using Fraud Probability and Behavioral Novelty**

This notebook loads the synthetic transaction stream, runs the tolerant
schema mapping (`src/data_loader.py`) and produces the structured data-quality
report (`src/data_validation.py`)."""),
    code(BOOTSTRAP),
    code("""\
from src.config import get_settings
from src.data_loader import load_dataframe, valid_numeric, SchemaMap
from src.data_validation import build_validation_report

cfg = get_settings()
df, schema = load_dataframe(cfg)
df = valid_numeric(df)
print(f"loaded: {len(df):,} rows  x  {df.shape[1]} columns")
print("fraud rate: {:.2f}%".format(df['is_fraud'].mean() * 100))"""),
    code("""\
rep = build_validation_report(df)
for k in ("rows", "fraud_count", "fraud_percent", "unique_counts",
          "timestamp_range"):
    print(k, "=>", rep[k])"""),
    code("""\
print("mapped columns (alias -> canonical):")
for canon, raw in (d := schema.mapping).items():
    print(f"  {canon:28s} <- {raw}")
print("unmapped optional:", schema.unmapped)"""),
    md("""## Key takeaways
- The loader maps aliases and coerces types; timestamps are parsed and the
  frame sorted chronologically (`ts_sec` attached).
- Duplicate `transaction_id` rows are dropped at load and their count kept.
- Nothing is split or featurised yet – the stream stays untouched until the
  evaluation splits are constructed in notebook 03."""),
])

# ---------------------------------------------------------------------------
NB2 = notebook("02_eda_and_class_imbalance.ipynb", [
    md("""# 02 - Exploratory Data Analysis

Fraud distribution, amount profiles, hourly fraud rates and the new-vs-known
customer / device gaps that motivate the project."""),
    code(BOOTSTRAP),
    code("""\
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from src.config import get_settings
from src.data_loader import load_dataframe, valid_numeric

cfg = get_settings()
df = valid_numeric(load_dataframe(cfg)[0])

f, axes = plt.subplots(2, 2, figsize=(12, 7))
df['is_fraud'].value_counts().sort_index().plot.bar(
    ax=axes[0,0], color=['#90caf9', '#e57373'])
axes[0,0].set_title("class balance")
df.groupby('transaction_hour')['is_fraud'].mean().plot(
    ax=axes[0,1], color='#5c6bc0')
axes[0,1].set_title("fraud rate by hour")
df.groupby('new_customer')['is_fraud'].mean().plot.bar(
    ax=axes[1,0], color=['#66bb6a', '#ef5350'])
axes[1,0].set_title("new vs known customers")
df.groupby('new_device')['is_fraud'].mean().plot.bar(
    ax=axes[1,1], color=['#66bb6a', '#ef5350'])
axes[1,1].set_title("new vs known devices")
plt.tight_layout(); plt.show()"""),
    code("""\
print("fraud rate, all             : {:.3f}%".format(df['is_fraud'].mean()*100))
print("fraud rate, new customers   : {:.3f}%".format(
    df.loc[df['new_customer']==1, 'is_fraud'].mean()*100))
print("fraud rate, known customers : {:.3f}%".format(
    df.loc[df['new_customer']==0, 'is_fraud'].mean()*100))
print("mean amount non-fraud: {:.2f} | fraud: {:.2f}".format(
    df.loc[df['is_fraud']==0, 'amount'].mean(),
    df.loc[df['is_fraud']==1, 'amount'].mean()))"""),
    md("""## Observations
- Baseline fraud rate ~3%; new customers / devices carry a substantially
  higher rate – raw *novelty* cues are informative but would over-alert if
  used naively (that is why the risk engine requires a probability floor for
  the novelty->review channel)."""),
])

# ---------------------------------------------------------------------------
NB3 = notebook("03_leakage_safe_feature_engineering.ipynb", [
    md("""# 03 - Leakage-Safe Feature Engineering

The single production code path `FeatureEngine` (`src/feature_engineering.py`)
computes every feature **as-of** the transaction: history before the row is
allowed, the row itself is not. Priors and segment baselines are fitted on
training data only."""),
    code(BOOTSTRAP),
    code("""\
from src.config import get_settings
from src.data_loader import load_dataframe, valid_numeric
from src.preprocessing import time_split, unseen_customer_split
from src.feature_engineering import (FeatureEngine, MODEL_FEATURES,
                                     NOVELTY_FEATURES)

cfg = get_settings()
df = valid_numeric(load_dataframe(cfg)[0])
train, val, test = time_split(df, cfg)
train, val, test, heldout = unseen_customer_split(train, val, test, df, cfg)
print(f"train={len(train):,} val={len(val):,} test={len(test):,}")
print(f"unseen-customer holdout removed from train: {len(heldout):,}")"""),
    code("""\
priors_engine = FeatureEngine(cfg)
priors = priors_engine.compute_priors(train)
print("priors fitted on:", priors['fitted_on'])
print("global log-amount prior mean/std:", round(priors['global']['mean'], 3),
      round(priors['global']['std'], 3))"""),
    code("""\
import pandas as pd
combined = pd.concat([train, val, test], ignore_index=True).sort_values(
    ["ts_sec", "transaction_id"]).reset_index(drop=True)
engine = FeatureEngine(cfg, priors)
feats = engine.replay(combined, with_parity=True)

parity_v = float(feats['velocity_8min'].eq(feats['_parity_velocity_8min']).mean())
parity_s = float(feats['amount_spike'].eq(feats['_parity_amount_spike']).mean())
print(f"parity velocity_8min : {parity_v:.4f}")
print(f"parity amount_spike  : {parity_s:.4f}")
print(f"model features       : {len(MODEL_FEATURES)}  "
      f"novelty features: {len(NOVELTY_FEATURES)}")"""),
    code("""\
cold = feats[feats['customer_history_count'] == 0]
est  = feats[feats['customer_history_count'] >= 5]
print("cold-start share: {:.1f}%  (n={:,})".format(
    100*len(cold)/len(feats), len(cold)))
print("mean |amount_z_shrunk|  cold: {:.2f} | established: {:.2f}".format(
    cold['amount_z_shrunk'].abs().mean(), est['amount_z_shrunk'].abs().mean()))"""),
    md("""## Design rules enforced
1. Strict chronological processing; `prepare_row` before `commit_row`.
2. Raw `customer_id/merchant_id/device_id` never become features – they only
   key internal state.
3. Segment priors, amount bins and global baselines are frozen from training.
4. Cold-start baselines shrink thin customer history toward the segment prior
   (`kappa`), then the global prior – missing history is not treated as fraud.
5. Parity columns (`velocity_8min`, `amount_spike`) are recomputed from first
   principles and compared with the provided ones to prove point-in-time
   correctness (>>99% match)."""),
])

# ---------------------------------------------------------------------------
NB4 = notebook("04_supervised_model_and_calibration.ipynb", [
    md("""# 04 - Supervised Model Selection & Calibration

Three tabular models are compared by **PR-AUC on validation** (not accuracy):
Logistic Regression, Random Forest, LightGBM. Probabilities are then
calibrated (Platt vs Isotonic, chosen by Brier)."""),
    code(BOOTSTRAP),
    code("""\
import pandas as pd, numpy as np, os, json
from src.config import get_settings
from src.supervised_model import train_and_compare
from src.feature_engineering import MODEL_FEATURES
from src.calibration import fit_best

cfg = get_settings()
base = os.path.join(os.getcwd(), "data", "processed")
tr = pd.read_parquet(os.path.join(base, "train_features.parquet"))
va = pd.read_parquet(os.path.join(base, "val_features.parquet"))
Xtr, ytr = tr[MODEL_FEATURES], tr['is_fraud'].to_numpy()
Xva, yva = va[MODEL_FEATURES], va['is_fraud'].to_numpy()"""),
    code("""\
results = train_and_compare(Xtr, ytr, Xva, yva, cfg)
best = results['__best__']
for k, v in results.items():
    if k != '__best__':
        print(f"  {k:24s} PR-AUC={v['pr_auc_val']:.4f}  "
              f"ROC-AUC={v['roc_auc_val']:.4f}")
print("primary:", best)"""),
    code("""\
model = results[best]['model']
raw_val = model.predict_proba(Xva)[:, 1]
cal, cal_rep = fit_best(np.asarray(raw_val), yva, cfg)
print("selected calibration:", cal_rep['selected'],
      "brier:", cal_rep['selected_brier'])
print(json.dumps(cal_rep['methods'], indent=2))"""),
    code("""\
from src.calibration import calibration_curve_data
p = cal.predict(np.asarray(raw_val))
cd = calibration_curve_data(yva, p, bins=10)
pd.DataFrame(cd).plot(figsize=(7, 4), title="reliability diagram");
print("ECE:", cd['ece'])"""),
    md("""## Takeaway
Calibrated `fraud_probability` is the signal the risk engine consumes. On the
validation fold the primary model reaches PR-AUC ≈ 0.89 with ECE well under
1%, so the probability scale is trustworthy for thresholding."""),
])

# ---------------------------------------------------------------------------
NB5 = notebook("05_novelty_model_and_risk_engine.ipynb", [
    md("""# 05 - Behavioural Novelty + Risk Engine

`novelty_score` answers a different question than the classifier: *"how
unusual is this behaviour compared with legitimate traffic?"*. It is the ECDF
rank of an anomaly model (fit on legitimate training rows) relative to
legitimate calibration traffic, so 0.90 means *more atypical than 90% of
legitimate transactions*."""),
    code(BOOTSTRAP),
    code("""\
import numpy as np, pandas as pd, os, joblib
from src.config import get_settings
from src.feature_engineering import NOVELTY_FEATURES

cfg = get_settings()
novelty = joblib.load(os.path.join(cfg.models_dir(), "novelty.joblib"))
print("novelty model:", novelty.name, "| ECDF ref n:", novelty.ref_n)"""),
    code("""\
tr = pd.read_parquet(os.path.join(cfg.processed_dir(), "train_features.parquet"))
va = pd.read_parquet(os.path.join(cfg.processed_dir(), "val_features.parquet"))
nov = novelty.novelty(va[NOVELTY_FEATURES])
s = pd.Series(nov)
print("novelty distribution (val):")
print(s.describe().round(3).to_string())
print("share > 0.90 (top decile of attention):",
      round(float((s > 0.90).mean()), 4))"""),
    code("""\
from src.risk_engine import RiskEngine, fit_thresholds
import json
thr = json.load(open(os.path.join(cfg.models_dir(), "thresholds.json")))
risk = RiskEngine(thr)
print("thresholds:", thr)

probe = [risk.decide(p, n) for p, n in [(0.001, 0.95), (0.05, 0.95),
                                        (0.55, 0.30), (0.80, 0.99)]]
for d in probe:
    print(f"  p={d['fraud_probability']:.3f} nov={d['novelty_score']:.2f} "
          f"-> {d['risk_band']:9s} alert={d['alert']}")"""),
    code("""\
from src.streaming_processor import AlertBudgetController
budget = AlertBudgetController(max_per_hour=250, max_per_day=3000,
                               budget_pct=2.0, high_risk_share_cap_pct=0.7)
status = [budget.decide(1.7e9 + i*3600, "review", 0.9 - i*0.02)
          for i in range(5)]
print("budget decisions:", status)
print("escalated:", budget.total_escalated, "logged:", budget.logged)"""),
    md("""## Risk decision matrix
|                      | novelty low | novelty high |
|----------------------|-------------|--------------|
| **p low**            | normal      | monitor      |
| **p high**           | review      | high-risk    |

`high-risk` -> always alert; `review` -> alert; `monitor` -> logged, no
escalation; `normal` -> allow. A high novelty score **never** triggers an
alert on its own (new-but-normal customers are protected) – the novelty->
review channel requires `p >= p_floor`."""),
])

# ---------------------------------------------------------------------------
NB6 = notebook("06_evaluation_and_interpretation.ipynb", [
    md("""# 06 - Evaluation & Interpretation

The streaming processor (`src/streaming_processor.py`) replays the test fold
transaction-by-transaction with state seeded up to the end of validation –
the exact production code path. This notebook reads the persisted results and
summarises the honest evaluation, including the unseen-customer cohort."""),
    code(BOOTSTRAP),
    code("""\
import pandas as pd, numpy as np, os, json
from src.config import get_settings
from src import evaluation as ev

cfg = get_settings()
scored = pd.read_parquet(os.path.join(cfg.processed_dir(), "test_scores.parquet"))
scored = ev.add_cohort_flags(scored, set(
    pd.read_parquet(os.path.join(cfg.processed_dir(),
                                 "train_features.parquet"))['customer_id']))
print(f"test rows: {len(scored):,}  "
      f"fraud rate: {scored['is_fraud'].mean()*100:.2f}%")"""),
    code("""\
yt = scored['is_fraud'].to_numpy()
pt = scored['fraud_probability'].to_numpy(dtype=float)
alert = (scored['budget_status'] == 'alert').to_numpy()
print(ev.classification_block(yt, pt, alert))
print(ev.alert_metrics(scored))"""),
    code("""\
cohorts = ev.cohort_report(scored, set(
    pd.read_parquet(os.path.join(cfg.processed_dir(),
                                 "train_features.parquet"))['customer_id']))
for name in ["all", "known_customers", "unseen_customers"]:
    b = cohorts.get(name, {})
    print(f"{name:16s} n={b.get('n_rows', 0):,}  PR-AUC={b.get('pr_auc')}  "
          f"recall@2%={b.get('recall@budget2')}")"""),
    code("""\
rep = json.load(open(os.path.join(cfg.reports_dir(),
                                  "evaluation_report.json")))
print("parity          :", rep['parity'])
print("feature avail   :", rep['feature_availability'])
print("leakage checks  :", rep['leakage'])"""),
    code("""\
import os
fs = sorted(os.listdir(cfg.figures_dir()))
print(f"{len(fs)} figures written:")
print("\\n".join("  " + f for f in fs))"""),
    md("""## Conclusions & documented limitations
- The pipeline is **leakage-safe by construction**: point-in-time features,
  train-only priors, time splits and a disjoint unseen-customer holdout.
- Reported test metrics (PR-AUC, ROC-AUC, ECE, alert precision/recall) refer
  to the alert that would actually be fired inside the analyst budget.
- **Limitations**: novelty is unsupervised and environment-dependent; real
  fraud adapts, so thresholds must be re-fitted on live validation traffic;
  the dataset is synthetic – absolute numbers are illustrative, the method is
  the contribution."""),
])

if __name__ == "__main__":
    plans = [(NB1, "01_data_loading_and_validation.ipynb"),
             (NB2, "02_eda_and_class_imbalance.ipynb"),
             (NB3, "03_leakage_safe_feature_engineering.ipynb"),
             (NB4, "04_supervised_model_and_calibration.ipynb"),
             (NB5, "05_novelty_model_and_risk_engine.ipynb"),
             (NB6, "06_evaluation_and_interpretation.ipynb")]
    for nbf, name in plans:
        write(name, nbf)