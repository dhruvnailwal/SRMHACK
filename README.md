# Unseen-Customer Fraud Detection Using Fraud Probability and Behavioral Novelty

A complete fraud-detection pipeline that combines a **calibrated supervised
classifier** (fraud probability) with an **unsupervised behavioural novelty
model** (novelty score) into a single risk decision. The design centres the
project's core problem: **new / first-seen customers** have almost no history,
so a pure classifier is systematically blind or over-alerting for them.

The system is leakage-safe by construction (point-in-time features, train-only
priors, chronologically ordered splits, a disjoint unseen-customer holdout),
and every headline number in the reports refers to the **test fold** replayed
through the *exact production streaming path* with a working alert budget.

---

## Quick start

```bash
# 1. environment (Python 3.10+)
pip install -r requirements.txt

# 2. generate the synthetic transaction stream (schema matches data_loader)
python scripts/generate_dataset.py
#    -> data/raw/synthetic_fraud_transactions.csv

# 3. full pipeline: features -> models -> calibration -> novelty -> risk
#    -> streaming replay -> evaluation report + 15 figures
python run_pipeline.py

# 4. notebooks (Jupyter)
python tools/build_notebooks.py
jupyter lab notebooks/

# 5. dashboard
streamlit run dashboard/app.py

# 6. API service
uvicorn api.main:app --reload
#    docs: http://localhost:8000/docs
```

`python run_pipeline.py --limit 20000` runs a fast development smoke run.

---

## Project structure

```
config.yaml                     pipeline configuration (paths, splits, budgets)
requirements.txt
run_pipeline.py                 end-to-end orchestrator
src/
  config.py                     Settings from config.yaml (sane defaults)
  data_loader.py                tolerant schema mapping + chronological load
  data_validation.py            data-quality report (JSON + Markdown)
  preprocessing.py              time split + unseen-customer holdout
  feature_engineering.py        FeatureEngine - one leakage-safe code path
  supervised_model.py           LR / RandomForest / LightGBM comparison
  novelty_model.py              IsolationForest (+ LOF/OCSVM/autoencoder) + ECDF
  calibration.py                Platt vs Isotonic by Brier
  risk_engine.py                probability x novelty decision matrix + thresholds
  explainability.py             exact TreeSHAP + grouped reason explanations
  streaming_processor.py        streaming scoring + alert-budget controller
  evaluation.py                 metrics, cohorts, reliability, report writer
  figures.py                    the 15 evaluation figures
scripts/generate_dataset.py     synthetic data generator (reproducible, seed 42)
notebooks/01..06                guided walkthrough notebooks
dashboard/app.py                Streamlit dashboard (metrics/figures/live-scoring)
api/main.py                     FastAPI service (POST /predict)
models/                         trained artifacts (joblib + json)
data/raw synthetic_fraud_transactions.csv
data/processed  *_features.parquet, *_scores.parquet
reports/figures 15 PNG figures, reports/artifacts
```

---

## How the two signals work together

|              | novelty low       | novelty high      |
|--------------|-------------------|-------------------|
| **p low**    | `normal` → allow  | `monitor` → log   |
| **p high**   | `review` → alert  | `high-risk` → alert |

* `fraud_probability` – calibrated classifier output; well-calibrated, so a
  value of 0.8 genuinely means ~80% fraud probability (ECE < 1% on test).
* `novelty_score` – ECDF rank of the anomaly model vs *legitimate* traffic.
  0.9 = more atypical than 90% of legit transactions.
* A high novelty score **never** alerts by itself (the novelty→review channel
  requires `p >= p_floor`), protecting new-but-normal customers.
* Alerts run through an `AlertBudgetController`: budget 2% of volume, high-risk
  priority, caps per hour/day. All thresholds are selected on **validation**.

---

## Reported results (reproducible with seed 42)

Produced by `python run_pipeline.py` (see `reports/02_evaluation_report.md`):

* model comparison (val): LightGBM wins by PR-AUC
* test: PR-AUC, ROC-AUC, ECE, Brier, alert precision / recall @ 2% budget
* cohorts: `all`, `known_customers`, `unseen_customers`, `first_time_*`
* novelty AUROC on the cold cohort, false-novelty rate, false-positive rate on
  legitimate first-time transactions
* data-integrity: parity of recomputed `velocity_8min` / `amount_spike` vs the
  provided columns, feature-availability (cold-start rate), leakage checks

## API example

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "timestamp": "2025-12-31 16:28:36+00:00",
  "amount": 150.0,
  "customer_id": "C-0001",
  "merchant_id": "M-0001",
  "device_id": "D-0001"
}'
```

Returns the risk band, probability, novelty, alert decision, and per-transaction
explanations (`top_contributing_features`).

---

## Delivery artifacts

| artifact | purpose |
|---|---|
| `docs/DATA_JOURNEY.md` | frontend ↔ backend bridge: the full upload → stream → two-signals → budget lifecycle plus an output glossary (fraud probability, novelty, risk bands, top reasons) for non-technical stakeholders |
| `docs/MASTER_SUBMISSION.md` | one beginner-friendly file with the full 7-section narrative (big picture, UI/UX, architecture, innovations, restriction compliance, measured results, judge Q&A) — zero-prior-knowledge readable |
| `docs/STYLING_GUIDE.md` | UI/UX design language (dark/navy theme, Tailwind v4 utility spec, component breakdown) mapped 1:1 onto the shipped Streamlit dashboard |
| `reports/qa_report.md` | end-to-end QA suite results (frontend upload flow, FastAPI endpoints, leakage + alert-budget constraints) |
| `reports/ieee_cis_run/` | the *same* unmodified pipeline run against the real IEEE-CIS benchmark through a schema adapter (`scripts/adapt_ieee_cis.py`, `config.ieee_cis.yaml`) — architecture transfer proof |

Regenerate the QA suite anytime with `python scripts/qa_suite.py` (18 checks);
re-run the IEEE-CIS feasibility run with
`python scripts/adapt_ieee_cis.py` then
`python run_pipeline.py --config config.ieee_cis.yaml --limit 300000`.

## Reproducibility

* Synthetic data: `scripts/generate_dataset.py` – fixed seed, documented schema.
* Splits: chronological by position (no shuffle); a fixed-seed disjoint set of
  customers is removed from training only (the *unseen-customer* cohort).
* Priors/encoders/bins are fit on **training only** and frozen.
* Feature engineering has one code path (`prepare_row`/`commit_row` in the
  engine) shared by batch replay and streaming.

## Documented limitations

* The dataset is **synthetic** – the absolute numbers are illustrative; the
  *method* (leakage-free features + calibrated probability + protected novelty
  channel + budgeted alerting) is the contribution.
* Novelty is unsupervised and environment-dependent; in production it must be
  re-fitted periodically on live legitimate traffic.
* A single API transaction with no surrounding history is scored at cold start
  (segment/global priors) – history is never invented. For real deployments,
  seed the service state from the persisted stream (see `api/main.py`).
* Isotonic calibration can saturate probability at the extremes on strongly
  separated data; thresholds are therefore selected from empirical quantiles on
  validation, which keeps alert volume inside the configured budget.