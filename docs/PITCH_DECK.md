# Unseen-Customer Fraud Detection — 7-Slide Pitch Deck

**Team:** NOOB CODERS · **Event:** SRM Hackathon (Engineers' Day — ML Challenge)

*Slide copy restricted to features actually built in this repo.*

---

## Slide 1 — Problem

- Hackathon brief (SRM_PS.md): fraud detection must work even when the customer is **never seen before** — "new" must not be treated as "fraud."
- Two distinct questions, not one: *"Is this fraud-like?"* and *"Is this behaviour we've watched before?"*
- Real constraints: streaming, one-pass decisions, a hard alert budget, and plain-language explanations on every call.
- Headline metric, not accuracy: precision/recall at a fixed alert budget — `known` vs `unseen` cohorts reported separately.

---

## Slide 2 — Solution

- Streaming pipeline: `validate → point-in-time features → fraud_probability → novelty_score → risk_band/alert → explanation → commit`.
- Two **independent signals**: a calibrated fraud probability (LightGBM + isotonic calibration) and an **identity-free novelty score** (IsolationForest + ECDF).
- A 2×2 risk matrix combines the signals; a budget controller **hard-caps** high-risk and review volume.
- Novelty alone can never escalate — `p ≥ p_floor` is required, keeping legitimate first-time customers out of the alert queue.
- Every transaction carries grouped, plain-language **contributing factors** (contribution, never causation).

---

## Slide 3 — Technical Implementation

- Core package `src/fraud_detection/`: `config / data_gen / features / models / policy / explain / pipeline / train / export`.
- `HistoryStore` — streaming per-customer, per-segment and global state; `extract_features` is a pure, point-in-time-only function; `commit_row` always runs last.
- One shared object, `StreamingScorer` — the identical code path drives the **FastAPI service** (`/predict`, `/predict/batch`), the **Streamlit dashboard**, and the **CLI scorer**.
- 9-stage training script: unseen-customer holdout → time split → priors on TRAIN only → history-dropout features → train → calibrate on validation → fit novelty → persist store → replay + evaluate.
- Verification: `reports/leakage_report.json` (zero leakage, parity re-check) and an 18-check QA suite (18/18 PASS).

---

## Slide 4 — Innovation

- **Unseen-customer fairness built in:** customers held out from TRAIN; legacy holdout + time split eliminate leakage by design.
- **Novelty ≠ fraud enforced in the policy layer**, not just the data: the matrix requires both signals before escalation.
- **TreeSHAP contributions grouped into six human families** (velocity, amount, time, device, merchant, history) — analysts read "these factors moved the score", not a verdict.
- Explanations use exact TreeSHAP contributions (LightGBM native) with a perturbation fallback.
- Identity-free design: raw IDs are verified absent from model features (QA check I).

---

## Slide 5 — Restriction Handling

- Environment had **no internet**: LightGBM/FastAPI could not be installed at build time.
  - `ClassifierModel` auto-detects LightGBM at runtime and falls back to `HistGradientBoostingClassifier` — no code changes needed.
  - API and dashboard were syntax-checked and covered by the FastAPI `TestClient` / Streamlit `AppTest` QA suite (18/18), not just eyeballed.
- Per-transaction explanations use a perturbation-based contribution estimate when the LightGBM booster (native `pred_contrib`) is unavailable.
- **Synthetic data is explicitly labelled illustrative**, not a benchmark claim — the pipeline is dataset-agnostic (schema adapter for real IEEE-CIS data exists and was run).
- Honest limitations are documented in the README, master submission, and delivery docs.

---

## Slide 6 — Prototype Progress

- End-to-end working prototype: training, CLI scoring, REST API, and Streamlit dashboard (dark/navy themed) all live.
- CLI: `analyse_file.py` scores a CSV with no server (2000-row run: 52 alerts, 2.6% budget, 46.5% unseen customers).
- Exports in 5 formats — CSV, JSON, Markdown, TXT, **PDF** — each carrying per-transaction amounts and readings (fraud probability, novelty, risk band, rank), fixed after the "missing values" review.
- Delivery docs written from a beginner's perspective: `MASTER_SUBMISSION.md`, `STYLING_GUIDE.md`, `DATA_JOURNEY.md`, QA report, IEEE-CIS run report.
- Compatibility proven on **published IEEE-CIS data** (300k rows, LightGBM): full pipeline ran all 10 stages; results preserved under `reports/ieee_cis_run/`.

---

## Slide 7 — Key Results

- **Synthetic stream (prototype's own data):** PR-AUC (unseen) 0.826, ROC-AUC 0.997, recall-at-2%-budget (unseen) 83.8%, **false-positive rate on legitimate first-time customers: 0.0%**.
  - Baseline re-run confirms stability: PR-AUC 0.894, ROC-AUC 0.994.
- **Real data (IEEE-CIS, honest numbers):** ROC-AUC 0.716, calibration error 0.004 (ECE), alerts kept at 1.68% of volume — demonstrates the streaming/calibration engine on a hard real dataset.
- **Budget control proven:** alert volume hard-capped at the configured budget in QA (60-txn burst → exactly 3 alerts).
- **Leakage cleared:** `leakage_report.json` + parity re-check verify time- and customer-based integrity.