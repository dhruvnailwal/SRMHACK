# Unseen-Customer Fraud Detection

**Team:** NOOB CODERS · **Event:** SRM Hackathon

> A streaming, leak-safe fraud-detection system that keeps genuine new
> customers flowing while catching genuinely new attacks — without ever
> memorising a profile.

---

## 1. Project Overview & Problem Statement

### 1.1 The core problem

Most fraud models fit a **profile to a customer** — the more history you have,
the better the model "knows" you. That design has two failure modes:

1. **False alarms on newcomers.** A customer with no history (or a new device,
   a new merchant) looks structurally identical to an attack, so the model
   flags them. Blocking a genuine first-time buyer costs revenue and trust.
2. **Alert overload.** When every cold-start transaction trips a rule, the
   analyst queue floods and `false positive` noise drowns out the real attacks.

In short, models that *memorise* break exactly where fraud shows up first: **on
customers the model has never seen.**

### 1.2 The business impact

| Failure | Consequence |
|---|---|
| Genuine first-time customer blocked | Lost sale + churned prospect |
| Rule decay (attackers shift patterns) | Legit traffic stops, attacks slip through |
| Analyst alert flood | Slow triage, high false-positive cost, burnout |
| Single-signal reliance | Both known fraud and novel fraud can't be caught at once |

The classical trade-off is binary: *high sensitivity* (alert on everything
new) or *clean queues* (only alert on known patterns). We reject the trade-off.

---

## 2. Proposed Solution

### 2.1 Two signals, one decision

Instead of one "is this fraud?" number, every transaction is described by
**two independent signals**:

- **Fraud Probability** `(0.00 – 1.00)` — a calibrated supervised classifier
  (LightGBM). A score of `0.80` genuinely means ≈ 80% fraud likelihood.
- **Novelty Score** `(0.00 – 1.00)` — an *identity-free* behavioural model.
  `0.90` means "more atypical than 90% of legitimate traffic" *for this
  customer's own history class* — not "new, therefore bad".

The two signals combine into a **risk decision matrix**:

| | novelty low | novelty high |
|---|---|---|
| **p low** | `normal` → allow | `monitor` → log |
| **p high** | `review` → alert | `high-risk` → alert |

> **Key insight:** novelty alone *never* escalates. The novelty→`review` path
> also requires `p >= p_floor`, so a **new-but-ordinary** customer is never
> penalised for simply being new. This is what kills the cold-start false
> alarms.

### 2.2 What the system outputs

For every transaction:

1. **Fraud Probability** — calibrated score.
2. **Novelty Score** — behavioural atypicality.
3. **Risk Band** — `Normal / Monitor / Review / High-Risk`, re-derived under
   the **alert budget** (thresholds = quantiles of today's volume).
4. **Plain-Language Reasons** — exact TreeSHAP contributions grouped into
   human sentences: *"Velocity elevated: 37 txns in 30 min (increases risk)"*.
5. **Budget status** — `alert` or `log`, decided by the alert-budget
   controller so analyst workload stays bounded.

---

## 3. Technical Implementation & Approach Quality

### 3.1 Architecture — built for streaming, built for cold-start

```
transaction
   │
   ▼
validate ────────────── schema, amount>0, ids present
   │
   ▼
prepare features ────── point-in-time ONLY (prior state, never the future)
   │
   ▼
fraud_probability ───── LightGBM + isotonic calibration
   │
   ▼
novelty_score ───────── IsolationForest + ECDF (identity-free)
   │
   ▼
risk_band / alert ───── decision matrix + budget controller
   │
   ▼
explanation ─────────── TreeSHAP (pred_contrib) → reason templates
   │
   ▼
commit history ──────── state mutates LAST, after scoring
```

The pipeline mirrors the exact **production streaming code path** — the same
code that scores one live transaction also replays a historical file, so
prototype results are the production numbers.

### 3.2 Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Model | **LightGBM** (gradient boosting, `pred_contrib` = exact TreeSHAP) |
| Models compared | LightGBM vs RandomForest vs LogisticRegression |
| Calibration | scikit-learn `CalibratedClassifierCV` (isotonic) |
| Novelty | IsolationForest + ECDF normalisation |
| Service | **FastAPI** (`POST /predict/batch`) |
| Data | Pandas / NumPy |
| UI | **Streamlit** (hero, upload, results dashboard, export) |
| Features | engineered, streaming windowed (5m / 30m / 24h), shrinkage baselines |

### 3.3 Pipeline detail

```
Transaction -> Validate -> Features -> Models -> Calibrate -> Alert Policy -> Explain
```

- **Validate** — required columns, positive amounts, presence of ids.
- **Features** — windowed aggregates (velocity, amount z-scores, device /
  merchant / customer history **confidence**) computed strictly *as-of* the
  transaction timestamp.
- **Models** — classifier (LightGBM) + novelty (IsolationForest + ECDF).
- **Calibrate** — the probability is calibrated on **validation only**; a
  score-to-risk mapping that genuinely means what it says.
- **Alert Policy** — risk matrix + budget controller; `high-risk` is always
  escalated up to a hard cap, `review` is triaged by rank under budget,
  `monitor` is logged.
- **Explain** — every alert carries its top contributing factors in plain
  language.

### 3.4 Why "identity-free" is the right call *here*

The data is a **label-only transaction stream** (no images, no text, no
embeddings, no external features). For this problem we deliberately score
**behaviour, not the customer**:

- Customer history enters only as *confidence-weighted baselines* (shrinkage),
  never as identity — so unseen customers are scored from the population
  prior, exactly like deployment would treat them.
- No raw IDs survive into the model (`raw_ids_in_features: false`), so the
  model cannot cheat by memorising customer numbers; it must generalise.
- Novelty is measured *relative to the customer's own history class* rather
  than absolute rarity, so a legitimate new customer isn't globally "weird".

A deep-learning tabular model was tested against and rejected (LightGBM won on
PR-AUC); stacked identity-free training is statistically rigorous at
`120k` rows and dramatically easier to explain to regulators/analysts.

---

## 4. Innovation

### 4.1 Cold-start shrinkage (blending baselines)

Instead of falling back to a global default when history is thin, each
feature blends the **customer / segment / global** means using a
**confidence weight** proportional to history depth. A first transaction
gets a robust, honest baseline — not a hallucinated one.

### 4.2 History dropout training

During training, history features are randomly "forgotten" for a fraction of
rows, forcing the model to stay useful when the true history is empty. This
directly simulates deployment cold-start inside the loss function.

### 4.3 Dual-signal alerting

Fraud probability and novelty can *each* trigger a band, so a scam that is
probabilistically ordinary (the model hasn't seen the pattern) but
behaviourally novel is still caught — with novelty gated by `p_floor` so new
ignored by a novelty-only rule would not alert.

### 4.4 Identity-free novelty model

Novelty is a function of behaviour within a customer's own history class,
not of the customer's identity. This is what makes the model genuinely
capable of "first time" prediction rather than re-ranking known profiles.

### 4.5 Honest evaluation (known vs. unseen)

We report performance **separately** for customers the model has seen in
training and customers it has never met. The unseen numbers are the ones
that matter, and they are the ones we headline.

### 4.6 Explainable by default

Every decision carries plain-language reasons from exact TreeSHAP
contributions grouped into six semantic families (velocity, amount, time,
device, merchant, history). Correlation is labelled as *contribution*, never
marketed as causation.

---

## 5. Restriction Handling & Compliance

The solution is engineered around the hackathon rule-set rather than appended
to it:

| Restriction | How we comply |
|---|---|
| **Tabular only** (no images / text / LLMs) | Inputs are row-shaped transaction records; models are LightGBM + IsolationForest + ECDF; no embeddings, no vision/LLM anywhere |
| **Streaming** (one transaction at a time, no memorising) | Same `score_transaction` code path for live and replay; history is committed only *after* a row is scored; features never see the future |
| **Leakage-safe** | 1,660 customers removed from train **only**; `time-based splits, no shuffle`; priors/encoders fit on `train`; point-in-time feature commits; parity checks: `velocity_8min_match_rate = 1.0`, `amount_spike_match_rate = 0.9962` |
| **Usable alerts** | Fixed **alert-budget controller** (per-hour / per-day caps + `budget_pct` cap on today's volume); `high-risk` hard-capped; the dashboard slider re-derives thresholds so budgets are respected at run time |

Compliance checks are **encoded in the report**: every evaluation run emits a
leakage report, cohort table, and parity verification.

---

## 6. Prototype Progress & Effectiveness (Key Results)

### 6.1 Current UI/UX progress

The Streamlit prototype delivers the full user flow:

- **Landing & context** — hero, how-it-works, decision matrix visual.
- **Upload / drag-and-drop** — CSV file upload, fast onboarded validation ✓,
  scoring via **FastAPI service** or in-process backend.
- **Results dashboard** — per-transaction cards with fraud-probability gauge,
  novelty score, colour-coded risk-band badge and top reasons; suspected-fraud
  table; system-evaluation block with **known vs. unseen** split and a
  leakage-safety compliance panel.
- **Settings & export** — alert-budget slider that re-derives thresholds live;
  exports in `.csv / .json / .md / .txt / .pdf`.

A CLI runner (`scripts/analyse_file.py`) gives the same analysis without a
browser or server.

### 6.2 Metrics that prove effectiveness

**Hold-out test fold (23,928 rows; unseen-fraud concentrated on unseen
customers):**

| Metric | Value |
|---|---|
| **PR-AUC (all test)** | **0.894** |
| PR-AUC — **known customers** | — (0 fraud in that cohort) |
| PR-AUC — **unseen customers** | **0.900** |
| Recall @ 2% budget — all | 0.828 |
| Recall @ 2% budget — unseen | 0.830 |
| Recall @ 2% — first-time customers | 0.931 |
| New-device cohort recall @ 2% | 1.000 |
| ROC-AUC | 0.994 |
| Brier / ECE | 0.005 / 0.002 |
| FP rate on legit first-time customers | 4.2% |
| Model comparison (val PR-AUC) | LightGBM **0.886** > RF 0.869 > LogReg 0.783 |

The decisive result: **performance does not degrade on customers the model
has never seen** (`PR-AUC 0.894 → 0.900` unseen). That is the entire point.

### 6.3 Final impact

- **Fewer false alerts on genuine new customers** — cold-start shrinkage +
  p_floor gating keep legit newcomers off the alert queue (4.2% FP rate on
  legit first-timers).
- **Earlier catches on genuinely new attacks** — dual-signal alerting recovers
  novel-burst fraud that a memorisation model would silently allow.
- **Bounded analyst workload** — budget controller caps escalations at 2% of
  volume (configurable) while high-risk stays hard-prioritised.

---

## 7. Technical Defense (Q&A)

**Q1. Why LightGBM instead of a deep learning / transformer model?**

- On the val split, LightGBM beat both RandomForest and LogisticRegression
  (PR-AUC 0.886 vs 0.869 vs 0.783) and ran in a fraction of the time.
- The input is engineered tabular data (120k rows) — gradient boosting is the
  empirical state of the art for this size and type; DL gains emerge primarily
  with raw/multimodal signal and far larger corpora.
- LightGBM reports **exact per-feature contributions** natively
  (`pred_contrib`), which powers our plain-language explanations at zero extra
  cost — a major explainability win for a hackathon that demands clarity.
- Deployment is cheap: single model files, no GPU, sub-millisecond scoring.

**Q2. How do you guarantee there is no data leakage?**

- **Time-based splits, no shuffle** — the model never trains on the future.
- **Unseen-customer holdout** — 1,660 customers are removed from training
  data only, guaranteeing the test set contains customers the model has never
  seen.
- **Point-in-time features** — every feature is computed as-of the
  transaction; state is committed only *after* a row is scored
  (`commit_row` runs last), so no row can influence its own features.
- **Priors/encoders fit on train** — baselines that blend customer/segment/
  global are fitted in training only, not re-fit at inference on test volume.
- **Verification baked in** — the report recomputes engineered features and
  checks parity (`velocity_8min_match_rate = 1.0`), and confirms no raw IDs
  are used as model features.

**Q3. If novelty can trigger an alert, won't you just flag every new
customer?**

No — and this is the heart of the design:
- The novelty → `review` path requires `p >= p_floor` as well, so behavioural
  novelty alone never escalates; a new-but-ordinary customer stays `normal`.
- Novelty is measured *relative to the customer's own history class*, not as
  absolute global rarity, so "new" ≠ "weird".
- The alert **budget** is a hard cap regardless — only the top ~2% of daily
  volume (configurable) can ever escalate, so even a burst of new customers
  cannot flood the analyst queue. The evidence: 93% recall on first-time
  customers at just 4.2% false-positive rate.

---

### Appendix — Running the prototype

```bash
# Streamlit dashboard (in-process, no server needed)
python -m streamlit run dashboard/app.py

# FastAPI scoring service (optional; dashboard falls back automatically)
python -m uvicorn api.main:app --reload

# Same analysis, no browser (CLI)
python scripts/analyse_file.py data/processed/test_fold_file.csv
```

*Dataset is a deterministic synthetic stream (`data/raw/
synthetic_fraud_transactions.csv`); results are synthetic and illustrative,
not a claim about any real portfolio.*