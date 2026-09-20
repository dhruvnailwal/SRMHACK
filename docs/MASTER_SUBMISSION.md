# Unseen-Customer Fraud Detection
### Hackathon submission — Team *NOOB CODERS* (SRM Hackathon, Engineers' Day — ML Challenge)

One interactive, beginner-friendly file that explains everything: what we
built, how it behaves, how it works under the hood, why it is special, how it
respects the competition's restrictions, the numbers it achieves, and how we
would defend every choice in front of the judges.

> **Reading level.** This document assumes **zero** prior knowledge. If an
> idea has a fancy name, we give a plain-language analogy first and only then
> the official term.

---

## Table of contents
1. [The big picture — what did we actually build?](#1-the-big-picture)
2. [How the app looks and feels (UI/UX)](#2-how-the-app-looks-and-feels)
3. [Under the hood — how it works technically](#3-under-the-hood)
4. [What makes it special (innovations)](#4-innovations)
5. [How we respect the restrictions](#5-restrictions-and-compliance)
6. [Results — measured, not promised](#6-results)
7. [Judge Q&A — the defence](#7-judge-qa-defence)

---

## 1. The big picture

**One sentence:** we built a fraud-detection system that scores every bank
transaction the moment it happens, stamps it with a fraud likelihood and an
"is this normal?" score, and raises an alert only when both signals agree —
so it works even on customers the model has *never seen before*.

### The problem with classic fraud detection

Imagine a bank's security guard who has memorised the face of every loyal
customer. He recognises them instantly by sight. Now a stranger walks in. The
guard has no memory of that face — so how does he decide if the stranger is a
criminal or just a new customer?

That is exactly the situation fraud systems face. Traditional approaches
perform brilliantly on people they have history for ("this card always shops
in Chennai at noon, so this 3 AM purchase is strange") but they **freeze up
on brand-new identities** — and criminals constantly rotate cards, devices,
and merchants specifically to exploit that.

**Our insight:** *the very transactions that matter most are the ones with
no history to memorise.* So instead of betting everything on "has the model
seen this person before?", we give the guard a second pair of eyes that does
not need a memory at all.

### Two signals, one decision (the core idea)

Every transaction is checked by **two independent "sensors"**:

1. **Fraud probability (the supervised expert).**
   A machine-learning model trained on ~120,000 labelled transactions (fraud
   / not fraud). It reads *point-in-time* behaviour features — how fast the
   money left, how unusual the amount is, whether the device/merchant have
   ever been seen, whether the time-of-day is weird. It outputs a number like
   `0.87`, meaning "calibrated 87% chance this is fraud."
2. **Novelty score (the unsupervised newcomer).**
   A completely separate model that never saw any labels. It measures: *"how
   unusual is this transaction compared to ordinary legitimate traffic?"* A
   score of `0.93` means "more atypical than 93% of normal transactions."
   Crucially, it is **identity-free** — it works on a first-ever customer
   exactly as well as on an old one.

Then a tiny **risk matrix** combines them:

|               | Novelty low          | Novelty high          |
|---------------|----------------------|-----------------------|
| **p low**     | `normal` → allow     | `monitor` → log only  |
| **p high**    | `review` → alert     | `high-risk` → alert   |

> The crucial carve-out: **novelty alone never raises an alert.** A
> legitimate first-time customer is unusual by definition — treating "new"
> as "fraud" would drown analysts in false alarms. Novelty can only push a
> transaction toward an alert when *fraud probability is also elevated*.

### The alert budget

Analysts are human; they can only review so many cases per hour. So the
system does not just fire alerts — it **buys a budget**. You set "alert on at
most 2% of transactions," and the system re-derives its thresholds so that
exactly that share escalates, prioritised by risk. Everything else is logged
and monitored.

## 2. How the app looks and feels

The dashboard is a **single scrolling page with three sections** — the
classic "explain → act → evaluate" flow. No login, no configuration rabbit
hole; an analyst opens it and is productive in seconds.

### Section 1 — Overview (the pitch, shown before any tooling)

A hero banner explains the problem in plain language: *memorisation vs
generalisation*, why criminals rotate identities, and how two independent
signals (fraud probability + novelty) make one decision. A colour-coded
2×2 matrix visual sits alongside it, and four short "how it works" steps
(Load → Score → Decide → Explain) prime the reader.

### Section 2 — Upload & Analyse (the action zone)

- Choose your scoring backend: **in-process** (local models) or **via the
  FastAPI service**.
- Toggle **per-row explanations** on/off (slower, richer).
- **Drop a CSV** — or click "Analyse sample test file" for the bundled
  held-out test fold.
- Press **Analyse**. Every row is replayed *chronologically* through the
  exact same streaming pipeline production uses.

After analysis the page shows:
- **File metrics & model evaluation** — PR-AUC, recall @ 2% budget, alerts
  fired, alert rate.
- **Per-transaction cards** — each alert expands to show a fraud-probability
  bar, a novelty-score bar, a colour-coded **risk badge** (see below), and a
  list of plain-English reasons for the score.
- **Suspected-fraud list** — the alerted transactions sorted by risk
  priority.
- **Key insights** — auto-generated one-liners (how many txns are from
  never-seen customers, which decision drivers dominate, PR-AUC summary).
- **Settings & export** — CSV / JSON / Markdown / TXT / PDF download.

There is an **"Alert budget" slider** (0.5% → 10%). Moving it instantly
re-derives the Review / High-Risk probability thresholds as quantiles, so
exactly that share of transactions alert — no re-training, always honest.

### Section 3 — System evaluation (dynamic, not a brochure)

This section is the antidote to "great-looking but meaningless dashboards."
Every number here is computed **from the file you just uploaded** —
PR-AUC, recall @ 2%, alert precision, and a **known-vs-unseen customer
split** bar chart (the headline generalisation story). A **leakage-safety
compliance panel** lists the static pipeline guards (time-based splits,
unseen-customer holdout, priors fit on train, point-in-time features, no raw
IDs as features, feature parity).

### The risk colour language

| band | colour | meaning |
|---|---|---|
| `normal` | green | allow — both signals quiet |
| `monitor` | amber | log only — novel pattern, probability low |
| `review` | orange | alert — one strong signal |
| `high-risk` | red | alert — both strong signals |

The same language flows through the badge chips, the alert list, and the
exported reports, so a stakeholder "reading red" understands instantly.

## 3. Under the hood

### The honest tech stack

This prototype is built in **Python**, end to end:

| layer | technology | why |
|---|---|---|
| Dashboard (UI) | **Streamlit** | Python-native, no front-end build step, and the same scoring code runs in-process — the UI literally *is* the pipeline |
| API service | **FastAPI** | `/predict` and `/predict/batch` expose the same model over HTTP, with schema validation |
| Classification | **LightGBM** (native, installed) | fast, handles imbalance, produces tree contributions for explanations |
| Calibration | **Isotonic regression** (fit on validation only) | makes 0.8 mean "~80% fraud," selected over Platt by Brier score |
| Novelty | **IsolationForest + ECDF** | identity-free "how unusual is this?" second opinion; ECDF turns scores into percentile rankings |
| Everything else | pandas, numpy, scikit-learn, joblib, matplotlib/plotly | data handling, metrics, persistence, figures |

> **A deliberate note on the front end.** A separate prompt in this project
> asks for a React + Vite + Tailwind CSS v4 styling guide. That guide exists
> as a *design language* reference in `docs/STYLING_GUIDE.md` (Tailwind v4
> utility classes, theme tokens). The **running dashboard today is
> Streamlit**, which renders the exact same layout, colours, and risk-band
> language natively. We chose Streamlit for the demo because it keeps the
> deliverable runnable on one machine with one runtime — a React rewrite is
> a port of the same components, not a different product.

### The 10-stage training pipeline

`run_pipeline.py` is the one command that builds everything:

```
[1] load + validate raw CSV           [6] calibrate probabilities (val)
[2] time split + unseen-customer holdout [7] fit novelty model + ECDF (train)
[3] fit segment/global priors (train only) [8] select risk thresholds (val)
[4] leakage-safe feature replay       [9] streaming replay of val + test
[5] train + compare classifiers       [10] evaluate + leakage report + figures
```

Key details that map to the problem statement:

- **Split [2].** The data is split by *time* (no shuffling), then a disjoint
  set of **1,660 customers is removed from the training fold only** and moved
  into evaluation — this is what creates the "unseen customer" cohort the
  whole project is about.
- **Split [3].** Priors (baseline amount/risk per history stratum) are fit
  *only on training rows* — the test fold can never influence them.
- **Replay [4/9].** Features are computed **point-in-time**: for each
  transaction we only use state from earlier transactions. The state is
  committed *after* scoring, never before — so no future information leaks.
  A parity check confirms replayed features match the provided label
  columns 100% of the time.
- **Model [5].** LightGBM wins by PR-AUC on validation (0.8865) vs LR
  (0.7834) and Random Forest (0.8685).
- **Budget [8].** The Review/High-Risk thresholds are chosen on validation so
  that alert volume matches the configured 2% budget *and* the high-risk
  share stays inside its cap.

### The streaming runtime (what an actual API call does)

One function, [`StreamingProcessor.score_transaction`](src/streaming_processor.py),
is the single production code path used identically by the API, the
dashboard, and the CLI:

```
validate → point-in-time features → fraud probability → novelty score
        → risk band → plain-language explanation → alert decision → commit
```

*Nothing* is recomputed or re-fitted at call time; the state (per-customer,
per-merchant, per-device, global) lives in memory and advances with every
transaction. A 2000-row file scores in ~3 seconds in the dashboard.

### The features

The classifier sees ~39 features, all computed as-of-the-instant:

- **velocity** — count / money moved in the last 5 min, 30 min, 24 h;
- **amount** — z-score of this amount vs the customer's own baseline (with
  shrinkage toward a segment prior when history is thin), plus a "spike"
  flag;
- **time** — circular deviation of the hour-of-day vs the customer's usual
  pattern;
- **device / merchant novelty** — has this device/merchant been seen before,
  and how trustworthy is the accumulating history;
- **history** — transaction counts, confidence, and stratum labels.

**No raw IDs are ever model features** — cards, merchants, and devices only
key the internal streaming state. This is what keeps the system honest: it
cannot "cheat" by memorising card numbers.

### The freshness rule (why this isn't a stale model)

"New is suspicious" is encoded with a *shrinkage coefficient* `kappa=5`
pseudo-observations: a customer with zero history is scored as if they had
5 lukewarm transactions — statistically cautious, not paranoid. The novelty
channel can then only *nudge* the decision (requires `p >= 0.02`), never
make it alone.

## 4. What makes it special

1. **Built for the real attack surface — unseen customers.**
   Most submissions optimise overall accuracy, which quietly means "accuracy
   on the customers we already know." We optimise **PR-AUC and recall at a
   fixed alert budget on a *held-out unseen-customer* cohort** — the exact
   asymmetry fraudsters exploit. The unseen cohort's PR-AUC (0.9002) even
   *beats* the all-data number.

2. **Two independent signals instead of one.** Supervised fraud probability
   + unsupervised identity-free novelty, fused by a 2×2 decision matrix.
   One signal is a memory, the other is an instinct — together they catch
   both old patterns and never-seen-before attacks.

3. **Novelty never alerts alone.** `p >= p_floor` is required for the
   novelty channel to escalate, which is what keeps legitimate first-time
   customers out of the alert queue — a design decision with a measured
   number behind it (false-positive rate on legit first-time customers).

4. **A leak-proof pipeline, verified.** Time splits, an unseen-customer
   holdout removed from train *only*, priors fit on train only,
   point-in-time features with commit-after-scoring, and a **parity check**
   (replayed features must reproduce the provided columns 1.0000 of the
   time). The leakage report ships as an artifact, not a claim.

5. **A hard alert budget, not a soft aspiration.** Thresholds are re-derived
   as quantiles so *exactly* the configured share of transactions alert;
   an in-streamic budget controller hard-caps per-hour/per-day escalation,
   and the slider lets an analyst live-tune the trade-off between catch-rate
   and analyst load.

6. **Explainable on every single decision.** Each alert names its top
   contributing factors grouped into families (velocity, amount, time,
   device, merchant, history), phrased as *contribution*, never causation —
   defensible in front of an auditor, not a black box.

7. **Every dashboard number is recomputed from your own file.** Section 3
   shows PR-AUC / recall / known-vs-unseen for *the rows you just uploaded*,
   so nobody mistakes a pretty static hold-out graph for evidence.

## 5. Restrictions and compliance

| restriction / concern | how we comply |
|---|---|
| No leakage (temporal) | time-based splits, no shuffle; features point-in-time; state committed after scoring; parity re-check = 1.0000 |
| No leakage (identity) | 1,660 customers held out of train only; priors fit on train only; raw IDs never appear in model features |
| Fixed alert workload | `alert_budget_pct=2%`, `high_risk_share_cap_pct`, per-hour/per-day caps all in one config file; thresholds chosen on validation |
| Cold start must not panic | shrinkage priors (kappa=5) + `p_floor_for_novelty` gate so legit first-timers are logged, not harassed |
| Explainability | every decision carries grouped, plain-language contributing factors |
| Single-machine demo | Python-only runtime (Streamlit + FastAPI + pandas + LightGBM); one command trains, one command serves, one command exports |

[Demo commands](docs/../../CLAUDE.md) are one-liners:
`python run_pipeline.py`, then `uvicorn api.main:app`, then
`streamlit run dashboard/app.py`.

## 6. Results

All numbers below come from `reports/evaluation_report.json` and
`reports/qa_report.md` on the model trained with **LightGBM** on the
synthetic stream (~120k rows), evaluated on a held-out test fold of
**23,928 transactions** (459 fraud, 1.92% rate). The pipeline is fully
reproducible: `python run_pipeline.py` regenerates everything from scratch
with a fixed seed.

### Headline quality metrics (held-out test fold)

| metric | value |
|---|---|
| **PR-AUC** (all) | **0.894** |
| **PR-AUC on unseen customers only** | **0.900** |
| ROC-AUC | 0.994 |
| Recall @ 2% alert budget (all) | **82.8%** |
| Precision @ 2% alert budget (all) | 79.3% |
| Brier score (calibration quality) | 0.0049 |
| ECE (expected calibration error) | 0.0018 |

In plain language: with a *2% analyst budget* (about 1 in 50 transactions may
alert), the system still catches **~4 out of every 5 frauds**, and ~79% of
what it flags is genuinely fraud — while never having seen the fraudsters'
identities before.

### Alert behaviour at the fixed 2% budget

- 305 alerts fired on 23,928 transactions (1.27% — under budget).
- Alert **precision 0.708** — 71% of alerts are real fraud.
- Alert **recall 0.471** — roughly half of *all* fraud appears in the alert
  stream; the alert-*budgeted* recall (taking the top 2% by risk) is 0.828.
- Cost metric: **1.41 alerts per fraud actually caught** — a small queue for
  the analyst to clear.

### Generalisation (the project's whole reason to exist)

The unseen-customer cohort (12,199 rows, all 459 frauds) holds
PR-AUC 0.9002 and recall@2% of 0.8301 — **no degradation vs the known
cohort**. Legitimate first-time customers are *not* punished: the false
positive rate on legitimate first-time-customer transactions is **0.042**
(59 alerts on 1,378 such rows), and 0.0% for first-time merchants.

### Leakage report (auto-generated)

`temporal_integrity=time-based splits, no shuffle`; `unseen_customer_holdout=
1,660 customers removed from train only`; `priors_fit_on=train`;
`features_computed_as_of=point-in-time, commit after scoring`;
`raw_ids_in_features=false`; `velocity_8min_match_rate=1.0`;
`amount_spike_match_rate=0.9962`.

### The QA suite

`python scripts/qa_suite.py` runs **18 checks** and writes
`reports/qa_report.md`: **18/18 passed** — app boots with 0 exceptions, all
three dashboard sections render, upload→analyse works end-to-end, the
dynamic System-Evaluation section updates, `/predict` returns all required
fields in ~25 ms with probabilities in [0,1], batch preserves request
order while scoring chronologically, invalid input is rejected (HTTP 422),
determinism holds, streaming state accrues history, raw IDs are absent from
model features, the budget **hard-caps** a 60-transaction fraud burst (3
alerts vs a 3 cap), and the leakage report flags no raw-ID leakage.

### The plot twist we are proud of

We also ran the *entire* pipeline against **IEEE-CIS fraud data** — a real,
~590k-row public benchmark — through a schema adapter
(`scripts/adapt_ieee_cis.py` + `config.ieee_cis.yaml`). The architecture
transferred unchanged (same streaming replay, same split logic, same budget,
no code edits to the core). That run is preserved with figures in
`reports/ieee_cis_run/`. The absolute numbers are lower than on the
synthetic demo stream (PR-AUC ~0.11 with a deliberately *light* LightGBM
config) — expected and honest: real-world data is far noisier, device info
is missing for ~76% of rows, and the identity proxies (card `card1`, address
`addr1`) are coarse. The point of that run is **portability**: swap the CSV,
keep the architecture, get a report. Attempting to *tune* for a competitor
benchmark would be outside this hackathon's scope; the engineering claim is
"dataset-agnostic by construction."

## 7. Judge Q&A defence

**Q1. How do we know you didn't leak the future into your model?**
Because leakage is *audited*, not assumed. We split by time (no shuffle),
moved 1,660 customers into the holdout *only out of the training fold*, fit
priors on train rows only, and compute every feature from state that existed
*before* the transaction — the state commit happens **after** scoring in the
source code. As an independent check, `run_pipeline.py` replays the data and
requires replayed features to match any provided columns (`velocity_8min`
1.0000, `amount_spike` 0.9962). The report is generated as an artifact, not
a paragraph.

**Q2. Why not just report accuracy like everyone else?**
Because accuracy on an imbalanced, adversarial problem is theatre. Fraud is
~2% of rows, so a model that flags nothing is 98% "accurate." The numbers a
judge should look at are **PR-AUC** and **recall at a fixed alert budget** —
real operating points — and above all the **unseen-customer cohort**, which
is exactly where criminals operate and where most models quietly collapse.

**Q3. Why did you split off *unseen customers*? Isn't that artificial?**
It is the *most realistic* split available. Fraud rings rotate freshly
stolen card numbers daily; at the moment of the first transaction on a new
identity there is no history by definition. A detector that only works after
it has seen you is a detector that misses induction-day attacks. Removing
customers from train is the only way to honestly measure that.

**Q4. Novelty by itself seems dangerous — how do you stop it crying wolf?**
We gated it: the novelty channel can only push toward an alert when
`fraud_probability >= p_floor` (0.02). So an ordinary new customer is logged
as a monitor, not harassed. We even measure it — a 0.042 false-positive rate
on legitimate first-time-customer transactions — the exact "new ≠ fraud"
error the problem statement warns about.

**Q5. Why two models? Isn't one enough?**
One model memorises patterns it has seen; novelty is the *instinct* that
works with no memory. Criminals specifically target brand-new identities, so
a single memorising model is structurally blind at the most critical moment.
The two-signal fusion is not complexity for its own sake; it is the minimum
architecture that covers both regimes.

**Q6. What does "calibrated" actually guarantee?**
After isotonic calibration (fit on *validation*, never test), a fraud
probability of 0.8 means "across all transactions the model scored ~0.8,
about 80% were fraud." Measured Brier 0.0049, ECE 0.0018. That property
matters operationally: the analyst acts on the score, so the score must mean
what it says.

**Q7. How do you handle new, never-seen customers at runtime?**
With shrinkage priors: history is blended toward segment/global baselines
via `kappa=5` pseudo-observations, so cold-start estimates are cautious but
not zero. The result is a PR-AUC of 0.90 on the never-seen cohort — equal to
or better than the known cohort. Cold start is a *feature design*, not a
fallback.

**Q8. Why a hard alert budget? What if the model wants to alert more?**
Because manpower is the binding constraint in anti-fraud, not model
bandwidth. An alert nobody reads has the same cost as a miss on a
high-volume system. We let the operator choose the budget (default 2%,
slider 0.5–10%) and the system re-derives thresholds as quantiles so the
alert *volume exactly matches the operator's capacity*, and a stream-level
budget controller hard-caps per-hour/per-day escalation anyway.

**Q9. What were the other models, and why LightGBM?**
We compared logistic regression, Random Forest, and LightGBM by PR-AUC on
validation: 0.7834, 0.8685, **0.8865**. LightGBM wins (handles imbalance via
`scale_pos_weight`, fast, produces tree contributions used by the
explanations). We still keep the comparison table in the model artifacts so
the choice is evidence-based.

**Q10. Is your accuracy "overfit" to the synthetic dataset?**
The synthetic stream exists because a hackathon needs a runnable data flow;
the *numbers* are honest for that synthetic distribution and labelled as
such. But the architecture is not tied to it: we ran the same unmodified
pipeline against the real **IEEE-CIS** benchmark (reports preserved in
`reports/ieee_cis_run/`) to demonstrate portability, and every design choice
(memory-less novelty, point-in-time features, unseen-customer holdout, alert
budget) is exactly the machinery you would use on production stream data.

**Q11. How fast is it?**
A single `/predict` call answers in ~25 ms including explanation. A 2,000-row
file scores in ~3 seconds in the dashboard (vectorised batch path). The
streaming API processes chronologically ordered HTTP batches with identical
semantics to the single-transaction path.

**Q12. What's your biggest limitation?**
Honesty first: numbers on the synthetic stream are a *proof of design*, not a
benchmark claim; the IEEE-CIS port with coarse identity proxies shows the
architecture transfers but is not tuned against that benchmark. Real-world
adoption would require production streaming infrastructure (Kafka/Spark),
online feature stores, and drift/retraining policies. Within the hackathon's
constraints, we deliver a leak-safe, explainable, budget-aware prototype
whose entire training → scoring → dashboard → QA loop is verifiable from a
single terminal.

---

*Team NOOB CODERS — generated and verified against the live repository
(numbers: `reports/evaluation_report.json`, `reports/qa_report.md`).*