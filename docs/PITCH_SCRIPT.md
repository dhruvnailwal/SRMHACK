# Pitch Deck — Speaker Script

**Team:** NOOB CODERS · **Event:** SRM Hackathon (Engineers' Day — ML Challenge)
**Pacing:** ~25 seconds per slide → whole pitch ≈ 2:55.

*Rules: say "contributing factors / contributed", never "the reason it's
fraud". All numbers are measured from this repo — synthetic-stream results
are labelled as such; IEEE-CIS numbers are the honest real-data results.*

---

## Slide 1 — Problem  (~25 s)

"Every fraud model quietly assumes it has seen you before. But in the real
world a huge share of fraud comes from identities the bank has *never*
met. The brief asks us to keep detecting that fraud while not punishing
legitimate first-time customers.

So this is really two questions, not one: *is the behaviour fraud-like?* and
*is it behaviour we've watched before?* — under streaming, one-pass
decisions, and a hard cap on how many alerts we may raise. And our headline
metric is what a fraud team actually optimises: precision and recall at a
fixed alert budget, reported separately for known and unseen customers."

---

## Slide 2 — Solution  (~25 s)

"We run a streaming pipeline: validate the row, build point-in-time features,
score a calibrated fraud probability, score an *identity-free* novelty score,
map them through a risk matrix into a band, and commit — in that order, every
time.

The two signals are deliberately independent. Probability comes from a
LightGBM classifier with isotonic calibration. Novelty — how atypical the
behaviour is versus everything we've ever observed — comes from an
IsolationForest and an ECDF, using no customer identity at all.

The matrix only escalates when the probability clears a floor, so novelty
alone can never flag someone. And every transaction carries plain-language
contributing factors — contribution, never causation."

---

## Slide 3 — Technical Implementation  (~25 s)

"Everything hangs off one package, `src/fraud_detection/`. A `HistoryStore`
keeps streaming per-customer, per-segment and global state, and feature
extraction is a pure point-in-time function — the commit always runs last,
which is what makes it leak-safe.

The key architectural choice: one shared scoring object drives *all three*
fronts. The FastAPI service's `/predict` and `/predict/batch`, the Streamlit
dashboard, and the command-line scorer execute the identical validate →
score → commit sequence, so a historical replay produces exactly what a live
stream would.

Training is a nine-stage script with an unseen-customer holdout and a time
split. And we verify rather than assume: a leakage report with a parity
re-check, and an 18-check QA suite — all 18 green."

---

## Slide 4 — Innovation  (~25 s)

"What's genuinely different here: first, *unseen-customer fairness is built
into the data pipeline*, not bolted on — a customer holdout plus a strict
time split, and priors fit on training data only.

Second, *novelty ≠ fraud is enforced in the policy layer*, by the matrix,
not just in a dataset.

Third, explanations are exact TreeSHAP contributions grouped into six human
families — velocity, amount, time, device, merchant, history — so an analyst
glances at a card and sees what *moved the score*, never a guilty verdict.

And the whole design is identity-free by construction — the QA suite
verifies no raw card or account ID ever reaches a model feature."

---

## Slide 5 — Restriction Handling  (~25 s)

"Being upfront about constraints: this environment had no internet, so
LightGBM and FastAPI couldn't be pip-installed during the build. Two
consequences.

First, the classifier auto-detects LightGBM at runtime and otherwise falls
back to a HistGradientBoosting classifier — no code changes, you just rerun
training. Second, the API and dashboard couldn't be executed in place, so we
proved them through the FastAPI test client and Streamlit's AppTest harness —
that's where the 18 QA checks run.

Explanations fall back to a perturbation-based contribution estimate when
the native LightGBM booster isn't present. And we're explicit that the
synthetic stream is illustrative, not a benchmark — the pipeline is
dataset-agnostic, with a working adapter for real IEEE-CIS data."

---

## Slide 6 — Prototype Progress  (~25 s)

"This is a working end to end: training, CLI scoring, the REST API, and a
dark-navy Streamlit dashboard, all live.

The CLI scores a CSV with no server at all — a 2000-row run raised 52
alerts at a 2.6 percent budget with 46.5 percent unseen customers. Exports
come in five formats — CSV, JSON, Markdown, TXT and PDF — every one carrying
per-transaction amount, fraud probability, novelty, risk band and rank.

The delivery set is written for a non-expert judge: a master submission, a
styling guide mapping our spec to the actual UI, a data-journey doc, the QA
report, and the IEEE-CIS run.

And the transfer proof: the *unmodified* pipeline ran the real IEEE-CIS
benchmark — 300k rows, all ten stages — with results preserved under
`reports/ieee_cis_run/`."

---

## Slide 7 — Key Results  (~25 s)

"The numbers, honestly labelled.

On our synthetic stream, PR-AUC for unseen customers is 0.826, ROC-AUC
0.997, and recall at a 2 percent budget for unseen customers is 83.8% —
with a zero-percent false-positive rate on legitimate first-time customers.
A re-run confirms stability at 0.894 PR-AUC.

On real IEEE-CIS data, ROC-AUC is 0.716 with calibration error under half a
percent and alerts held at 1.68% of volume — not a headline, but it proves
the streaming and calibration engine transfers to hard real data.

Budget control is verified, not just claimed: a 60-transaction fraud burst
produced exactly three alerts against a three-alert cap.

And the leakage report and parity re-check clear time- and customer-integrity
with a clean bill. Thank you."

---

## Judge Q&A quick answers

| judge asks | answer (30 s) |
|---|---|
| "Is 0.716 ROC on IEEE-CIS a failure?" | "It's the honest result of an unmodified prototype on real, noisy data — the point is transferability and calibration, not the score. The two-signal architecture and budget caps are what generalise." |
| "How do you stop novelty from nuking first-time customers?" | "Two layers: the ECDF/IsolationForest never sees identity, and the risk matrix requires `p ≥ p_floor` — novelty alone can't escalate. 0.0% FP on legit first-timers." |
| "Where's the leakage?" | "Point-in-time commits, customer holdout removed from TRAIN only, priors on TRAIN only, time split — plus the independent parity re-check in the leakage report. QA check H2 shows the store accumulates history exactly one txn per call." |
| "Why two signals?" | "Fraud-like and out-of-pattern are different evidence. A single model collapses them; the matrix keeps them independent and the explanation honest." |
| "Can it go into production?" | "The dashboard/API are demos; the `StreamingScorer` code path is the production object. Swap the schema adapter for your bank's feed and re-run training." |