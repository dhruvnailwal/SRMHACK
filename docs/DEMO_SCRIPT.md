# Prototype Demo — Video Recording Script

**Team:** NOOB CODERS · **Event:** SRM Hackathon (Engineers' Day — ML Challenge)
**Target length:** ~5 min · **One long take or cut per segment.**

Before you record, run once and leave running:
- Terminal A: `python -m uvicorn api.main:app --reload`  (FastAPI :8000)
- Terminal B: `python -m streamlit run dashboard/app.py` (Streamlit :8501)
- Terminal C: at a project prompt, ready to type the CLI line.

Screen size: browser windows side by side (dashboard left, API docs right).
Dark/navy theme is already applied. Speak slowly; read the numbers you see
live, not these canned ones (they vary per run).

---

## Segment 0 — Cold open (0:00–0:15)

**Screen:** full project in a folder view (`fraud_detection/`, `api/`,
`dashboard/`, `scripts/`, `reports/`).

**Narration:**
> "This is our fraud-detection prototype for the SRM hackathon. One shared
> streaming engine powers three surfaces — a REST API, a dashboard, and a
> command-line scorer. We built it to solve one hard problem: catching fraud
> on customers the model has *never* seen."

---

## Segment 1 — CLI: the model runs on real rows (0:15–1:10)

**Screen:** Terminal C.

**Action:**

```
python scripts/analyse_file.py data/processed/test_fold_file.csv --budget 2
```

**Narration while it streams:**
> "First, the no-server path. We point the scorer at a held-out file of
> transactions the model has never trained on. Notice it streams row by row —
> every transaction is scored exactly the way a live feed would score it,
> and each row is committed to history *after* scoring, so no row ever sees
> its own future.

> Here it printed the risk-band distribution — normal, monitor, review,
> high-risk — and the alert count, hard-capped at our 2% budget. And it
> breaks the scores out by known versus unseen customers — because acting
> correctly on *unseen* identities is the whole point.

> It also wrote a report folder — all five formats."

**Action:** briefly open the `_analysis` folder, open the CSV, show the
columns `amount`, `fraud_probability`, `novelty_score`, `risk_band`,
`rank_score` filled for every row.

**Narration:**
> "Every download includes the per-transaction values and readings — amount,
> fraud probability, the novelty score, risk band and rank — not just a
> summary."

**Cut if recording in takes;** otherwise press on.

---

## Segment 2 — FastAPI: live single + batch scoring (1:10–2:00)

**Screen:** browser → `http://localhost:8000/docs` (Swagger UI).

**Action:** expand `POST /predict`, use a sample transaction:

```json
{
  "transaction_id": "VIDEO001",
  "timestamp": "2026-09-20 12:05:00",
  "customer_id": "CUST_77",
  "merchant_id": "MERCH_9",
  "device_id": "DEV_3",
  "amount": 4950.00
}
```

Click **Execute**. Point at the JSON response.

**Narration:**
> "The API answers in milliseconds. Every decision carries the same four
> outputs: fraud probability, novelty score, risk band, and plain-language
> contributing factors — phrased as *contributions*, never as a guilty
> verdict. Here, amount is nearly five thousand — far above the baseline —
> and the card tells us in words what pushed the score."

**Action:** scroll to `POST /predict/batch`, paste a 4-row velocity burst
(same customer, 4 quick transactions), Execute.

**Narration:**
> "Same engine, batch mode. The rows are replayed chronologically, so the
> second transaction already sees history built by the first — you can watch
> the velocity feature climb across the batch. That is what a live
> transaction stream does."

---

## Segment 3 — Dashboard Overview (2:00–2:35)

**Screen:** Streamlit → section **"1. Overview"**.

**Action:** scroll the overview, pause on the risk matrix table.

**Narration:**
> "The dashboard opens on the design. One classifier trained on labels, one
> unsupervised novelty model that needs no labels at all. Two independent
> signals. They combine in this matrix — and the rule that matters most is
> the bottom-left cell: *novelty alone never alerts*. A legitimate
> first-time customer is logged, not escalated."

---

## Segment 4 — Upload & Analyse on your own file (2:35–3:45)

**Screen:** scroll to **"2. Upload & Analyse"**.

**Action:** either click **"Analyse sample test file"** or drag the CSV into
"Drop your transaction CSV here", then click the **Analyse** button.

**Narration:**
> "Upload any CSV with our schema, or hit the one-click sample. The Analyse
> button runs the *same* `StreamingScorer` the API uses — this is not a
> canned result."

**After results render** (section **"File metrics & model evaluation"**):
point at the metric cards.

**Narration:**
> "Everything here is computed from *this file* live — alert rate, PR-AUC,
> recall at budget, precision — and it re-computes when you change the
> budget slider. It stays honest: headline numbers come from the unknown
> cohorts, not the whole file."

**Action:** scroll to **"Per-transaction outputs"**, hover one row's risk
badge; scroll to **"Suspected-fraud list"**; scroll to **"Key insights"**.

**Narration:**
> "Per-transaction outputs with colour-coded risk bands. A suspected-fraud
> list sorted by risk priority. And generated key insights — the same
> sentences a report analyst would write, here auto-generated from the data."

---

## Segment 5 — Exports, all five formats (3:45–4:30)

**Screen:** scroll to **"Settings & export"**.

**Action:** pick each format in the radio (CSV, JSON, Markdown, TXT, PDF),
click **Download CSV report** / etc. Show the downloaded PDF opening.

**Narration:**
> "Five export formats from one button. The PDF leads with the summary, then
> a full per-transaction table — every row's amount and readings. The
> Markdown and TXT carry the evaluation report including every scored row.
> CSV and JSON give downstream tools the raw matter — including the reason
> text per transaction."

---

## Segment 6 — The fraud burst + close (4:30–5:00)

**Screen:** back to Terminal C.

**Action:**

```
python scripts/demo_scenario.py
```

**Narration:**
> "The final clip is the brief's demo scenario replayed: a sudden velocity
> burst. Watch the velocity and history-confidence features climb exactly as
> designed — this is the signal the model was trained to find.

> On our synthetic stream, PR-AUC on unseen customers is about 0.83 with a
> zero false-positive rate on legitimate first-time customers — and we also
> ran the *unmodified* pipeline on the real IEEE-CIS benchmark to prove the
> architecture transfers.

> Nothing here memorises a customer. It measures behaviour, keeps two
> opinions independent, and explains itself in plain language. Thank you."

---

## Pre-recording checklist

- [ ] `uvicorn` + `streamlit` running, dashboards side-by-side
- [ ] Terminal C at project root
- [ ] Screen recorder ≥1080p, mic tested
- [ ] Pause on the risk-matrix and the per-transaction sections (judges read slowly)
- [ ] Read live numbers, not the canned ones — then the QA numbers in
      `reports/qa_report.md` match what you said.
- [ ] Swagger UI open with the sample `CUST_77` JSON pre-filled
- [ ] The burst JSON pre-pasted in `/predict/batch`
- [ ] Finder/Explorer showing the `_analysis` folder (+ an open CSV)

## Honesty guardrails for the narration

- "Contributing factors", never "the reason it's fraud".
- Synthetic-stream numbers are *synthetic*; IEEE-CIS numbers are
  *real-data* and lower (ROC-AUC ~0.72) — say "real-data result", not a
  headline.
- Every number you quote should be visible on screen at that moment.