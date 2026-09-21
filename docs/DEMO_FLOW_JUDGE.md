# Demo Flow — Aligned with `docs/DEMO_SCRIPT.md` + Judge Defense Points

> Companion to `docs/DEMO_SCRIPT.md` (the recording script) and
> `docs/JUDGE_PREP_GUIDE.md` (Q&A bible). For each demo segment: what to
> show, what to say, the **correct** number to quote (from the shipped
> artifacts), and the judge question that segment invites + where the
> answer lives.

## Setup (before you begin)

- Terminal A: `python -m uvicorn api.main:app --reload` (FastAPI :8000)
- Terminal B: `python -m streamlit run dashboard/app.py` (Streamlit :8501)
- Terminal C: project prompt, ready for the CLI scorer.
- Keep `docs/JUDGE_CHEAT_SHEET.md` printed beside you.

**IMPORTANT — two fixes to the existing script before you present:**

1. **`scripts/demo_scenario.py` does not exist.** `DEMO_SCRIPT.md` Segment 6
   tells you to run `python scripts/demo_scenario.py` — there is no such file
   in `scripts/` (only `adapt_ieee_cis.py`, `analyse_file.py`,
   `generate_dataset.py`, `qa_suite.py`). Either *(a)* remove that step, or
   *(b)* reuse Segment 6 to replay the batch velocity-burst in Swagger
   (`/predict/batch`) and narrate it — that is the "fraud burst" moment and
   it works with shipped code.
2. **Stale numbers.** The Segment-6 narration says *"PR-AUC on unseen
   customers is about 0.83 with a zero false-positive rate on legitimate
   first-time customers."* The shipped artifacts say:
   - synthetic run: unseen **PR-AUC 0.9002**, first-time-customer FP rate
     **4.2%** (`reports/02_evaluation_report.md:42,49`),
   - real IEEE-CIS run: first-time-customer FP **0.0%**, unseen recall@2%
     **32.6% vs 13.6%** (`reports/ieee_cis_run/02_evaluation_report.md:49,42`).
   Say: *"On the real IEEE-CIS data, zero false positives on legitimate
   first-time customers, and unseen customers catch 2.4× the fraud of known
   ones at the same budget."* Never claim zero FP on the synthetic stream.

---

## Segment 0 — Cold open (0:00–0:15)

**Screen:** repo folder view.
**Narrate:** *"One shared streaming engine powers three surfaces — an API, a
dashboard, a CLI scorer — built for one hard problem: catching fraud on
customers the model has never seen."*

**Watch for the question:** “What does 'one engine, three surfaces' actually
mean?” → *One `StreamingProcessor` code path reused by pipeline replay
(`run_pipeline.py`), FastAPI (`api/main.py`) and the dashboard
(`dashboard/app.py`) → no train/serve skew.* (Guide §2.3, Q13.)

---

## Segment 1 — CLI scorer (0:15–1:10)

**Run:**
```
python scripts/analyse_file.py data/processed/test_fold_file.csv --budget 2
```
**Narrate:** streaming, row-by-row; `commit` after `score` (no future leak);
risk-band distribution; alerts hard-capped at 2%; **known vs unseen
breakdown** — the whole point.

**Numbers to quote (live, from the file):** the alert count and band counts
printed by the tool. Don't improvise — read what appears.

**Watch for the question:** “How do I know this isn’t memorized?” →
*Chronological split + 1,660 customers removed from TRAIN only*
(`src/preprocessing.py:47`) *+ parity check 100%*
(`run_pipeline.py:53`). (Guide Q4, Q19.)

---

## Segment 2 — FastAPI single + batch (1:10–2:00)

**Show:** `/docs` → `POST /predict` with a high-amount sample; then
`/predict/batch` with a 4-row velocity burst.

**Narrate:** single = ms latency, four outputs; batch = same engine, rows
replayed chronologically so **the 2nd row sees history built by the 1st** —
watch velocity climb.

**Point at:** `fraud_probability`, `novelty_score`, `risk_band`, and the
plain-language `top_contributing_features`.

**Watch for the question:** “Is that response explainable / causally
phrased?” → *Exact TreeSHAP (`pred_contrib=True`) grouped into reasons;
descriptive, not causal* (`src/explainability.py:116`). (Guide Q15.)

---

## Segment 3 — Overview + the 2×2 (2:00–2:35)

**Show:** Overview; pause on the decision matrix.
**Narrate:** supervised signal + label-free novelty signal; the bottom-left
rule — **novelty alone never alerts** (`src/risk_engine.py:50`).

**Numbers to quote:** ECE **0.0018** (a 0.8 really means ~80%), test PR-AUC
**0.8938**, alert precision **0.708** · recall **0.471** · **1.41 alerts per
fraud caught** at 2%.

**Watch for the question:** “Why does 'calibrated' matter?” → *Operational
meaning + threshold budget guarantees* (Guide Q5). And “novelty AUROC looks
low” → segmented answer in Guide Q2.

---

## Segment 4 — Upload & Analyse (2:35–3:45)

**Show:** drag CSV → **Analyse** → metric cards → budget slider → per-txn
cards → suspected-fraud list → key insights.

**Narrate:** *"Everything here is computed from THIS file, live — not a
canned result. Move the budget slider and thresholds re-derive, alerts
recompute."*

**Watch for the question:** “Isn't re-deriving thresholds on the file tuning
on test?” → *No: it's a what-if console; production uses validation-derived
`models/thresholds.json`; model weights never change.* (Guide Q18.)

---

## Segment 5 — Exports (3:45–4:30)

**Show:** csv / json / md / txt / pdf radio → download → open PDF.
**Narrate:** every format carries **per-transaction readings** (amount,
fraud probability, novelty, risk band, rank), not just a summary.

---

## Segment 6 — Fraud burst + close (4:30–5:00)

⚠️ As noted above, fix the missing `demo_scenario.py` first. Recommended
replacement: replay the **4-row velocity burst in `/predict/batch`** and
show the second demo row from `reports/artifacts/demo_examples.json`:

- **TXN00101785** → p=0.9697, band **review**, ALERT.
  Reasons: velocity elevated (2 in 30 min) · amount 0.5σ above baseline ·
  history confidence 0.29 (only 2 prior txns) · device 14× known but behavior
  risky · 21:00. No identity history needed — all stream facts.

**Narrate:** *"It sees these attacks through behavioral stream context —
velocity, thin-history confidence, deviation-from-segment, device loyalty —
not a profile the attacker rotates around."*

**Close with transfer proof:** *"Same unmodified pipeline on the REAL
IEEE-CIS benchmark: 261,982 rows. Unseen customers catch 2.4× more fraud at
the same budget, first-time FP rate 0%, calibration holds (ECE 0.0037)."*

**Watch for the question:** “Why are the IEEE numbers lower?” → *Deliberate
identity-free mapping — we discarded the columns winning solutions use —
proving architecture transfer, not metric superiority.* (Guide Q9.)

---

## Segment-by-segment question map (print this table)

| Segment | Likely question | Answer in Guide |
|---|---|---|
| Cold open | “One engine, three surfaces?” | §2.3 / Q13 |
| CLI | “Memorization?” · “Parity check?” | Q4, Q19 |
| API | “Explainability / causality?” | Q15 |
| Overview | “Why calibrated?” · “Novelty AUROC low?” | Q5, Q2 |
| Upload | “Slider = tuning on test?” | Q18 |
| Exports | “Scalability?” | Q13 |
| Close | “IEEE numbers low?” · “Synthetic?” | Q9, Q8 |

## Last slide bullet (when the floor opens)

> "On the cohort that breaks memorizing fraud models — customers the model
> has never met — detection does not degrade: PR-AUC 0.9002 vs 0.8938
> overall in our test replay, and on real IEEE-CIS data unseen customers
> catch 2.4× the fraud at the same budget. One architecture, two datasets,
> one guarantee: we work on the fraud you haven't met yet."