# Judge Cheat Sheet — 2-Page Edition

> Print this. Everything on one sheet. Numbers match `reports/02_evaluation_report.md`,
> `reports/ieee_cis_run/`, `reports/qa_report.md`, `models/thresholds.json`.

## THE PITCH (30 seconds)

**Problem:** Fraud targets the *unseen* — attackers rotate cards, devices, merchants. Memorizing models fail exactly there.
**Fix:** Two independent signals → one decision. `fraud_probability` (calibrated LightGBM) × `novelty_score` (IsolationForest + ECDF). Novelty **never alerts alone** — new-but-normal customers are protected.
**Proof:** Test-fold numbers from the **real production streaming path** + the identical pipeline runs on **real IEEE-CIS** data (architecture transfer).

**Value:** Stop fraud on customers the model never met, at an analyst-sustainable alert budget, with plain-language reasons on every alert. Dashboard slider answers "what if we can only handle 1%?"

## THE ONE-LINERS

- "Novelty alone never alerts." → `p ≥ p_floor` (`src/risk_engine.py:50`)
- "A row never sees its own future." → `commit_row` runs last (`src/streaming_processor.py:194`)
- "History is never invented." → cold start = shrinkage to segment/global priors (`src/feature_engineering.py:266`)
- "Raw IDs are not features." → QA check I, `reports/qa_report.md:22`
- "Thresholds come from validation quantiles, never test." → `src/risk_engine.py:86`

## NUMBERS TO QUOTE (synthetic run, seed 42)

| Metric | Value |
|---|---|
| Dataset | 119,636 txns · 3.01% fraud · 365 days · 8,300 cust |
| Split | 60/20/20 chronological + **1,660 customers held out of train** |
| Model choice (val PR-AUC) | LR 0.7834 **<** RF 0.8685 **<** **LightGBM 0.8865** |
| Calibration | Isotonic (by Brier, halved val) → **ECE 0.0018**, Brier 0.0049 |
| Test | PR-AUC 0.8938 · ROC 0.9937 · precision .7082 / recall .4706 |
| Alerts | **305** of 23,928 = **1.27%** (under 2% budget) · alert prec .7082 · **1.41 alerts/fraud** |
| Thresholds | p_review 0.1613 · p_high 0.9697 · nov 0.9/0.97 · p_floor 0.02 |
| Unseen cohort | PR-AUC **0.9002** ≥ 0.8938 overall · recall@2% 0.830 |
| First-time cust recall@2% | **0.931** · new-device recall@2% **1.00** |
| FP first-time (synthetic) | customer 4.2% · device 2.2% · merchant 0% |
| Parity (leak-proof proof) | velocity_8min **100%** · amount_spike **99.62%** |
| QA | **18/18 pass** (~23s) |

## NUMBERS TO QUOTE (real IEEE-CIS run — same code)

| Metric | Value |
|---|---|
| Rows scored | 261,982 · holdout 2,232 customers |
| Test | PR-AUC 0.110 · ROC 0.716 · **ECE 0.0037** (calibration holds) |
| Unseen vs known | recall@2% **32.6% vs 13.6%** · PR-AUC 0.132 vs 0.103 |
| FP first-time (real) | **0.0%** customers · 0.0% merchants · 0.19% any |
| Why modest ROC? | identity-free mapping on purpose (card1→cust, addr1→merch proxy) — we threw away the winning V-columns to prove architecture transfer |

## TECH STACK (why)

- **LightGBM** — won head-to-head; exact TreeSHAP (`pred_contrib`) → free explainability; CPU, ms latency, class-weight via `scale_pos_weight=sqrt(neg/pos)`.
- **No deep learning** — tabular, no structure to exploit; costs explainability + GPU; loses the comparison we ran.
- **IsolationForest + ECDF** — labels-free second opinion; `novelty = 1 − ECDF(deviation)` = "more atypical than p% of legit traffic"; one interface for LOF/OCSVM/autoencoder.
- **FastAPI + Streamlit + plotly** — typed API w/ auto docs; live dashboard; same `StreamingProcessor` everywhere → **zero train/serve skew**.
- **YAML config + seed 42** — nothing operational hardcoded; fully reproducible.

## THE 2x2 (draw it)

```
                novelty low         novelty high
  p low         NORMAL → allow      MONITOR → log
  p high        REVIEW → alert      HIGH-RISK → alert
```

## TOP 6 LIKELY QUESTIONS

1. **Why not deep learning?** → Ran the comparison; LightGBM won (0.8865). Need exact SHAP + CPU latency. DL adds cost, no lift.
2. **Novelty AUROC is 0.14 — why keep it?** → Novelty isn't a fraud classifier; it's the *monitor* channel + tie-breaker (`rank = 0.7p + 0.3nov`). Its win is FP control: IEEE first-time FP = 0%. Judge on operational metrics (recall@2%, FP rate), not AUROC-as-fraudranker.
3. **Leakage?** → 6 guards: time split no shuffle · point-in-time (`prepare`→`commit` order) · train-only priors · unseen-customer holdout · no raw ID features · **parity check 100%/99.62%** proving as-of correctness.
4. **Synthetic data?** → Magnitude is illustrative; method is the product. Same pipeline ran on real IEEE-CIS; structure holds (unseen ≥ known, FP≈0, ECE 0.0037).
5. **Thresholds?** → Validation quantiles that *engineer the budget* (p_review = top-2% quantile). Slider re-derives; nothing retrained. Production = fixed file `models/thresholds.json`.
6. **memory/scaling?** → Welford running stats (O(1)), sorted timestamp lists + `bisect`, deques for budget. Batch scoring <1 ms/row. Key-shard per customer for parallelism.

## TWO DEMO ROWS (from `reports/artifacts/demo_examples.json`)

- **Normal txn** TXN00112216 → p=0.0, band **normal**, history conf 0.95, velocity low. Allowed.
- **Suspicious** TXN00101785 → p=0.9697, band **review**, ALERT. Reasons: velocity elevated (2 in 30min) · amount 0.5σ above baseline · history conf 0.29 · device 14× known · 21:00.

## DEMO FLOW (5 min, hooks the script)

1. CLI scorer on held-out file (streaming, known-vs-unseen split, 5 formats).
2. FastAPI — single + batch (watch velocity climb across batch).
3. Overview: the 2x2 + "novelty never alerts alone".
4. Upload & Analyse: budget slider, per-txn cards, suspected-fraud list, insights.
5. Exports: csv / json / md / txt / pdf.
6. Fraud burst → close with IEEE-CIS transfer + "responds to behavior, not identity".

## HONESTY RULES (they WILL probe)

- "Contributing factors," never "reasons it's fraud."
- Synthetic numbers are synthetic; IEEE numbers are real-data but lower — say so.
- Quote numbers visible on screen; if a judge asks something you don't have scoped on stage — "α on validation, β on test" — say "I'll pull that raw, but the honest summary is…"