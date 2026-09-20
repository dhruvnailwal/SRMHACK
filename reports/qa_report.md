# QA Report - Unseen-Customer Fraud Detection

Run finished in **23.1s** - **18/18 checks passed**.

| # | check | status | detail |
|---|---|---|---|
| 1 | A. app boots with no exceptions | PASS | 0 exception(s) |
| 2 | B. three sections render (Overview / Upload / System eval) | PASS | sections: overview=True upload=True eval=True |
| 3 | C1. Analyse button runs the pipeline | PASS | 0 exception(s) |
| 4 | C2. System evaluation shows a dynamic file caption | PASS | Enable **Per-row explanations** before analysing to see factor-level reasons. Enable **Per-row explanations** before ana |
| 5 | D. /health reports service state | PASS | ok |
| 6 | E1. /predict returns risk decision | PASS | http 200 in 126ms |
| 7 | E2. all four required fields present | PASS | fields=['alert', 'explanation', 'fraud_probability', 'novelty_score', 'rank_score', 'raw_probability', 'risk_band', 'tags', 'top_contributing_features', 'valid'] |
| 8 | E3. probabilities in [0,1] | PASS | p=0.017 nov=0.087 |
| 9 | E4. risk band is one of the four bands | PASS | normal |
| 10 | F1. /predict/batch returns decisions for every row | PASS | http 200, 2 decisions |
| 11 | F2. response preserves request order | PASS | p0=0.0 p1=0.9697 |
| 12 | F3. batch scores chronologically (all rows valid) | PASS |  |
| 13 | G. invalid transaction -> structured 4xx | PASS | http 422 |
| 14 | H1. identical input -> identical decision (determinism) | PASS | band=normal p=0.0168 |
| 15 | H2. streaming state accumulates history (2nd txn sees 1st) | PASS | history_count=2 |
| 16 | I. raw ids not in MODEL_FEATURES (identity-free) | PASS | raw cols leaked=[] |
| 17 | J. alert budget hard-caps escalations | PASS | alerts=3 cap=3 (2.0% of 60) |
| 18 | K. leakage report flags raw ids clean | PASS | raw_ids_in_features=False |
