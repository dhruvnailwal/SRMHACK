# Evaluation Report

Run info: seed=42  primary model=lightgbm  calibration=isotonic

## Classification (test fold)

| metric | value |
|---|---|
| pr_auc | 0.8938 |
| roc_auc | 0.9937 |
| precision | 0.7082 |
| recall | 0.4706 |
| f1 | 0.5654 |
| brier | 0.0049 |
| ece | 0.00182 |

## Reliability

- Brier score: **0.0049**
- Expected calibration error (ECE): **0.00182**

## Alert operations

| metric | value |
|---|---|
| total_transactions | 23928 |
| alerts_generated | 305 |
| alert_rate_pct | 1.2747 |
| high_risk_alerts | 76 |
| review_alerts | 229 |
| monitor_logged | 1741 |
| alert_precision | 0.7082 |
| alert_recall | 0.4706 |
| alerts_per_caught_fraud | 1.41 |

## Cohorts

| cohort | n_rows | n_fraud | fraud_rate | PR-AUC | ROC-AUC | recall@2% | precision@2% | novelty_AUROC | FNR@0.9 |
|---|---|---|---|---|---|---|---|---|---|
| all | 23928 | 459 | 0.019 | 0.8938 | 0.9937 | 0.8279 | 0.7933 | 0.1438 | 0.0745 |
| known_customers | 11729 | 0 | 0.000 | None | None | 0.0 | 0.0 | None | 0.0683 |
| unseen_customers | 12199 | 459 | 0.038 | 0.9002 | 0.9914 | 0.8301 | 0.7954 | 0.1583 | 0.0807 |
| first_time_customers | 1509 | 131 | 0.087 | 0.631 | 0.9348 | 0.9313 | 0.2547 | 0.1761 | 0.0 |
| new_devices | 455 | 7 | 0.015 | 0.5479 | 0.9828 | 1.0 | 0.0154 | 0.0907 | 0.0 |
| new_merchants | 21 | 0 | 0.000 | None | None | 0.0 | 0.0 | None | 0.0 |

## First-time false positives

- first_time_customer: legit rows=1,378, FP alerts=58, FP rate=0.0421
- first_time_device: legit rows=448, FP alerts=10, FP rate=0.0223
- first_time_merchant: legit rows=21, FP alerts=0, FP rate=0.0
- first_time_any: legit rows=1,657, FP alerts=60, FP rate=0.0362

## Leakage & data-quality checks

- **temporal_integrity**: time-based splits, no shuffle
- **unseen_customer_holdout**: 1,660 customers removed from train only
- **priors_fit_on**: train
- **features_computed_as_of**: point-in-time, commit after scoring
- **raw_ids_in_features**: False
- **velocity_8min_match_rate**: 1.0
- **amount_spike_match_rate**: 0.9962

## Feature availability (cold-start rate)

- customer_history_available_rate: 0.8167
- first_time_customer_rate: 0.0631
- new_device_rate: 0.019
- new_merchant_rate: 0.0009