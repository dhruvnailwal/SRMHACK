# Evaluation Report

Run info: seed=42  primary model=lightgbm  calibration=isotonic

## Classification (test fold)

| metric | value |
|---|---|
| pr_auc | 0.1097 |
| roc_auc | 0.7156 |
| precision | 0.1516 |
| recall | 0.0657 |
| f1 | 0.0916 |
| brier | 0.0358 |
| ece | 0.00371 |

## Reliability

- Brier score: **0.0358**
- Expected calibration error (ECE): **0.00371**

## Alert operations

| metric | value |
|---|---|
| total_transactions | 60000 |
| alerts_generated | 1009 |
| alert_rate_pct | 1.6817 |
| high_risk_alerts | 53 |
| review_alerts | 956 |
| monitor_logged | 4317 |
| alert_precision | 0.1516 |
| alert_recall | 0.0657 |
| alerts_per_caught_fraud | 6.59 |

## Cohorts

| cohort | n_rows | n_fraud | fraud_rate | PR-AUC | ROC-AUC | recall@2% | precision@2% | novelty_AUROC | FNR@0.9 |
|---|---|---|---|---|---|---|---|---|---|
| all | 60000 | 2330 | 0.039 | 0.1097 | 0.7156 | 0.1275 | 0.2475 | 0.4041 | 0.0849 |
| known_customers | 46308 | 1716 | 0.037 | 0.1027 | 0.7187 | 0.1364 | 0.195 | 0.4 | 0.0825 |
| unseen_customers | 13692 | 614 | 0.045 | 0.1324 | 0.7056 | 0.3257 | 0.1667 | 0.4141 | 0.0932 |
| first_time_customers | 963 | 49 | 0.051 | 0.1273 | 0.6742 | 1.0 | 0.0509 | 0.42 | 0.0 |
| new_devices | 152 | 10 | 0.066 | 0.0981 | 0.643 | 1.0 | 0.0658 | 0.5585 | 0.0 |
| new_merchants | 17 | 0 | 0.000 | None | None | 0.0 | 0.0 | None | 0.0 |

## First-time false positives

- first_time_customer: legit rows=914, FP alerts=0, FP rate=0.0
- first_time_device: legit rows=142, FP alerts=2, FP rate=0.0141
- first_time_merchant: legit rows=17, FP alerts=0, FP rate=0.0
- first_time_any: legit rows=1,062, FP alerts=2, FP rate=0.0019

## Leakage & data-quality checks

- **temporal_integrity**: time-based splits, no shuffle
- **unseen_customer_holdout**: 2,232 customers removed from train only
- **priors_fit_on**: train
- **features_computed_as_of**: point-in-time, commit after scoring
- **raw_ids_in_features**: False
- **velocity_8min_match_rate**: nan
- **amount_spike_match_rate**: nan

## Feature availability (cold-start rate)

- customer_history_available_rate: 0.9565
- first_time_customer_rate: 0.016
- new_device_rate: 0.0025
- new_merchant_rate: 0.0003