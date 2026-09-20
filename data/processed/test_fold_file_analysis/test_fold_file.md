# Fraud Risk Analysis

- **File**: `test_fold_file.csv`
- **Rows scored**: 2,000
- **Analysed at**: 2026-09-20T04:59:58.309962+00:00

## Schema mapping used

- `transaction_id` <- `transaction_id`
- `timestamp` <- `timestamp`
- `customer_id` <- `customer_id`
- `merchant_id` <- `merchant_id`
- `device_id` <- `device_id`
- `amount` <- `amount`
- `transaction_hour` <- `transaction_hour`
- `is_fraud` <- `is_fraud`
- Unmapped optional columns: is_night, new_customer, new_merchant, new_device, velocity_8min, amount_spike, customer_history_count, merchant_history_count, device_history_count

## Risk-band distribution

| risk band | transactions |
|---|---|
| normal | 1,917 |
| monitor | 31 |
| review | 16 |
| high-risk | 36 |

## Alert summary

- Alerts fired: **52** (2.60% of rows)
- High-risk: 36  |  Review: 16  |  Monitor (logged): 31

## Performance vs the fraud label (if provided)

| metric | value |
|---|---|
| fraud_rows_in_file | 28 |
| fraud_rate_pct | 1.4 |
| alert_precision | 0.3846 |
| alert_recall | 0.7143 |
| pr_auc | 0.6318 |
| roc_auc | 0.9796 |

## Top alerted transactions (by risk priority)

| transaction id | timestamp | amount | fraud probability | novelty score | risk band | rank score |
|---|---|---|---|---|---|---|
| TXN00096711 | 2025-11-11 13:33:34+00:00 | 682.01 | 1.0 | 0.4036 | high-risk | 0.821086 |
| TXN00096724 | 2025-11-11 13:54:50+00:00 | 212.93 | 1.0 | 0.2311 | high-risk | 0.769342 |
| TXN00097061 | 2025-11-12 05:56:17+00:00 | 854.69 | 1.0 | 0.1525 | high-risk | 0.745752 |
| TXN00096704 | 2025-11-11 13:17:08+00:00 | 735.3 | 1.0 | 0.1256 | high-risk | 0.737667 |
| TXN00095864 | 2025-11-09 17:30:09+00:00 | 2174.39 | 1.0 | 0.0796 | high-risk | 0.72387 |
| TXN00097048 | 2025-11-12 05:41:26+00:00 | 3636.82 | 1.0 | 0.0758 | high-risk | 0.722736 |
| TXN00097134 | 2025-11-12 09:55:35+00:00 | 796.75 | 0.9697 | 0.1388 | review | 0.720421 |
| TXN00097487 | 2025-11-13 02:17:27+00:00 | 802.09 | 0.9697 | 0.1371 | review | 0.719924 |
| TXN00096719 | 2025-11-11 13:46:27+00:00 | 1143.78 | 1.0 | 0.0542 | high-risk | 0.716271 |
| TXN00096427 | 2025-11-10 23:36:11+00:00 | 3960.26 | 1.0 | 0.0463 | high-risk | 0.713899 |

## Key insights

- 52 of 2,000 transactions (2.6%) fall into the review or high-risk bands and would alert under the current budget.
- 931 (46.5%) transactions are from customers never seen in training — scored from priors + novelty, no memorised history.
- PR-AUC 0.632 on this file (71% recall at the 2% budget, alert precision 38%).

## Thresholds used

- p_review: 0.7
- p_high: 0.7917
- novelty_high: 0.9
- p_floor_for_novelty: 0.02