# Data Validation Report

Generated: 2026-09-19T20:52:42.965591+00:00

## Overview

- Total transactions: **300,000**
- Total columns: **17**
- Fraud cases: **10,099** (3.37%)

## Data types

| Column | dtype |
|---|---|
| transaction_id | int64 |
| timestamp | datetime64[ns, UTC] |
| amount | float64 |
| customer_id | object |
| merchant_id | object |
| device_id | object |
| transaction_hour | int64 |
| is_fraud | int64 |
| ProductCD | object |
| card2 | float64 |
| card5 | float64 |
| addr2 | float64 |
| dist1 | float64 |
| P_emaildomain | object |
| R_emaildomain | object |
| ts_sec | int64 |
| is_night | int64 |

## Missing value report

| Column | Missing rows |
|---|---|
| card2 | 4,753 |
| card5 | 1,608 |
| addr2 | 33,408 |
| dist1 | 186,763 |
| P_emaildomain | 47,208 |
| R_emaildomain | 216,931 |

## Duplicates

- Duplicate `transaction_id` rows in the file: **0**
- Duplicates dropped at load time: **0**

## Class distribution

| Class | Count |
|---|---|
| 0 | 289,901 |
| 1 | 10,099 |

- Fraud percentage: **3.3663%**

## Unique entities

- Unique customers: **11,162**
- Unique merchants: **300**
- Unique devices: **1,417**

## Timestamp range

- Min: `2017-11-01 00:00:00+00:00`
- Max: `2018-01-24 19:43:51+00:00`
- Span: **84.8 days**

## Numerical feature summary

### `amount`

| stat | value |
|---|---|
| mean | 132.3861 |
| std | 236.4306 |
| min | 0.292 |
| p25 | 42.442 |
| median | 68.5 |
| p75 | 125.0 |
| max | 31937.391 |

### `transaction_hour`

| stat | value |
|---|---|
| mean | 13.804 |
| std | 7.7031 |
| min | 0.0 |
| p25 | 5.0 |
| median | 16.0 |
| p75 | 20.0 |
| max | 23.0 |

## Categorical feature summary

### `is_night`

| value | count |
|---|---|
| 0 | 204,041 |
| 1 | 95,959 |

### `is_fraud`

| value | count |
|---|---|
| 0 | 289,901 |
| 1 | 10,099 |
