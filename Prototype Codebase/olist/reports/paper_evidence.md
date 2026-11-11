# Empirical evidence for the BLL paper

Generated 2026-11-11T19:25:40+00:00 · seed 42 · Python 3.12.3

Every number below came from this run. Nothing is transcribed.


## Dataset

| Quantity | Value |
|---|---|
| Raw order rows | 99441 |
| **Modelling rows** | **96,171** |
| Columns | 89 |
| Late-delivery rate | 8.1376% |
| Date range | 2016-09-15 to 2018-08-29 |
| Censoring buffer | 30 days |
| Temporal split at | 2018-04-30 |
| Train | 70,883 rows (2016-09-15 to 2018-04-29), late 8.8117% |
| Test | 25,288 rows (2018-04-30 to 2018-08-29), late 6.2480% |

## Model performance (held-out temporal test set)

| Model | AUC | Avg precision | Brier | Threshold | Recall | Precision | Oracle thr. | Oracle recall | Oracle precision |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression / post_dispatch — uncalibrated | 0.8208 | 0.2946 | 0.2265 | 0.6003 | 0.7462 | 0.1623 | 0.7571 | 0.4823 | 0.2554 |
| logistic_regression / post_dispatch — **calibrated** | 0.8389 | 0.3482 | 0.0514 | 0.1309 | 0.7481 | 0.1723 | 0.2074 | 0.5608 | 0.2585 |
| xgboost / post_dispatch — uncalibrated | 0.7548 | 0.2890 | 0.0611 | 0.6212 | 0.1639 | 0.5385 | 0.4108 | 0.2956 | 0.3011 |
| xgboost / post_dispatch — **calibrated** | 0.7104 | 0.2322 | 0.0545 | 0.1202 | 0.2804 | 0.2286 | 0.1383 | 0.2557 | 0.2615 |
| logistic_regression / pre_dispatch — uncalibrated | 0.7274 | 0.1393 | 0.2554 | 0.6228 | 0.5424 | 0.1263 | 0.8392 | 0.1253 | 0.1992 |
| logistic_regression / pre_dispatch — **calibrated** | 0.7413 | 0.1549 | 0.0602 | 0.1487 | 0.5367 | 0.1354 | 0.2572 | 0.2424 | 0.1990 |
| xgboost / pre_dispatch — uncalibrated | 0.6459 | 0.1004 | 0.0778 | 0.6309 | 0.0234 | 0.1209 | 0.5628 | 0.0551 | 0.1460 |
| xgboost / pre_dispatch — **calibrated** | 0.5816 | 0.0932 | 0.0586 | 0.1387 | 0.1361 | 0.1338 | 0.1982 | 0.0614 | 0.1773 |

*Threshold* is the cost-optimal cutoff chosen on the last 20% of the training period (the calibration slice), before the test labels are seen; recall and precision are measured on test at that cutoff. *Oracle* columns pick the cutoff on the test labels themselves: an upper bound, not an estimate of deployed performance. The uncalibrated baselines were fitted on the whole training period, so the holdout slice is in-sample for them and their threshold is optimistic.


### Calibration effect

| Model | Brier before | Brier after | Improvement |
|---|---|---|---|
| logistic_regression / post_dispatch | 0.2265 | 0.0514 | 0.1751 |
| xgboost / post_dispatch | 0.0611 | 0.0545 | 0.0066 |
| logistic_regression / pre_dispatch | 0.2554 | 0.0602 | 0.1952 |
| xgboost / pre_dispatch | 0.0778 | 0.0586 | 0.0191 |

Sigmoid calibration is monotone, so AUC and average precision are preserved; only the Brier score moves.


## Baselines

- Positive rate in test: **6.2480%**
- No-skill AUC: 0.5 · no-skill average precision: **0.0625** · Brier 0.0586
- Always-predict-late: recall 1.0, precision 0.0625, flags 100% of orders


## Review score and lateness

- Estimator: difference in group means (late minus on-time), no controls
- n = 95,527 reviewed orders (7,661 late, 87,866 on time)
- Mean review score: late 2.5651, on time 4.2942
- **Mean difference: -1.7291 stars** (95% bootstrap CI -1.7680 to -1.6925, 2000 resamples)
- Mann-Whitney U = 150189671.5, p = 0
- Rank-biserial correlation: 0.5538
- Association, not a causal estimate. No covariates, no controls.


## SHAP — global driver ranking

Mean |SHAP| over 300 orders, background 200 rows.

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `purchase_month` | 1.1061 |
| 2 | `purchase_quarter` | 0.8952 |
| 3 | `purchase_year` | 0.7151 |
| 4 | `working_days_in_window` | 0.7010 |
| 5 | `shipping_limit_slack_days` | 0.5568 |
| 6 | `customer_state` | 0.5389 |
| 7 | `carrier_pickup_days` | 0.5351 |
| 8 | `estimated_delivery_days` | 0.5170 |
| 9 | `is_same_state` | 0.3970 |
| 10 | `mean_item_price` | 0.3487 |
| 11 | `max_item_price` | 0.2860 |
| 12 | `customer_lat` | 0.2670 |
| 13 | `seller_customer_distance_km` | 0.2625 |
| 14 | `seller_state` | 0.2513 |
| 15 | `days_from_black_friday` | 0.2369 |
| 16 | `holidays_in_window` | 0.1603 |
| 17 | `customer_city` | 0.1532 |
| 18 | `is_black_friday_week` | 0.1472 |
| 19 | `n_sellers` | 0.1390 |
| 20 | `purchase_dow` | 0.1330 |

SHAP runs on the inner uncalibrated classifier, so these rank drivers but do not additively decompose the calibrated probability.


## Cost parameters (assumptions, not measurements)

| Parameter | Value |
|---|---|
| `cost_fn` | 25.0 |
| `cost_fp` | 4.0 |
| `refund_share` | 0.42 |
| `churn_share` | 0.46 |
| `support_cost` | 12.0 |
| `intervention_cost` | 4.0 |
| `currency` | BRL |

These set the decision threshold and every monetary figure. They are stipulated, not observed.
