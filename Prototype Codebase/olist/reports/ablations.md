# Ablations

Generated 2026-11-11T19:29:46+00:00 · seed 42 · logistic_regression / post_dispatch


## A. Calibration, isolated

Same fitted classifier throughout. Fitted on 56,706 orders (2016-09-15 to 2018-02-26); the sigmoid mapping is learned on 14,177 later orders (2018-02-26 to 2018-04-29, 2,011 positives). Evaluated on the same held-out test set.

| | AUC | Avg precision | Brier | Threshold | Recall | Precision | Oracle thr. | Oracle recall | Oracle precision |
|---|---|---|---|---|---|---|---|---|---|
| uncalibrated | 0.8389 | 0.3482 | 0.3085 | 0.7247 | 0.7481 | 0.1723 | 0.8247 | 0.5608 | 0.2585 |
| calibrated | 0.8389 | 0.3482 | 0.0514 | 0.1309 | 0.7481 | 0.1723 | 0.2074 | 0.5608 | 0.2585 |

Thresholds are chosen on the calibration slice, before the test labels are seen. Oracle columns choose them on test: an upper bound.


- Brier: **0.3085 → 0.0514** (83.3% reduction)
- AUC moved by 0.000000, average precision by 0.000000
- Sigmoid scaling is strictly monotone, so ranking metrics should be unchanged. If they are, this is the clean calibration result the paper should report.


## B. Calendar features

| Feature set | Features | AUC | Avg precision | Brier | Recall |
|---|---|---|---|---|---|
| all features | 63 | 0.8389 | 0.3482 | 0.0514 | 0.7481 |
| without purchase_year | 62 | 0.8199 | 0.3026 | 0.0525 | 0.7051 |
| without all calendar fields | 60 | 0.8159 | 0.2982 | 0.0538 | 0.6778 |

- **without purchase_year**: AUC -0.0190 against the full set.

- **without all calendar fields**: AUC -0.0229 against the full set.

### Do the calendar values transfer?

| Feature | Test values unseen in training | Test rows affected |
|---|---|---|
| `purchase_year` | none | 0 (0.0%) |
| `purchase_quarter` | none | 0 (0.0%) |
| `purchase_month` | none | 0 (0.0%) |

A feature whose test values never occur in training cannot carry a learned effect into the test period, whatever attribution it receives.


## C. Permutation importance (held-out test set)

Drop in ROC AUC when a single feature is shuffled. Model-agnostic, measured where it matters.

| Rank | Feature | Mean AUC drop | SD |
|---|---|---|---|
| 1 | `working_days_in_window` | 0.21861 | 0.00581 |
| 2 | `estimated_delivery_days` | 0.14131 | 0.00529 |
| 3 | `carrier_pickup_days` | 0.13144 | 0.00473 |
| 4 | `mean_item_price` | 0.04688 | 0.00293 |
| 5 | `customer_state` | 0.04306 | 0.00321 |
| 6 | `max_item_price` | 0.03385 | 0.00344 |
| 7 | `days_from_black_friday` | 0.03380 | 0.00372 |
| 8 | `purchase_month` | 0.02902 | 0.00204 |
| 9 | `purchase_quarter` | 0.02248 | 0.00169 |
| 10 | `seller_state` | 0.02050 | 0.00458 |
| 11 | `n_sellers` | 0.01913 | 0.00205 |
| 12 | `customer_lat` | 0.01707 | 0.00267 |
| 13 | `shipping_limit_slack_days` | 0.01704 | 0.00268 |
| 14 | `non_working_days_in_window` | 0.01172 | 0.00108 |
| 15 | `seller_lat` | 0.01157 | 0.00153 |
| 16 | `is_same_state` | 0.00996 | 0.00213 |
| 17 | `is_multi_seller` | 0.00956 | 0.00089 |
| 18 | `seller_customer_distance_km` | 0.00750 | 0.00151 |
| 19 | `customer_state_late_rate_to_date` | 0.00714 | 0.00029 |
| 20 | `total_payment_value` | 0.00585 | 0.00224 |

### Grouped

| Group | Total AUC drop |
|---|---|
| dispatch & promise | 0.52012 |
| routing / geography | 0.11815 |
| basket / price | 0.10718 |
| calendar / seasonal | 0.08634 |
| other / unmapped | 0.02157 |

Unmapped: `approval_time_hours`, `category_history_confidence`, `category_late_rate_to_date`, `customer_state_history_confidence`, `has_listing_metadata`, `is_multi_seller`, `is_same_city`, `is_weekend_purchase`, `max_installments`, `max_volume_cm3`, `max_weight_g`, `mean_density`, `mean_description_length`, `mean_photos_qty`, `n_payment_methods`

The dispatch-versus-routing claim turns on these two totals, not on any single feature.
