# Sensitivity of the recommendation to its cost assumptions

Generated 2026-10-06T22:16:13+00:00 · logistic_regression / post_dispatch · 500 held-out test orders · seed 42

Defaults: refund_share 0.42, churn_share 0.46, support_cost ₦1,200, intervention_cost ₦400.


Monetary values are Olist order values in Brazilian reais, converted per order at its purchase year's official annual average rate (effective ₦83.86/BRL). 2016: ₦72.63/BRL; 2017: ₦95.86/BRL; 2018: ₦83.86/BRL. Source: International Monetary Fund, International Financial Statistics: official exchange rate (LCU per US$, period average).


Mean order value ₦10,658 (R$127.10). Converted values express Brazilian transactions on a naira scale; they are not Nigerian transactions.


At those defaults: mean risk 0.1126, mean saving ₦250 per order. Recommendation mix: Ship as-is 360, Extend the promise 77, Expedite dispatch 63.


## One-way sweeps

`changed` is the share of orders whose recommendation differs from the one given at the default values.


### `refund_share`

| Value | Changed | Intervene | Mean saving |
|---|---|---|---|
| 0 | 12.6% | 15.4% | ₦105 |
| 0.05 | 11.2% | 16.8% | ₦120 |
| 0.1 | 10.0% | 18.0% | ₦136 |
| 0.15 | 8.6% | 19.4% | ₦153 |
| 0.2 | 6.0% | 22.0% | ₦170 |
| 0.25 | 4.2% | 23.8% | ₦187 |
| 0.3 | 3.0% | 25.0% | ₦205 |
| 0.35 | 1.4% | 26.6% | ₦224 |
| 0.4 | 0.4% | 27.6% | ₦242 |
| 0.45 | 0.2% | 28.2% | ₦261 |
| 0.5 | 1.6% | 29.6% | ₦280 |
| 0.55 | 3.0% | 31.0% | ₦299 |
| 0.6 | 3.6% | 31.6% | ₦318 |
| 0.65 | 4.4% | 32.4% | ₦338 |
| 0.7 | 5.0% | 33.0% | ₦357 |
| 0.75 | 6.0% | 34.0% | ₦377 |
| 0.8 | 7.6% | 35.6% | ₦397 |
| 0.85 | 9.0% | 37.0% | ₦417 |
| 0.9 | 10.2% | 38.2% | ₦437 |
| 0.95 | 10.8% | 38.8% | ₦457 |
| 1 | 11.2% | 39.2% | ₦478 |

### `churn_share`

| Value | Changed | Intervene | Mean saving |
|---|---|---|---|
| 0 | 14.2% | 13.8% | ₦92 |
| 0.05 | 12.0% | 16.0% | ₦108 |
| 0.1 | 10.8% | 17.2% | ₦123 |
| 0.15 | 9.6% | 18.4% | ₦140 |
| 0.2 | 8.4% | 19.6% | ₦156 |
| 0.25 | 5.4% | 22.6% | ₦173 |
| 0.3 | 4.0% | 24.0% | ₦191 |
| 0.35 | 2.4% | 25.6% | ₦209 |
| 0.4 | 1.0% | 27.0% | ₦227 |
| 0.45 | 0.2% | 27.8% | ₦246 |
| 0.5 | 0.4% | 28.4% | ₦265 |
| 0.55 | 2.2% | 30.2% | ₦283 |
| 0.6 | 3.2% | 31.2% | ₦303 |
| 0.65 | 4.0% | 32.0% | ₦322 |
| 0.7 | 4.6% | 32.6% | ₦341 |
| 0.75 | 5.4% | 33.4% | ₦361 |
| 0.8 | 6.0% | 34.0% | ₦381 |
| 0.85 | 8.0% | 36.0% | ₦401 |
| 0.9 | 9.4% | 37.4% | ₦421 |
| 0.95 | 10.6% | 38.6% | ₦441 |
| 1 | 10.8% | 38.8% | ₦461 |

### `support_cost`

| Value | Changed | Intervene | Mean saving |
|---|---|---|---|
| 0 | 3.8% | 24.2% | ₦218 |
| 500 | 2.0% | 26.0% | ₦231 |
| 1000 | 0.4% | 27.6% | ₦244 |
| 1500 | 0.4% | 28.4% | ₦258 |
| 2000 | 2.0% | 30.0% | ₦272 |
| 2500 | 4.0% | 32.0% | ₦286 |
| 3000 | 5.2% | 33.2% | ₦301 |
| 3500 | 7.0% | 35.0% | ₦316 |
| 4000 | 8.2% | 36.2% | ₦331 |
| 4500 | 10.6% | 38.6% | ₦347 |
| 5000 | 11.6% | 39.6% | ₦364 |
| 5500 | 12.6% | 40.6% | ₦380 |
| 6000 | 14.6% | 42.6% | ₦397 |

### `intervention_cost`

| Value | Changed | Intervene | Mean saving |
|---|---|---|---|
| 0 | 72.0% | 100.0% | ₦460 |
| 200 | 19.0% | 47.0% | ₦321 |
| 400 | 0.0% | 28.0% | ₦250 |
| 600 | 10.4% | 17.6% | ₦204 |
| 800 | 15.6% | 12.4% | ₦175 |
| 1000 | 18.2% | 9.8% | ₦152 |
| 1200 | 21.2% | 6.8% | ₦136 |
| 1400 | 22.8% | 5.2% | ₦123 |
| 1600 | 23.8% | 4.2% | ₦114 |
| 1800 | 24.6% | 3.4% | ₦107 |
| 2000 | 25.0% | 3.0% | ₦100 |
| 2200 | 25.2% | 2.8% | ₦94 |
| 2400 | 25.2% | 2.8% | ₦88 |
| 2600 | 25.4% | 2.6% | ₦83 |
| 2800 | 25.4% | 2.6% | ₦78 |
| 3000 | 25.4% | 2.6% | ₦73 |
| 3200 | 25.6% | 2.4% | ₦68 |
| 3400 | 25.8% | 2.2% | ₦63 |
| 3600 | 26.0% | 2.0% | ₦59 |
| 3800 | 26.0% | 2.0% | ₦55 |
| 4000 | 26.2% | 1.8% | ₦51 |

## Break-even action cost

The action cost at which intervening stops paying, per order.

| Percentile | Value |
|---|---|
| p10 | ₦29 |
| p25 | ₦69 |
| median | ₦178 |
| p75 | ₦453 |
| p90 | ₦983 |

**28.0% of orders break even above the assumed action cost of ₦400** — that is, intervening pays for them. The rest do not, and the brief should not recommend intervening on them.


## Stability

- Mean agreement with the default recommendation across the whole grid: **88.8%**
- Orders whose recommendation never changes: **1.8%**
- Orders whose recommendation changes in more than a quarter of settings: 17.6%


Read this as the honest bound on every monetary claim in the paper: the model's ranking of risk is empirical, but the decision drawn from it is a function of four stipulated numbers, and this is how much it moves when they do.
