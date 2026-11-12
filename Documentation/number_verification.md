# Number verification — every figure used in §4 and §5

Checked 7 October 2026 against the three saved run outputs on James's machine:
`reports/paper_evidence.md`, `reports/ablations.md`, `reports/sensitivity.md`.

Rule applied: a number enters the manuscript only if it appears in one of those
three files. Nothing is carried over from the README, the notebooks, the
`config.py` docstrings or the deploy-bundle artefact.

## Used

| Value | Source file | Section |
|---|---|---|
| 99,441 raw orders | paper_evidence | §4.1 |
| 96,171 modelling rows, 89 columns | paper_evidence | §4.2 |
| 96,470 pre-censoring, 299 dropped | collect_paper_evidence run log | §4.2 |
| Late rate 8.14% (8.1376%) | paper_evidence | §4.2 |
| 2016-09-15 to 2018-08-29 | paper_evidence | §4.1 |
| Train 70,883 / test 25,288 | paper_evidence | §4.4 |
| Train late 8.81%, test late 6.25% | paper_evidence | §4.4 |
| 63 post-dispatch / 62 pre-dispatch features | paper_evidence | §4.3 |
| Calibration slice 14,177 orders, 2,011 positives | ablations | §4.5 |
| Base fit 56,706 orders | ablations | §4.5 |
| AUC 0.8389, AP 0.3482, Brier 0.0514 | paper_evidence + ablations (agree) | §5.1 |
| Threshold 0.1309, recall 0.7481, precision 0.1723, flag rate 27.1%, cost 1.29/order (holdout-chosen) | both (agree) | §5.1 |
| Oracle threshold 0.2074, recall 0.5608, precision 0.2585 (test-chosen upper bound, Table 3 note only) | both (agree) | §5.1 |
| Pre-dispatch threshold 0.1487, recall 0.5367, precision 0.1354 | paper_evidence | §5.1 |
| No-skill AP 0.0625, Brier 0.0586 | paper_evidence | §5.1 |
| Pre-dispatch AUC 0.7413, AP 0.1549 | paper_evidence | §5.1 |
| Brier 0.3085 → 0.0514, 83.3% | **ablations only** | §5.2 |
| AUC/AP change under calibration: 0.000000 | ablations | §5.2 |
| Threshold 0.7247 → 0.1309 | ablations | §5.2 |
| −1.7291 stars, CI [−1.7680, −1.6925] | paper_evidence | §5.3 |
| n = 95,527 (7,661 / 87,866); means 2.5651 / 4.2942 | paper_evidence | §5.3 |
| U = 150,189,671.5; rank-biserial 0.5538 | paper_evidence | §5.3 |
| Permutation groups 0.520 / 0.118 / 0.107 / 0.086 | ablations | §5.4 |
| Permutation top features and values | ablations | §5.4 |
| SHAP top features and values | paper_evidence | §5.4 |
| Calendar ablation −0.0190 / −0.0229 AUC | ablations | §5.5 |
| Recall 0.7481 → 0.7051 without purchase_year | ablations | §5.5 |
| Cost parameters 25 / 4 / 0.42 / 0.46 | paper_evidence | §4.6 |
| Exchange rates ₦72.63 / ₦95.86 / ₦83.86 | IMF IFS, cited | §4.1 |
| 159 tests | verified by collection | §4.7 |

## Deliberately NOT used

| Value | Why |
|---|---|
| AP 0.332 | Run gives 0.3482. The 0.332 is a `config.py` docstring figure for a regularised model under *random* CV folds. |
| Brier 0.28 → 0.06 | Run gives 0.3085 → 0.0514. |
| Brier 0.2265 (uncalibrated) | Appears in `paper_evidence` but is **not comparable** — that baseline was fitted on the full training period while the calibrated model was fitted on 80% of it. Only the `ablations` pair (0.3085 → 0.0514) holds the classifier constant. Citing 0.2265 as the pre-calibration figure would reintroduce the confound. |
| 0.266 vs 0.051 | A logistic-regression-versus-XGBoost comparison in `README.md`, not a calibration effect. |
| −1.73 stars | Superseded by the run's −1.7291. |
| 96,470 as the headline count | It is the pre-censoring count; reported as such, not as the sample size. |
| AUC 0.862 (XGBoost) | A README figure with no run behind it. The verified XGBoost figures are from an **untuned** fit and are reported as such or omitted. |
| ₦99.51/BRL | Implies ~₦363/USD (NAFEX window). No citable 2018 annual average found. **REQUIRES SOURCE VERIFICATION.** |

## Internal consistency checks performed

- `paper_evidence` and `ablations` agree to four decimals on AUC (0.8389), AP
  (0.3482), calibrated Brier (0.0514), threshold (0.1309), recall (0.7481) and
  precision (0.1723). Two independent runs, same values.
- Sigmoid calibration left AUC and AP unchanged to six decimal places, as a
  strictly monotone transform must. This is a correctness check on the
  implementation, and it passed.
- No-skill average precision (0.0625) equals the test positive rate (6.2480%),
  as it must by definition.
- Censoring arithmetic: 96,470 − 299 = 96,171. Late rate moved 8.11% → 8.14%,
  consistent with dropping orders that could not yet be observed late.

## Flagged for the manuscript

**Calibration-in-the-large.** `sensitivity.md` reports a mean predicted risk of
0.1126 across 500 random test orders, against an observed test late rate of
6.25%. The model over-predicts in the mean by roughly 1.8×, consistent with the
sigmoid having been fitted on Feb–Apr 2018 while the base rate continued to fall
through the test period. Reported in §5.2 as an observation with its numbers; a
full reliability analysis has not been run and is named as outstanding rather
than implied.

**XGBoost.** The verified XGBoost figures come from a default-hyperparameter fit
(`--tune` was not passed). They are not evidence about XGBoost's capability and
are reported with that caveat or left out.

**Mann-Whitney p.** The run printed `p = 0`, which is double-precision underflow
rather than an exact zero. Reported as *p* < .001 per APA 7, with the test
statistic given.

## Revision, 11 November 2026 — threshold chosen out of sample

The 7 October figures selected the cost-optimal threshold on the **test labels**
(`evaluate()` minimised cost on the evaluation set it then scored). AUC, AP and
Brier are unaffected, but the operating point — threshold, recall, precision, and
every recall in the calendar ablation — was an upper bound, not an estimate.

`evaluate()` now takes a threshold chosen by `holdout_threshold()` on the final 20%
of the training period (the calibration slice), and reports the test-optimal value
separately as `oracle_*`. All three reports were regenerated from `data/raw/` with
`--rebuild`. Everything not depending on the threshold reproduced to four decimals.

| Figure | Old (test-chosen) | New (holdout-chosen) |
|---|---|---|
| Post-dispatch threshold | 0.2074 | 0.1309 |
| Post-dispatch recall / precision | 0.5608 / 0.2585 | 0.7481 / 0.1723 |
| Pre-dispatch threshold | 0.2572 | 0.1487 |
| Pre-dispatch recall / precision | 0.2424 / 0.1990 | 0.5367 / 0.1354 |
| Uncalibrated → calibrated threshold | 0.8247 → 0.2074 | 0.7247 → 0.1309 |
| Recall without purchase_year | 0.5608 → 0.4329 | 0.7481 → 0.7051 |
| Recall without all calendar fields | 0.5006 | 0.6778 |

The holdout threshold is lower than the oracle because the calibration slice
(Feb–Apr 2018, 14.2% late) has a higher late rate than the test period (6.25%) —
the same drift behind the 1.8× over-prediction noted above. The model therefore
flags more orders than is cost-optimal in hindsight: 27.1% of orders, at 1.29
per order against the oracle's lower cost. **The manuscript (`BLL-Paper-revised`)
still carries the old operating-point figures and must be updated.**
