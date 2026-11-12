# Methodology

## Late-Delivery Risk Prediction: An Empirical Instantiation of the Business Language Layer

**Case study author:** James Wandiya — African Institute for Mathematical Sciences (AIMS Senegal) · DataLab Technology Limited

**Prepared for:** Oyeleke, O. S., Wandiya, J., Eze, C., Charisa, O., & Agboola, E. T. (2026). *Business Language Layer (BLL): A Business-Centric Layer for Explainable Artificial Intelligence.* DataLab Technology Limited, Abuja, Nigeria.

**Corresponding author of the paper:** Oyeleke Olayemi Seun, Senior Data Scientist, DataLab Technology Limited — o.oyeleke@datalabanalyticshub.com

---

## How to read this document

This document has two audiences and is written for both.

If you are **reviewing the business claims** — the risk percentages, the naira
and real figures, the recommendations the system produces — Sections 1, 8, 9
and 10 are written in plain language and will tell you how far to trust them.

If you are **reviewing the modelling** — whether the numbers would survive
scrutiny from a referee — Sections 3 through 7 record every decision that
shaped them, including the ones that turned out to be wrong.

Technical terms are explained the first time they appear. Where a section is
unavoidably technical, it opens with a one-sentence summary of what it
establishes.

---

## 1. Why this document exists

The Business Language Layer paper demonstrates its framework through interface
figures. Section 6 of the paper is candid about what those figures are:

> The demonstration is illustrative, drawn from a single e-commerce logistics
> scenario, and the figures presented use exemplary values to convey the
> layer's behaviour rather than results from a deployed system.

In other words: the paper shows what the layer *would* look like, using
plausible numbers chosen by hand.

**This case study replaces those hand-chosen numbers with measurements from a
working system.** A real model was trained on a real dataset of roughly 100,000
e-commerce orders, and the Business Language Layer was built on top of it. Every
figure the system displays is computed, not designed.

That raises an obvious question for any reviewer: *are the measurements
trustworthy?* This document answers it by recording the decisions behind them —
so a co-author, a referee or a colleague can judge the work without reading the
source code.

It also records **what did not work**. Two features expected to help contributed
almost nothing. Two implementation choices actively degraded performance before
being caught by measurement. Both are documented here, because a methodology
that reports only its successes is not a methodology.

---

## 2. The data

**Source:** Brazilian E-Commerce Public Dataset by Olist, published on Kaggle
under a CC BY-NC-SA 4.0 licence. Nine linked tables covering roughly 100,000
orders placed between September 2016 and October 2018.

Olist is a Brazilian marketplace that connects small sellers to larger
e-commerce platforms. When a customer places an order, the seller packs it, a
carrier collects it, and it travels to the customer — often across a country
larger than the continental United States.

### 2.1 Defects in the raw data

Public datasets are rarely as clean as they appear. Each defect below was
verified against the source files, is corrected in `cleaning.py`, and has a test
that fails if the correction is ever removed.

| Table | Defect found | How it is handled |
|---|---|---|
| `order_payments` | 3 rows typed `not_defined`; 9 with zero value; 2 with zero instalments | Set to missing, flagged, or floored at 1 |
| `products` | 610 rows with no category; 4 with a weight of 0 grams; the source misspells "length" as `lenght` | Category becomes `unlisted`; impossible weights set to missing; columns renamed |
| `sellers` | `seller_city` holds 611 free-text spellings — including `sao paulo`, `são paulo`, `sao  paulo`, `sao paulo - sp` and `sao paulop`; one city is recorded as a postcode | Accent folding, compound splitting and an alias table reduce this to 579 genuine cities |
| `category_translation` | A byte-order mark corrupts the header; two categories used by `products` have no translation | Mark stripped on load; missing translations supplied |
| `geolocation` | About 1 million rows for 19,000 postcode prefixes; some coordinates fall outside Brazil | Bounded to Brazil, then collapsed to one centroid per postcode |
| `orders` | 166 orders record the carrier collecting the parcel *before* the customer placed the order | Set to missing |

**Cleaning is non-destructive.** Rows are flagged or set to missing, never
deleted. Counts therefore stay reconcilable against the raw files, and
downstream code decides what to do with each problem rather than having the
decision made silently upstream.

### 2.2 Why the city spellings matter

An earlier analysis converted `seller_city` into roughly 594 numeric codes, one
per distinct spelling. Because `sao paulo` and `são paulo` are different
strings, the same physical city was split across up to six separate codes.

Any signal the city carried was fragmented across those codes, and the model
could not recover it. This is a good illustration of a general point: a data
defect does not announce itself as an error. It quietly weakens the result.

---

## 3. What is being predicted, and at what level

**The target:** an order is *late* if it reached the customer after the delivery
date they were promised.

Formally: `is_late_delivery = (order_delivered_customer_date > order_estimated_delivery_date)`

Only delivered orders are included. Cancelled and undelivered orders have no
observable outcome, so including them would mean guessing at their labels.

**Result: 96,171 orders, of which 8.14% arrived late.**

### 3.1 One row per order, not one row per item

The source tables mix levels of detail. Payments are recorded per instalment,
items per line, reviews roughly per order. The thing being predicted — whether
the delivery was late — is a property of the **order**.

Modelling at item level gives every item in a multi-item order an identical
label and nearly identical features. When the data is then split randomly into
training and test sets, those near-duplicate rows are scattered across both
sides. The model appears to perform well because it has effectively seen the
test cases during training.

The earlier analysis reported 7.8% late across 109,362 item-level rows. At order
level the rate is **8.14% across 96,171 orders** — multi-item orders were being
counted several times, diluting the true rate.

This matters more than it sounds. Precision, average precision, and the ratio
that resampling corrects toward all depend on the base rate. Every one of them
was being computed against a figure that did not match the business question.

**Item-level detail is preserved as explicit features** rather than discarded:
`n_items`, `n_sellers`, `is_multi_seller`, price spread, maximum weight.

`n_sellers` has a real-world justification. An order involving two sellers ships
only when the *slower* one dispatches, so its exposure is the maximum of two
processes rather than the average of one.

**Enforcement:** every merge asserts that the row count is unchanged. If a join
key turns out not to be unique, the code raises an error rather than silently
duplicating rows.

---

## 4. Preventing the model from cheating

A predictive model can score well for the wrong reason: because it was given
information that would not exist at the moment a real prediction is needed. This
is called *leakage*, and it is the single most common way a promising model
turns out to be worthless in deployment.

Four categories of column are excluded, each for a different reason.

### 4.1 Information that only exists after the outcome

`actual_delivery_days`, `delivery_delay_days`, `order_delivered_customer_date`,
`review_score`.

All of these are knowable only after delivery. `delivery_delay_days` literally
defines the target.

`review_score` is subtler and worth stating: the customer writes the review
*after* the parcel arrives. A late delivery causes a poor review, not the other
way round. Using it to predict lateness would be reasoning backwards.

### 4.2 Before dispatch, or after? — two distinct products

A feature can be present in the historical data and still be unavailable at the
moment a prediction is needed.

| Scenario | Features | What is known | What can be done about it |
|---|---|---|---|
| **Pre-dispatch** | 68 | Everything known when the order is placed | Expedite dispatch, re-route, warn the customer early |
| **Post-dispatch** | 69 | Adds how long the seller took to hand the parcel to the carrier | Escalate a parcel already in transit |

`carrier_pickup_days` is the gap between purchase and the carrier collecting the
parcel. It cannot exist until the parcel has been collected. A model that uses
it therefore cannot run at checkout.

The earlier analysis correctly excluded `actual_delivery_days` on exactly these
grounds — and then kept `carrier_pickup_days` as its third most important
feature. The principle was understood; the application was inconsistent.

### 4.3 Historical rates, and a trap that looks correct

Rolling late rates per seller, route, category and marketplace are the most
leak-prone features in this project — and the leak hides inside an
implementation that looks obviously right.

The natural approach groups orders by seller, sorts them by **purchase date**,
and takes a running average of past lateness.

That leaks. An order placed on 1 June is not *resolved* on 1 June. Whether it
was late becomes known only when it is delivered, perhaps twenty days later. A
running average by purchase date therefore includes outcomes that had not yet
happened.

**The leak is also selective, which makes it worse.** Late orders are delivered
later than on-time ones. So a purchase-date running average systematically
over-represents outcomes that were still unknown — biased toward precisely the
thing being predicted.

**The rule enforced here:** an order contributes to a seller's history only once
its delivery date has passed. This is implemented as a backward as-of join on
resolution time, configured so that an order can never contribute to its own
history. Three tests pin this behaviour.

Rates are smoothed toward a running marketplace average, so that a seller with
one late order out of one does not read as a 100% risk.

### 4.4 Counters that only go up — a failure found by measurement

`seller_tenure_days`, `seller_orders_to_date`, `category_orders_to_date`,
`route_orders_to_date` and `market_late_rate_to_date` were all built, then
removed.

The problem is that they only ever increase with time. Because the data is split
by date, the test period sits entirely in the future relative to training — so
these counters are systematically larger there than anything the model has seen.

Measured: **89% of test rows exceeded the training maximum** for
`seller_tenure_days`, and 68% for `category_orders_to_date`.

Linear models extrapolate such values without bound. Tree-based models cannot
extrapolate at all. Both degraded.

The bounded equivalents — `seller_late_rate_to_date`, a rate between 0 and 1,
and `seller_history_confidence`, likewise — carry the same information on a
stable scale, and are retained.

---

## 5. How the model is validated

### 5.1 Splitting by time, not at random

- **Training:** purchases through 30 April 2018 — 70,883 orders, 8.81% late
- **Testing:** 1 May to 29 August 2018 — 25,288 orders, 6.25% late

A random split allows a model to train on December and be tested on November. It
then scores well by memorising which months ran late.

This is not hypothetical. In the earlier analysis, `purchase_quarter`,
`purchase_month` and `purchase_year` were among the top-ranked features —
roughly 58% of total feature importance sat in calendar position. The model had
learned the seasons, not the problem. Deployment never provides next month's
seasonality.

### 5.2 Accounting for orders still in transit

An order enters the dataset only once delivered. Late orders are delivered later
than on-time ones. So near the end of the data, late orders are
disproportionately still in transit and therefore absent — making the observed
late rate fall for reasons that have nothing to do with performance. This is
called *right-censoring*.

Orders promised within 30 days of the last observed delivery are removed.

**Result: 299 orders removed. The late rate moved from 8.11% to 8.14%.**

This deserves stating plainly: **the filter found almost nothing.** The gap
between the training period (8.81% late) and the test period (6.25% late) is
therefore *genuine improvement by Olist through 2018*, not an artefact of how
the data was assembled.

The filter ruled out a hypothesis rather than fixing a problem. That is still a
result, and a reviewer is entitled to know it was checked.

### 5.3 Cross-validation must also respect time

Cross-validation repeatedly splits the training data to tune model settings. The
method used here is `TimeSeriesSplit`, which always validates on later data than
it trains on.

This was changed *after* measurement showed the alternative failing:

| Model | Cross-validation score | Forward test score |
|---|---|---|
| XGBoost (post-dispatch) | 0.460 | 0.259 |
| Logistic regression (post-dispatch) | 0.399 | **0.332** |

**XGBoost won every cross-validation and lost every forward test.** Random folds
shuffle within the training period, which rewards models that memorise
seasonality. The search obligingly selected a deep, high-capacity configuration
(`max_depth=8, n_estimators=800`) — exactly the kind of model best at
memorising. On strictly future data, that memorisation is worthless.

One honest caveat: cross-validation scores still exceed test scores even for the
winning model (0.288 versus 0.135 pre-dispatch). This is **not** the same defect
as the earlier analysis's 0.98-versus-0.84 gap, which came from synthetic
resampled data leaking across folds. Here the folds are clean; the remaining gap
is seasonality that genuinely does not transfer forward.

**Where the two disagree, report the test figure.**

---

## 6. Imbalance and calibration

Only about 8% of orders are late. Left alone, a model can score 92% accuracy by
predicting "on time" for everything — and be useless.

### 6.1 Resampling, applied once

SMOTE (a technique that generates synthetic examples of the rare class) is
applied as a step *inside* the modelling pipeline. This matters: applying it to
the whole training set beforehand lets synthetic points in a validation fold be
generated from real points in the training fold. The folds stop being
independent and the score inflates.

**Imbalance is corrected once.** The earlier analysis applied SMOTE to a 1:1
ratio, `class_weight="balanced"`, *and* `scale_pos_weight` simultaneously —
correcting for the same problem three times and pushing the decision boundary
much further than intended.

XGBoost uses `scale_pos_weight` rather than SMOTE here, since resampling 70,000
rows is a large cost per fit and buys gradient boosting nothing the weight does
not already provide.

### 6.2 Calibration — the finding that matters most

The paper asserts that calibration is "load-bearing rather than optional" for
the Business Language Layer. **This case study measures that claim.**

A calibrated model is one whose probabilities mean what they say: among orders
it scores at 30%, about 30% should actually be late. An uncalibrated model can
rank orders perfectly while its numbers are badly wrong.

**Four model variants, pre-dispatch, tested forward:**

| History features | Calibrated | ROC AUC | Avg precision | Brier | Threshold | Savings |
|---|---|---|---|---|---|---|
| off | off | 0.7202 | 0.1361 | 0.2804 | 0.842 | 1,717 |
| on | off | 0.7275 | 0.1393 | 0.2547 | 0.821 | 1,783 |
| off | **on** | 0.7394 | 0.1519 | 0.0596 | 0.233 | 3,305 |
| on | **on** | **0.7414** | **0.1547** | 0.0600 | 0.255 | **3,402** |

*Brier score measures how accurate the probabilities are — lower is better.
Threshold is the cutoff at which an order is flagged for action. Savings is in
Brazilian reais, against doing nothing.*

**Calibration is the dominant effect.** The ranking is untouched — the
transformation is strictly order-preserving, so AUC barely moves — yet the Brier
score falls by roughly a factor of four and modelled savings roughly double.

**The mechanism is the decision threshold.** SMOTE trains the classifier on an
artificially balanced world. On a problem with a 6% base rate, its raw outputs
are inflated, and the cost-optimal cutoff has to climb above 0.8 to compensate.
After calibration the cutoff sits near the base rate, where it belongs.

**Why this matters specifically for the Business Language Layer.** An
uncalibrated model would display roughly 84% where the honest figure is about
23%. The layer then multiplies that number by an order value and renders it as a
confident monetary recommendation.

This is precisely the failure mode the paper's Section 6 warns about — a fluent
output more persuasive than its evidence — now demonstrated rather than
hypothesised.

**Method note:** sigmoid (Platt) scaling was used rather than isotonic
regression. Isotonic fits a step function; on this data it collapsed 3,812
distinct probabilities into 48 levels, and the resulting ties reduced average
precision even as calibration improved. The calibration sample is taken by date
rather than at random, so the correction is learned in the same direction as
deployment.

### 6.3 Seller history — a hypothesis that largely did not pay off

Seller history contributes **+0.007 AUC uncalibrated, +0.002 calibrated**. Real,
but marginal.

More was expected, given that dispatch behaviour dominates the scenario
comparison below. Two plausible explanations: most Olist sellers have few
resolved orders by the time any given order is placed, so the smoothed rate
stays close to the marketplace average; and the encoded `seller_city` and
`product_category` features may already absorb some seller-quality signal.

The features are retained — positive on every metric, and free at prediction
time — but the honest report is that the hypothesis was largely not borne out.

**Relevance to the paper:** Figure 5 lists "Slow recent dispatch" as a key
contributing factor. On this data, seller dispatch history is a weak driver. If
that figure is regenerated from real output, the driver will rarely appear
prominently. This is noted as an open question in Section 12.

---

## 7. Choosing the model, and the cutoff

**Selection is by expected business cost, not by accuracy or AUC.**

The conventional 0.5 cutoff assumes a false alarm and a missed late delivery
cost the same. They plainly do not. Missing a late delivery costs a refund, a
support contact, and the customer goodwill implied by a 1.7-star review drop.
Flagging an on-time order costs one message or one shipping upgrade.

`evaluate.choose_threshold` sweeps every distinct predicted probability and
minimises `cost_of_miss × misses + cost_of_false_alarm × false alarms`, then
reports the saving against doing nothing.

**This is not a refinement — ranking metrics can select the wrong model
outright.** In an earlier run, XGBoost won on AUC (0.862) while catching only
**29.6%** of late deliveries, the worst of four candidates, on a problem whose
entire justification was catching them.

Alongside cost, `recall_at_budget` answers the question an operations lead
actually asks: *if we can chase 500 orders a week, what share of the late ones
do we catch?*

---

## 8. Results

### 8.1 The operational finding

| Scenario | ROC AUC | Avg precision | Recall at threshold | Savings |
|---|---|---|---|---|
| Pre-dispatch | 0.741 | 0.155 | 24.2% | BRL 3,402 |
| **Post-dispatch** | **0.839** | **0.332** | **58.2%** | **BRL 11,716** |

The two scenarios differ by essentially one feature: how long the parcel took to
reach the carrier. That single column moves AUC by about 0.10 and multiplies
modelled savings roughly 3.4-fold.

**Read as a business statement rather than a modelling one: lateness at Olist is
a dispatch problem, not a routing problem.** Very little that is knowable at
checkout predicts it well. How quickly the seller hands the parcel over predicts
it strongly.

The product implication follows directly. What Olist needs is not an order-time
risk score but a **dispatch monitor** that escalates when a seller is slow to
hand off.

In the paper's terms this is a *recommend*-level output — the kind of conclusion
the framework claims to produce, and that feature attributions alone cannot
reach.

### 8.2 The deployed model

Post-dispatch logistic regression, calibrated. Decision threshold **0.204**, 63
features.

On the most recent 500 orders: mean predicted risk 10.3% against a true base
rate near 8% — evidence that calibration holds on unseen data — with 12.8% of
orders flagged, an operationally realistic alert volume.

Calibrated risk spans 0% to 99.7% across the test set, so the model does
separate confidently at both ends rather than hedging near the base rate.

### 8.3 Evidence behind the cost model

The Business Language Layer's monetary figures rest on an assumption: that late
delivery damages the customer relationship. That assumption is testable, and was
tested.

- Mean review score, **late** orders: **2.565** (n = 7,661)
- Mean review score, **on-time** orders: **4.294** (n = 87,866)
- Difference: **−1.729 stars**, 95% bootstrap confidence interval [−1.768, −1.692]
- Mann-Whitney U test: p < 1 × 10⁻³⁰⁰

**A late delivery costs 1.73 stars** — the difference between a satisfied
customer and a dissatisfied one, on a five-point scale.

This is an association rather than proven causation, but the effect is large,
the interval is tight, and the mechanism is plausible.

---

## 9. The Business Language Layer itself

`src/olist/bll/` implements the paper's four functions on top of the trained
model and its SHAP explainer. **The layer reads explanations; it does not
retrain or replace the model.** This is the architectural constraint the paper
sets, and it is enforced by the test suite.

| Module | Role |
|---|---|
| `glossary.py` | Maps each feature to business language. Keeping this separate is what makes the domain-agnostic claim concrete: porting to credit risk means writing a new glossary, not rewriting the reasoning. |
| `explain.py` | SHAP attributions through the calibrated pipeline; counterfactual simulation |
| `brief.py` | The four functions — interpret, contextualise, quantify, recommend |
| `similarity.py` | Retrieves comparable historical orders and their actual outcomes |
| `confidence.py` | Estimates how much a given prediction should be trusted |
| `service.py` | The single interface both the dashboard and the REST API call |
| `report.py` | Exports a decision brief as a PDF |

### 9.1 Faithfulness to the underlying evidence

SHAP explains the base classifier's internal score. The number displayed to the
user is the **calibrated** probability. These are related by an order-preserving
transformation, but they are not identical.

Attributions are therefore presented as contributions to the risk *ranking*, not
as an additive decomposition of the displayed percentage.

Figure 6 of the paper prints `E[f(x)] = 0.310 → f(x) = 0.710`, which implies the
stronger, additive claim. With a genuinely calibrated model that is not exactly
right, and the figure would benefit from a footnote. This is raised as an open
question in Section 12.

One-hot encoded columns are summed back onto their source feature, so a reader
receives one statement about customer state rather than thirty about individual
states.

### 9.2 Traceability

`DecisionBrief.audit()` returns every business statement alongside the feature
and SHAP value behind it.

A feature with no glossary entry is **flagged as unmapped** rather than given a
generic phrase. The layer cannot silently invent business language it was never
given.

This addresses the governance direction named in the paper's future work,
implemented rather than proposed.

### 9.3 What the system can and cannot offer

Olist records no courier tier, no warehouse identifier, no weather observation
and no seller rating. Consequently **no courier-upgrade lever is offered**.

The levers this data supports are **dispatch speed** (`carrier_pickup_days`) and
**promise length** (`estimated_delivery_days`).

Unavailable levers are displayed as unavailable, with the missing field named,
rather than hidden. A silently dropped option looks identical to an option that
never existed — and the distinction matters to anyone judging whether the system
is honest.

Counterfactual simulations run the full deployed pipeline rather than
approximating from SHAP values, so every what-if figure is the system's own
output.

---

## 10. Limitations

Stated plainly, because a fluent costed recommendation reads as more
authoritative than its evidence supports — the exact risk the paper identifies.

1. **Cost parameters are assumed, not measured.** Every monetary figure inherits
   that assumption. The notebook includes a sensitivity table showing how the
   recommendation moves as the assumptions change.

2. **Precision at the operating point is modest** (about 25%). The model ranks a
   limited alert budget usefully. The score for any individual order is weak
   evidence and should be treated as a prioritisation signal, not a verdict.

3. **`estimated_delivery_days` is partly tautological.** The target is defined
   relative to the promise, so a longer promise mechanically lowers late risk.
   This is a lever Olist genuinely controls — on-time performance can be bought
   with more conservative estimates, at some cost to conversion — but it is
   worth naming.

4. **No causal claim is made.** Counterfactuals are the model's answer under an
   all-else-equal assumption. Any intervention policy built on them should be
   validated by a holdout experiment before being trusted.

5. **Geographic and temporal scope.** Brazilian data ending October 2018.
   Carrier networks change. The paper's illustrative figures are Nigerian and in
   naira; these results are in Brazilian reais.

6. **The 30-day censoring buffer is a judgement call**, not a derived quantity.

7. **Currency mismatch with the paper's figures**, which requires an authorial
   decision rather than a technical one. See Section 12.

---

## 11. Reproducing this work

```bash
pip install -e ".[app,api,dev]"
pytest                                    # 129 tests

python -m olist.validate                  # preflight: files, schemas, join keys
python -m olist.validate --build          # clean and merge
python -m olist.train --scenario post_dispatch

streamlit run app/dashboard.py            # the decision surface
uvicorn olist.api.app:app --reload        # the REST API, docs at /docs
```

Random seed 42 throughout, set in `config.py`.

Note that the train/test split is **temporal**, not seeded. It is fixed by the
calendar, which is a stronger form of reproducibility than "same seed, same
answer" — a different machine with a different random implementation still
produces the same split.

Environment versions are recorded inside the saved model artefact. This is not
bureaucratic: scikit-learn may load an object saved by a different version with
*altered behaviour* rather than refusing outright, which is worse than an error.
The recorded versions must be pinned wherever the model is served.

---

## 12. Open questions for the team

These are decisions for the authors, not technical gaps.

1. **Should the Olist instantiation replace the illustrative figures, or sit
   alongside them as a validation section?** This determines how the currency
   mismatch is resolved — either present the results in Brazilian reais as a
   separate section, or restate the illustrative figures.

2. **Figure 5's "Slow recent dispatch" driver is weakly supported by this data**
   (+0.002 AUC). Regenerate the figure from real output, or report the
   discrepancy as a finding?

3. **Figure 6's additive notation** implies SHAP values decompose the displayed
   probability. With a calibrated model this is not exactly right. Add a
   footnote?

4. **Cost parameters.** If DataLab has access to real e-commerce refund, support
   and churn figures, substituting them would convert every monetary claim in
   the paper from illustrative to grounded. This is the single highest-value
   change available.

5. **Should the trained model file (3.6 MB) live in git history**, or in GitHub
   Releases? Every retrain adds permanently to repository size.

---

*This document accompanies the prototype codebase at `Prototype Codebase/olist/`
and the case study notebook at `notebook/`. Every figure quoted here is
reproducible from that code.*