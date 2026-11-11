# Olist late-delivery risk

Predicting, at the moment an order is placed, whether it will arrive after its
promised delivery date — so that Brazilian marketplace Olist can intervene
before the customer is affected rather than apologise afterwards.

Built on the [Brazilian E-Commerce Public Dataset by Olist][kaggle]
(~100k orders, September 2016 – October 2018).

[kaggle]: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

---

## Why late delivery

Delivery timeliness is the strongest observable driver of review score in this
dataset, and review score is the only available proxy for churn. Lateness is
also the rare outcome a marketplace can actually *act* on: it is knowable in
advance, and the intervention — expedite dispatch, or warn the customer — is
cheap relative to the loss.

The target is `delivery_delay_days > 0`: delivered later than the date the
customer was promised. Roughly 8% of delivered orders.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"        # editable install; puts olist on the path

# Place the nine Kaggle CSVs in data/raw/ — see data/README.md
python -m olist.validate                  # preflight: files, schemas, join keys
python -m olist.validate --build          # clean + merge only, writes parquet
python -m olist.train --scenario pre_dispatch            # baseline, minutes
python -m olist.train --scenario pre_dispatch --tune --fast   # halving search
python -m olist.train --scenario pre_dispatch --tune          # full search, slow

pytest                                    # 161 tests
streamlit run app/streamlit_app.py
```

Outputs land in `models/` (one pickled pipeline) and `reports/` (metrics,
cost curve, permutation importances, cleaning audit).

---

## Two prediction scenarios

The distinction is what makes the model deployable, and it is easy to get
wrong: a feature can be present in the training data and still be unavailable
at the moment you need to predict.

| Scenario | Available | Intervention |
|---|---|---|
| `pre_dispatch` | Everything known when the order is placed | Expedite dispatch, re-route, warn the customer |
| `post_dispatch` | Adds carrier handover timings | Escalate a parcel already in transit |

`carrier_pickup_days` belongs only to the second. It requires the carrier to
have already collected the parcel, so a pre-dispatch model that uses it cannot
run when it needs to. The scenario split is enforced in
`pipeline.feature_columns` and asserted in `tests/test_pipeline.py`.

---

## Method

**Grain.** One row per order. The source tables mix grains — payments are per
instalment, items per line, reviews roughly per order — and each is aggregated
to order level before joining. Every merge asserts the row count is unchanged,
so a fan-out from a non-unique key fails loudly instead of silently
duplicating rows.

Modelling at order-item grain, as is common with this dataset, gives every
item of a multi-item order the same label. A random split then scatters those
near-identical siblings across train and test, which raises the measured score
without improving the model.

**Validation.** Temporal split: train on orders through April 2018, test on
everything after. A random split lets the model train on December and be
tested on November, and it then scores well by memorising which months had
high late rates — information deployment never provides.

**Imbalance.** SMOTE is a step *inside* the pipeline, so cross-validation
refits it within each fold and it never sees validation data. Resampling
before cross-validation lets synthetic points in a validation fold be
interpolated from real points in the training fold; the folds stop being
independent and CV scores detach from reality.

**Encoding.** Low-cardinality categoricals are one-hot encoded.
`customer_city` has ~4,000 levels, so it is target encoded — scikit-learn's
`TargetEncoder` cross-fits internally and does not leak. Label encoding these
would impose an arbitrary ordinal ranking, meaningless to a linear or
distance-based model.

**Calibration.** SMOTE trains the classifier on an artificially balanced
world, so its outputs are not probabilities of anything real — logistic
regression scored a Brier of 0.266 against XGBoost's 0.051, and its
cost-optimal threshold landed at 0.77 on a 6% base rate. The final estimator
is therefore calibrated on a held-out *later* slice of the training period, so
the correction is learned in the same direction as deployment.

**Selection.** By expected business cost, not AUC. See below.

---

## Feature engineering

Beyond the usual price, weight, distance and calendar decomposition, two
groups do most of the work.

**Transit-window events.** Rather than flagging whether an order was *placed*
on a holiday, the pipeline counts non-working days — weekends, national
holidays, and the five days of Carnival — falling between purchase and the
promised date. Lateness is caused by the days the carrier must traverse, not
by the purchase date itself. An order placed the day before Carnival is at
risk because of the five days that follow it, which a purchase-date flag
cannot express. `working_days_in_window` is the derived quantity that matters:
how many usable days the promise actually contains.

**Order composition.** `n_sellers`, `is_multi_seller`, `n_items`, and price
spread. A two-seller order ships only when the slower seller dispatches, so
its exposure is the maximum of two processes rather than the average of one.

**History** (`history.py`). Rolling late rates per seller, route, category and
marketplace, smoothed toward a running prior. These are the easiest features
in the project to leak: an order's lateness only becomes known when it is
*delivered*, and late orders resolve later than on-time ones, so an expanding
mean over purchase dates is biased toward exactly the outcome being predicted.
An order contributes to history only once its delivery date has passed.

Counters that only increase with time — `seller_orders_to_date`,
`seller_tenure_days`, `market_late_rate_to_date` — are **excluded**. Under a
temporal split their test values fall outside the training range entirely (89%
of test rows exceed the training maximum for tenure). Linear models
extrapolate them without bound and trees cannot extrapolate at all. The
bounded equivalents, `*_late_rate_to_date` and `*_history_confidence`, carry
the same information on a stable scale.

**Right-censoring** (`history.apply_censoring_filter`). An order enters the
table only when delivered, so near the data cutoff the late ones are
disproportionately still in transit. Orders whose promised date falls within
30 days of the last observed delivery are dropped, so the measured late rate
reflects performance rather than observation windows.

---

## Choosing the operating point

The default 0.5 threshold assumes a false positive and a false negative cost
the same. Here they plainly do not: missing a late delivery costs a refund, a
support contact and the churn implied by the review drop; flagging an on-time
order costs one message or a shipping upgrade.

`evaluate.choose_threshold` sweeps every distinct predicted probability and
picks the one minimising `cost_fn × FN + cost_fp × FP`, then reports the
saving against the do-nothing baseline. The cost figures live in
`config.CostConfig` — **the defaults are placeholders**; substitute
Olist-specific numbers before quoting any monetary result.

Alongside it, `recall_at_budget` answers the question operations teams
actually ask: *if we can chase 500 orders a week, what share of late ones do
we catch?*

This matters because ranking metrics can pick the wrong model outright. In an
earlier iteration of this analysis, XGBoost won on AUC (0.862) while recalling
only 29.6% of late deliveries — the worst of four candidates — on a problem
whose entire justification was catching them.

---

## Tuning cost

`--tune` runs 40 configurations x 5 folds per model. `--fast` uses successive
halving instead: many configurations on a small subsample, with survivors
given progressively more data. It reaches a comparable optimum for a fraction
of the budget.

Three further things keep the search tractable. XGBoost uses
`scale_pos_weight` rather than SMOTE — resampling ~70k rows is a large share
of every fit and buys gradient boosting nothing the weight does not. The
estimator runs single-threaded inside the search, because nesting `n_jobs=-1`
within `n_jobs=-1` oversubscribes the CPU and inflates memory until workers
are killed. And XGBoost uses CUDA when a device is genuinely visible; the
probe checks `nvidia-smi` and the build flags rather than attempting a fit,
because XGBoost logs its CPU fallback from C++ and returns successfully, so a
try/except would report a GPU that is not there.

## Interpretation

Permutation importance on the held-out set, not impurity-based importance.
Impurity importance is biased toward high-cardinality continuous features,
which is how coordinates and product dimensions end up outranking real signal.

---

## Business Language Layer

`src/olist/bll/` instantiates the four functions of the BLL framework
(Oyeleke et al., 2026) — interpret, contextualise, quantify, recommend — on
top of the trained model and its SHAP explainer. It reads explanations; it
does not retrain or replace the model.

```bash
streamlit run app/dashboard.py        # dashboard: risk, sales and explainability
uvicorn olist.api.app:app --reload    # REST API; interactive docs at /docs
streamlit run app/decision_app.py     # single-screen decision surface
```

`app/decision_app.py` is the primary interface: one screen per order,
structured as the four rungs of the paper's Figure 7 — *why*, *so what*,
*what then*, *now what*.

It covers all six explanation types of Table 1 (Speckmann et al., 2025):

| Type | Where |
|---|---|
| What / Output | Risk probability, band, confidence |
| Why | SHAP drivers in business language |
| How | Model card, global importance |
| **Why Not** | "Why not lower?" — what would each factor have to move, and by how much |
| What If | Lever simulation against the live pipeline |
| **When** | "What would it take?" — bisection search for the input value achieving a target risk |

Plus similar historical orders, the Figure 6 before/after view, the Figure 4
five-register strip, an audit trail, and PDF export.

Also available: `app/platform/Home.py` (multi-page platform with a portfolio
dashboard and order queue), `app/cloud_app.py` (self-contained, for Streamlit
Community Cloud), and `app/bll_app.py` (the original minimal surface).

`app/cloud_app.py` is the deployable variant: it reads only a bundled model
artefact and a sample of scored orders, so it runs on Streamlit Community
Cloud without the raw data. Section 7 of
`notebooks/olist_bll_case_study.ipynb` builds that bundle.

Two design commitments, both taken from Section 6 of the paper, which names
them as load-bearing rather than optional:

**Traceability.** Every business statement carries the feature and SHAP value
it came from; `DecisionBrief.audit()` returns that mapping. A feature with no
glossary entry is flagged as unmapped rather than given a generic phrase, so
the layer cannot silently invent business language it was never given.

**Honesty about uncertainty.** SHAP explains the base classifier's decision
function; the number shown is the *calibrated* probability. These are related
by a monotone map, not identical, so attributions are presented as
contributions to the risk *ranking* rather than as an additive decomposition
of the displayed percentage. The brief also carries the model's precision at
its operating point, and declares loudly when a model is uncalibrated.

Counterfactuals run the full deployed pipeline rather than approximating from
SHAP values, so what-if figures are the system's own output. Levers absent
from a scenario — `carrier_pickup_days` under pre-dispatch — are shown as
unavailable rather than hidden.

The interventions offered are the ones Olist data supports: dispatch speed and
promise length. There is no courier-tier field in this dataset, so no
courier-upgrade lever is offered.

## REST API

`src/olist/api/` is a thin FastAPI wrapper over `PredictionService`. It
reimplements nothing: the leakage rules, thresholds and cost logic all stay in
the tested package.

```bash
uvicorn olist.api.app:app --reload     # docs at http://localhost:8000/docs
```

| Endpoint | Purpose | Cost |
|---|---|---|
| `POST /score` | probability, band, confidence | milliseconds |
| `POST /score/batch` | up to 1,000 orders in one call | milliseconds |
| `GET /orders/{id}/explanation` | SHAP drivers in business language | slow (SHAP) |
| `GET /orders/{id}/brief` | the full decision brief | slow (SHAP) |
| `GET /orders/{id}/similar` | comparable historical orders | fast |
| `GET /orders/{id}/why-not` | what would have to change for a lower band | medium |
| `POST /what-if` | recompute with one input changed | fast |
| `POST /target` | what value reaches a target risk | medium |
| `GET /model` | model card and capability disclosure | cached |
| `GET /model/importance` | global feature importance | cached |
| `POST /outcomes` | report what actually happened | fast |
| `GET /monitoring/calibration` | predicted risk vs observed outcomes | fast |
| `GET /monitoring/predictions` | recent predictions, for audit | fast |
| `GET /health` | readiness and loaded model version | fast |

Three conventions the API holds to:

**Fast and slow paths are separate.** Bundling SHAP into `/score` would make
every caller pay for an explanation most do not want.

**Capabilities are disclosed, not omitted.** `/model` returns the levers this
dataset cannot support along with the field each would require, and `/what-if`
returns a 422 naming an absent lever rather than a silent zero. A client that
only sees available options cannot distinguish "the data has no courier field"
from "not implemented yet".

**Money never travels without its assumptions.** `/brief` takes cost
parameters per request and echoes them back, so a client cannot display a
figure without the assumptions that produced it.

**Calibration is the metric to monitor**, not accuracy. Ranking can stay intact
while probabilities drift, and when that happens every monetary figure derived
from them becomes quietly wrong. `POST /outcomes` feeds
`GET /monitoring/calibration`; without it the service can never know whether
its probabilities are still honest.

## Repository layout

```
src/olist/
  config.py       Paths, column groups, cost and tuning settings
  io.py           Raw loading: explicit dtypes, BOM handling
  cleaning.py     Per-table cleaners, each returning an audit report
  calendar_br.py  Brazilian holidays, Black Friday, Carnival; window counts
  features.py     Order-level aggregation, merge chain, derived features
  pipeline.py     Preprocessing, model zoo, search spaces, temporal split
  evaluate.py     Cost-based threshold selection, budget-constrained recall
  validate.py     Preflight checks; clean-and-merge CLI
  train.py        CLI entry point
  bll/
    glossary.py   Feature -> business-language table (domain knowledge)
    explain.py    SHAP through the calibrated pipeline; counterfactuals
    brief.py      Interpret, contextualise, quantify, recommend
    service.py    PredictionService — the boundary the UI and API both call
  api/
    app.py        FastAPI routes
    schemas.py    Request and response contracts
    store.py      Prediction log and calibration monitoring
tests/
  synthetic.py    Olist-shaped fixtures reproducing the real defect patterns
  test_pipeline.py
app/streamlit_app.py
```

Cleaning is non-destructive: rows are flagged or set to `NaN`, never dropped,
so counts stay reconcilable against the raw files and downstream code decides
what to do with them. Every rule corresponds to a defect verified in the raw
release and pinned by a test.

---

## Limitations

- **`estimated_delivery_days` is partly tautological.** The target is defined
  relative to the estimate, so a longer promise mechanically lowers late risk.
  This is a lever Olist controls: on-time performance can be bought with more
  conservative estimates, at some cost to conversion. Treat the feature's
  importance as a statement about promise-setting, not about logistics.
- **The data ends in October 2018.** Carrier networks change; nothing here
  should be assumed to hold today.
- **Review score is a weak churn proxy.** Repeat purchase is observable but
  rare (~3% of customers), so any lifetime-value claim rests on thin evidence.
- **Cost parameters are assumed, not measured.** Every monetary figure the
  pipeline reports inherits that assumption.
- **No causal claim.** The model ranks risk; it does not establish that
  holidays *cause* delay, and an intervention policy built on it should be
  validated by a holdout experiment before being trusted at scale.
