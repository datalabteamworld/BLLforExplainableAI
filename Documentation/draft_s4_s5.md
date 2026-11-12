# Phase 3 draft — §4 Data and Methods, §5 Empirical Results

Formatted to the MIRG-ICAIR template: numbered bold level-1 headings, bold-italic
level-2, italic level-3, APA 7 citations, tables with a source line. Tables are
numbered from 2, since Table 1 is the existing IXAII table.

Every number is traceable to `reports/paper_evidence.md`, `reports/ablations.md` or
`reports/sensitivity.md`. See `Documentation/number_verification.md` for the audit.

---

**4. Data and Methods**

This section describes the dataset, the construction of the modelling table, the
controls applied to prevent information leakage, the evaluation protocol, and the
measures taken to make the analysis reproducible. The Business Language Layer is a
presentation and reasoning layer rather than a predictive model, so the evaluation
below establishes the quality of the predictions the layer consumes. It does not
evaluate the layer itself; §9 states what such an evaluation would require.

***4.1 Data selection and Nigerian context***

Because publicly available Nigerian order-level e-commerce datasets with both
promised and actual delivery information are limited, the empirical evaluation uses
the Olist Brazilian e-commerce dataset, a real-world transaction dataset that
provides the required order, seller, product, logistics, and delivery information.
The use of Olist is intended to provide a reproducible empirical test of the
proposed framework rather than to represent Nigerian transaction behaviour. To
examine contextual applicability in Nigeria, a separate synthetic Nigerian
environment was developed using Nigerian locations, logistics scenarios, and
explicitly documented assumptions. The synthetic environment is therefore treated
as a contextual demonstration rather than empirical evidence of Nigerian market
behaviour. Validation with anonymised real-world Nigerian logistics or e-commerce
data is identified as an important direction for future work.

***4.2 Dataset***

The empirical case study uses the Brazilian E-Commerce Public Dataset by Olist
(Olist & Sionek, 2018), a publicly available record of 99,441 orders placed on a
Brazilian marketplace between September 2016 and August 2018 and released under a
CC BY-NC-SA 4.0 licence. The dataset comprises nine relational tables covering
orders, order items, customers, sellers, products, payments, reviews, geolocation
and a category-name translation.

Two properties make it suitable for the evaluation described above. It records
both the promised and the actual delivery date for every order, so late delivery
is directly observable rather than inferred from a proxy; and it is openly
licensed, so the analysis reported here can be reproduced independently.

Monetary values in the source data are denominated in Brazilian reais (BRL). For
contextual presentation they are reported here in Nigerian naira (NGN), converted
per order at the official annual average exchange rate for that order's purchase
year, derived from International Monetary Fund International Financial Statistics
official rates for both currencies (2016: ₦72.63/BRL; 2017: ₦95.86/BRL; 2018:
₦83.86/BRL; value-weighted effective rate ₦83.86/BRL). Conversion is applied per
order rather than at a single average because the cross-rate moved by more than
20% across the observation window. The conversion provides a common naira
representation of monetary scale. **It does not imply that the underlying
transactions occurred in Nigeria or that they represent Nigerian market prices.**

***4.3 Target definition and censoring***

The prediction target is binary: whether an order was delivered to the customer
after the delivery date promised at purchase. Only orders with a status of
*delivered* and a recorded customer delivery date are retained, since lateness is
undefined for orders that were cancelled or never arrived.

A right-censoring filter is then applied. An order whose promised delivery date
falls within 30 days of the last delivery observed in the dataset cannot yet be
known to be late, because the window in which it might still arrive on time
extends beyond the data. Retaining such orders would label an unknown outcome as
on-time and bias the target downward. The filter removed 299 orders, moving the
observed late rate from 8.11% to 8.14% — a small change in magnitude, but one that
corrects a systematic bias rather than a random one.

**Table 2.** Construction of the modelling table

| Stage | Orders | Late rate |
|---|---|---|
| Raw order records | 99,441 | — |
| Delivered, with a recorded delivery date | 96,470 | 8.11% |
| After 30-day right-censoring filter | **96,171** | **8.14%** |

***Source:** Authors' creation (2026)*

The resulting modelling table contains 96,171 orders described by 89 columns and
spans purchases from 15 September 2016 to 29 August 2018.

A note on grain is warranted, because it is a common source of error in analyses of
this dataset. The order-items table contains 112,650 rows, one per item. Joining at
that grain and modelling late delivery would count multi-item orders more than
once and weight the evaluation toward larger baskets. All features are therefore
aggregated to the order grain before modelling, and every count reported in this
paper is a count of orders.

***4.4 Leakage controls***

Three classes of variables were excluded from the feature set.

*Post-outcome fields.* Variables that are derived from, or revealed at the same
time as, the outcome cannot be features: the actual delivery date, the realised
delivery duration, the delay in days, and the customer review and its timestamps.
Including any of these would allow the model to observe the outcome it is asked to
predict.

*Fields unavailable at the moment of prediction.* The time at which the seller
handed the parcel to the carrier is knowable only once that handover has occurred.
Two prediction scenarios are therefore defined and evaluated separately.
**Pre-dispatch** uses only information available when the order is placed (62
features); **post-dispatch** additionally admits carrier-handover timing (63
features). The distinction is operational rather than technical: a pre-dispatch
prediction supports a wider set of interventions because the parcel has not yet
moved, whereas a post-dispatch prediction is more accurate but arrives later. Both
are reported.

*Unbounded time counters.* Cumulative counters — a seller's orders to date, tenure
in days, and similar — increase monotonically with calendar time. Under a temporal
split their values in the evaluation period fall outside the range observed during
training entirely, so linear models extrapolate them without bound and tree models
cannot extrapolate at all. Bounded equivalents, expressed as rates to date with an
accompanying confidence weight, carry the same information on a stable scale and
were retained in their place.

***4.5 Evaluation split***

The data were divided by purchase date rather than at random. Orders placed on or
before 30 April 2018 form the training set (70,883 orders); orders placed after
that date form the evaluation set (25,288 orders).

A random split would permit the model to train on later months and be evaluated on
earlier ones, allowing it to score well by memorising which periods had elevated
late rates — information that is unavailable at deployment. A temporal split
reproduces the direction in which a deployed model actually operates.

The split is not merely methodologically preferable; it is also the harder test.
The late rate in the training period is 8.81%, while in the evaluation period it
falls to 6.25%. The distribution the model is evaluated against therefore differs
measurably from the one it learned, as it would in deployment.

***4.6 Model and calibration***

A logistic regression classifier was fitted within a pipeline that performs all
preprocessing — imputation, scaling, and encoding of categorical variables —
inside cross-validation, so that no information from the evaluation data
influences any fitted transformation. Class imbalance was addressed with SMOTE
(Chawla et al., 2002) applied inside the same pipeline, which ensures the synthetic
minority examples are generated only from training folds.

Probability calibration was applied as a separate step. Resampling methods such as
SMOTE train the classifier on an artificially balanced distribution, with the
consequence that its outputs rank cases correctly but are not probabilities of
anything observable. Because the Business Language Layer multiplies the predicted
probability by a monetary exposure to produce an expected loss, a miscalibrated
probability does not merely mislead — it propagates a multiplicative error into
every monetary figure the layer presents.

Platt scaling (Platt, 2000; Niculescu-Mizil & Caruana, 2005) was fitted on a
held-out slice of the training period, taken by date rather than at random: the
classifier was fitted on the first 56,706 training orders (to 26 February 2018) and
the calibration mapping learned on the 14,177 orders that follow (to 29 April 2018,
containing 2,011 late deliveries). Taking the calibration slice by date means the
mapping is learned on data later than the classifier saw, matching the direction of
deployment. Sigmoid scaling was preferred to isotonic regression because it is
strictly monotone and therefore cannot alter the ranking of cases.

***4.7 Evaluation protocol and cost parameters***

Models are evaluated on the held-out temporal test set using the area under the
ROC curve, average precision, and the Brier score (Brier, 1950). Average precision
is reported alongside the test-set positive rate, which is the value a no-skill
classifier attains, because on an imbalanced problem an absolute average precision
is uninterpretable without it.

An operating threshold was selected by minimising expected cost rather than
defaulting to 0.5. The cost model assigns a weight of 25 to a missed late delivery
and 4 to a false alarm. **These weights are stipulated, not measured.** They
reflect an assumption that failing to anticipate a late delivery is substantially
more costly than investigating an order unnecessarily; no observed cost data
supports the specific ratio.

The same qualification applies to the parameters that convert a prediction into a
monetary figure: a refund exposure of 42% of order value, a repeat-purchase risk of
46%, a fixed support-handling cost, and a fixed intervention cost. These are
configurable assumptions, exposed in the interface rather than embedded in code,
and §8 reports how far the layer's recommendation moves when they vary.

***4.8 Reproducibility***

All randomness is seeded. The analysis pipeline is implemented as a package with
an automated test suite of 159 tests. Dataset construction is verified by a
regression test that fingerprints the feature table — its row count, column list,
late rate, and per-column summary statistics to a tolerance of 10⁻⁶ — so that any
change to shared code which alters the empirical pipeline's output causes a test
failure rather than a silently different result.

The empirical Olist environment and the synthetic environment described in §7 are
separated structurally: each owns its own raw, processed and model directories,
every trained model records the environment that produced it, and loading a model
into the wrong environment raises an error rather than scoring silently. All
generation assumptions for the synthetic environment are documented separately.

Reproducibility in this sense means that the reported numbers can be regenerated
from the same inputs. It is a necessary condition for verification and not a claim
about external validity; §9 addresses the latter.

---

**5. Empirical Results**

***5.1 Predictive performance***

Table 3 reports performance on the 25,288 held-out orders, against the two
baselines a reviewer should expect.

**Table 3.** Model performance on the held-out temporal evaluation set (*n* =
25,288)

| Model | AUC | Average precision | Brier | Threshold | Recall | Precision |
|---|---|---|---|---|---|---|
| No-skill (predict base rate) | 0.500 | 0.0625 | 0.0586 | — | — | — |
| Always predict late | 0.500 | 0.0625 | — | — | 1.000 | 0.0625 |
| Logistic regression, pre-dispatch | 0.7413 | 0.1549 | 0.0602 | 0.1487 | 0.5367 | 0.1354 |
| **Logistic regression, post-dispatch** | **0.8389** | **0.3482** | **0.0514** | **0.1309** | **0.7481** | **0.1723** |

*Note.* All models calibrated. Threshold selected by expected cost, not fixed at
0.5, on the final 20% of the training period (the calibration slice), before the
evaluation labels are seen. Choosing it on the evaluation set instead would give
0.2074 (recall 0.5608, precision 0.2585) for the post-dispatch model — an upper
bound, not an estimate. The positive rate in the evaluation set is 6.25%.

***Source:** Authors' creation (2026)*

The post-dispatch model attains an AUC of 0.8389 and an average precision of
0.3482. The latter figure is the more informative of the two on an imbalanced
problem: it is **5.6 times the no-skill value of 0.0625**, meaning the model
concentrates late deliveries into the top of its ranking far above chance. At the
cost-optimal threshold, chosen on the training period, it identifies 74.8% of
late deliveries, with 17.2% of flagged orders in fact arriving late; it flags 27.1%
of orders, at an expected cost of 1.29 per order under the Table 2 parameters.

The pre-dispatch model is materially weaker (AUC 0.7413, average precision
0.1549), which quantifies the cost of predicting before the carrier handover is
observed. The gap is not a deficiency to be corrected but an operational trade-off
to be stated: the earlier prediction supports a wider set of interventions and is
less accurate, and an organisation deploying this layer must choose where on that
trade-off to sit.

***5.2 The effect of calibration***

Calibration was isolated by holding the fitted classifier constant and evaluating
it with and without the sigmoid mapping. No other quantity differs between the two
rows of Table 4.

**Table 4.** Effect of probability calibration, same fitted classifier

| | AUC | Average precision | Brier | Cost-optimal threshold |
|---|---|---|---|---|
| Uncalibrated | 0.8389 | 0.3482 | 0.3085 | 0.7247 |
| Calibrated | 0.8389 | 0.3482 | **0.0514** | **0.1309** |

***Source:** Authors' creation (2026)*

The Brier score falls by 83.3%, from 0.3085 to 0.0514, while AUC and average
precision are unchanged to six decimal places. That ranking metrics are exactly
preserved is the expected consequence of a strictly monotone transformation and
serves here as a correctness check: calibration altered the probabilities attached
to the ranking without altering the ranking itself.

The practical consequence is visible in the threshold. Before calibration, the
cost-minimising cutoff sits at 0.7247 on a problem with a 6.25% base rate — a value
that cannot be interpreted as a probability of anything. After calibration it sits
at 0.1309. For a system that displays a probability to a non-technical reader and
multiplies it by an order value, this is the difference between a number that
carries meaning and one that does not. This is the specific sense in which
calibration is load-bearing for the Business Language Layer rather than an optional
refinement, and it is a requirement that the layer's own design documentation
identifies but that the literature on post hoc explanation has little occasion to
raise.

One qualification must be stated. Across a random sample of 500 evaluation orders,
the mean predicted probability is 0.1126 against an observed late rate of 6.25% in
that period: the model over-predicts in the mean by approximately 1.8 times. This
is consistent with the calibration mapping having been fitted on February–April
2018 while the base rate continued to decline through the evaluation window.
Calibration corrects the scale of the probabilities at the point it is fitted, and
that correction decays as the underlying rate moves. A full reliability analysis
across probability deciles has not been conducted and is identified in §9 as
required work. The practical implication is that monetary figures produced by the
layer should be read as upper estimates under the present calibration, and that
periodic recalibration is a deployment requirement rather than a refinement.

***5.3 Delivery lateness and customer satisfaction***

Among the 95,527 orders carrying a review, those delivered late received a mean
review score of 2.5651 against 4.2942 for those delivered on time — a difference of
**−1.7291 stars on a five-point scale** (95% bootstrap confidence interval
[−1.7680, −1.6925], 2,000 resamples; Mann–Whitney *U* = 150,189,672, *p* < .001;
rank-biserial correlation = 0.5538). The comparison rests on 7,661 late and 87,866
on-time orders.

The effect is large by any conventional standard, and its interval is narrow. It is
nonetheless **an association and not a causal estimate**. It is computed as a
difference in group means with no covariates and no controls, on observational
data in which lateness is not randomly assigned. Orders that arrive late may differ
systematically from those that do not — in route, in seller, in product category,
in season — and any of those differences could contribute to the gap. The figure is
reported because it establishes the magnitude of what is at stake in a late
delivery, which is the quantity the Business Language Layer translates into a
business consequence. It is not reported as evidence that preventing a late
delivery would recover 1.73 stars.

***5.4 What drives lateness, and what can be acted upon***

Two complementary analyses of feature influence were conducted, and their
disagreement is itself a finding.

Permutation importance (Breiman, 2001) was measured on the held-out evaluation set
by shuffling each feature in turn and recording the resulting drop in AUC. Grouping
the features by the operational category they belong to gives Table 5.

**Table 5.** Permutation importance by feature group, held-out evaluation set

| Feature group | Total AUC drop when shuffled |
|---|---|
| **Dispatch and promise** | **0.520** |
| Routing and geography | 0.118 |
| Basket and price | 0.107 |
| Calendar and seasonality | 0.086 |

*Note.* Computed on 8,000 evaluation orders, 5 repetitions.

***Source:** Authors' creation (2026)*

Dispatch-related features account for **4.4 times** the predictive dependence of
routing and geography. The three largest individual contributions are the number of
working days available within the promised window (0.219), the length of the
promised window itself (0.141), and the time taken to hand the parcel to the
carrier (0.131). On this dataset, late delivery is predominantly a dispatch problem
rather than a routing problem — a conclusion with a direct operational reading,
since dispatch timing is under the seller's control in a way that distance is not.

The second analysis used SHAP attributions (Lundberg & Lee, 2017), computed on the
fitted model against a background sample drawn from the training period. The two
rankings diverge substantially, as Table 6 shows.

**Table 6.** Divergence between attribution magnitude and predictive dependence

| Feature | SHAP rank | Permutation rank |
|---|---|---|
| Purchase month | 1 | 8 |
| Purchase quarter | 2 | 9 |
| Purchase year | 3 | outside top 20 |
| Working days in window | 4 | **1** |
| Time to carrier handover | 7 | **3** |

***Source:** Authors' creation (2026)*

The two measures answer different questions. SHAP quantifies how much a model
*uses* a feature in forming a particular prediction; permutation importance
quantifies how much predictive performance *depends* on it. A feature can be used
heavily and yet be substitutable, in which case shuffling it costs little because
correlated features absorb the loss.

The divergence matters specifically for the contribution advanced in this paper. An
explanation surface that ranks contributing factors by attribution magnitude would
present seasonality as the dominant driver of this prediction. The statement would
be faithful to the model. It would also be useless to an operations lead, because
no action available to that person changes the month in which an order was placed.
Ranking by predictive dependence instead surfaces three factors that are all
operational levers.

This is the empirical basis for a design commitment that the Business Language Layer
implements and that §3.3 describes: the layer distinguishes features that can be
acted upon from features that merely predict, and discloses which interventions the
available data cannot support rather than offering a lever it cannot evaluate. The
result reported here indicates that the distinction is not a stylistic preference
but a requirement, because faithful attribution and actionable attribution are
empirically not the same ordering.

***5.5 Feature-set ablation***

Because three calendar features dominate the attribution ranking, the model was
refitted without them to establish whether they carry independent signal or merely
absorb attribution.

**Table 7.** Calendar-feature ablation

| Feature set | Features | AUC | Average precision | Recall |
|---|---|---|---|---|
| All features | 63 | 0.8389 | 0.3482 | 0.7481 |
| Without purchase year | 62 | 0.8199 | 0.3026 | 0.7051 |
| Without all calendar features | 60 | 0.8159 | 0.2982 | 0.6778 |

***Source:** Authors' creation (2026)*

Removing the purchase year alone costs 0.019 AUC and reduces recall from 0.7481 to
0.7051; removing all three costs 0.023 AUC. The features therefore carry genuine
signal, consistent with the observed decline in the marketplace's late-delivery
rate across the observation window, which a year indicator captures as a level
shift.

This creates a deployment caveat that the evaluation reported here cannot itself
test. Every value of the calendar features appearing in the evaluation period also
appears during training, because the evaluation window falls inside the training
years. A model deployed on a year it has never observed would have no learned
effect to apply, and the ablation quantifies what is at risk: approximately 0.019
AUC. Periodic refitting addresses this, and is recommended alongside the
recalibration identified in §5.2.

---

## Drafting notes for James — not part of the manuscript

**1. The generative-AI figure policy is a submission blocker.** The template states
that figures must not be created with generative AI and that genAI must not be used
for manuscript development or figure creation; only editing for grammar and clarity
is permitted, with declaration. The paper currently discloses AI-assisted creation
of all six original figures. Three options, in my order of preference:

   (a) **Redraw the figures without generative tools.** Figures 3, 4, 7 and 8 are
   diagrams and tables — they can be rebuilt in Inkscape, draw.io, PowerPoint or
   matplotlib and marked "Authors' creation". Figure 5 should become a screenshot
   of the running dashboard, which is not a created figure at all. This resolves the
   audit's Figure 5 problem at the same time.

   (b) Ask the organisers whether AI-assisted *layout* of author-designed content
   falls within the prohibition. The policy may be aimed at generated imagery.
   Get any answer in writing.

   (c) Submit elsewhere.

   I would not submit with the current disclosure unchanged. The paper would be
   declaring a policy violation on its own first page.

**2. Format items I could not resolve without you.** The abstract must be 150–200
words; the current one is roughly 300. Keywords must be 3–5 in alphabetical order;
the paper has 8, unordered. Both are in §10's rewrite rather than here.

**3. Required closing statements.** The template requires ethical, funding,
conflicting-interest and data-availability statements before the references, plus a
generative-AI declaration. I can draft all five, but the funding and conflict
statements need facts only you and your director have.

**4. Figures 9 and 10.** §5.4 and §5.2 would each benefit from a figure — permutation
importance by group, and a calibration reliability curve. Both plot measured values
and can be generated from `reports/`. Say the word and I'll write the plotting
script. Note that matplotlib output is not generative AI, so these are safe under
the policy.

**5. References used in these two sections**, APA 7. **DOIs require verification
before submission** — I have not confirmed them and will not supply unverified ones:

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability.
  *Monthly Weather Review, 78*(1), 1–3.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE:
  Synthetic minority over-sampling technique. *Journal of Artificial Intelligence
  Research, 16*, 321–357.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
  predictions. In *Advances in Neural Information Processing Systems 30* (pp.
  4765–4774).
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with
  supervised learning. In *Proceedings of the 22nd International Conference on
  Machine Learning* (pp. 625–632).
- Olist, & Sionek, A. (2018). *Brazilian e-commerce public dataset by Olist*
  [Data set]. Kaggle.
- Platt, J. C. (2000). Probabilistic outputs for support vector machines and
  comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, B.
  Schölkopf, & D. Schuurmans (Eds.), *Advances in large margin classifiers* (pp.
  61–74). MIT Press.
- International Monetary Fund. (2026). *International financial statistics:
  Official exchange rate (LCU per US$, period average)* [Data set].

**6. One claim I softened deliberately.** §5.4 could be written to say the paper
*discovered* that dispatch dominates routing. It is more defensible as a finding
about this dataset under this model, which is how it is written. If your director
wants it stronger, the honest strengthening is a second dataset, not firmer wording.
