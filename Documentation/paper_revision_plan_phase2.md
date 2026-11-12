# Phase 2 — Section-by-section revision plan

> **Superseded figures (11 Nov 2026).** The operating-point numbers below (threshold 0.2074, recall 0.5608, precision 0.2585, 0.8247 → 0.2074, recall 0.5608 → 0.4329) were chosen on the test labels. See the revision section of `number_verification.md` for the out-of-sample replacements.

Date: 6 October 2026
Status: **plan only.** No manuscript text has been written. Awaiting approval before Phase 3.

Decisions taken: figures keep naira and are labelled illustrative; sensitivity analysis to be
built; empirical sections written in full.

---

## 1. What the two runs settled

Every figure below is now traceable to `reports/paper_evidence.json` and
`reports/ablations.json`. These supersede §0.1 of the Phase 1 audit, which was written before
the runs existed.

| Quantity | Verified value | Status against the brief |
|---|---|---|
| Orders | **96,171** (96,470 before the 30-day censoring filter) | Confirmed; the two figures reconciled |
| Late rate | **8.14%** | Confirmed |
| Train / test | **70,883 / 25,288**, split 2018-04-30 | New — was nowhere |
| Train / test late rate | 8.81% / 6.25% | New |
| Features | **63** post-dispatch, 62 pre-dispatch | New |
| ROC AUC | **0.8389** | Confirmed |
| Average precision | **0.3482** | **Corrected** from 0.332 |
| Brier, uncalibrated → calibrated | **0.3085 → 0.0514** (83.3%) | **Corrected** from 0.28 → 0.06 |
| Decision threshold | 0.8247 → **0.2074** | New |
| Review-score difference | **−1.7291** stars, 95% CI [−1.7680, −1.6925] | Confirmed to the third decimal |
| Review sample | n = 95,527 (7,661 late, 87,866 on time) | New |
| Mann-Whitney | U = 150,189,671.5, p < 1e-300 | New |
| Rank-biserial | 0.5538 | New |
| Dispatch vs routing | **0.520 vs 0.118** AUC drop, 4.4× | New — the claim's first actual measurement |
| Tests | 159 | Confirmed (README's 129 is stale) |

Two corrections to the Phase 1 audit, recorded so the file is not left misleading:

- I suspected "Brier 0.28 → 0.06" was a mislabelled logistic-regression-versus-XGBoost
  comparison. It was not. It is a genuine calibration effect, and larger than claimed.
- I flagged `purchase_year` as a probable leakage artefact. It is not. Every test value occurs
  in training, and removing the feature costs 0.019 AUC. The narrower deployment caveat
  survives and is now quantified.

---

## 2. Proposed structure

Current paper: 7 sections, 8 figures, 1 appendix, ~3,900 words.
Revised paper: 9 sections, 8 figures, no appendix, ~8,000–8,500 words.

| § | Section | Change |
|---|---|---|
| — | Abstract | Rewrite |
| 1 | Introduction | Light edit |
| 2 | Background and Related Work | Fix attribution; add §2.2 derivation |
| 3 | The Business Language Layer | Keep; add §3.3 on the lever mechanism |
| 4 | **Data and Methods** | **New** |
| 5 | **Empirical Results** | **New** |
| 6 | The Layer as a Decision Surface | Existing §4, relabelled illustrative |
| 7 | **Synthetic Nigerian Demonstration Environment** | **New** |
| 8 | Discussion | Existing §5, rebuilt on the evidence hierarchy |
| 9 | Limitations and Future Work | Existing §6, strengthened |
| 10 | Conclusion | Rewrite |
| — | References | Fix; add ~6 entries |
| — | Appendix A | **Delete** |

---

## 3. Section by section

### Abstract — rewrite

Current abstract describes a demonstration and never mentions data, model or measurement. It
is the main reason a reviewer concludes the work is illustrative.

Revised arc: translation gap → the BLL and its four functions → **empirical instantiation on
96,171 real Olist orders** → headline numbers (AUC 0.839; calibration 0.309 → 0.051;
−1.73 stars; dispatch 4.4× routing) → the synthetic Nigerian environment, named as synthetic
in the abstract itself → the honest limitation that no user study exists.

Target ~250 words. The limitation stays in the abstract; removing it would be the kind of
quiet overclaim the whole revision is meant to eliminate.

### §1 Introduction — light edit

Keep. Three changes:

- Contribution 3 currently reads "an interface-level demonstration". Change to an empirical
  instantiation, with the dataset named.
- Contribution 4 claims "We show that the layer generalises beyond logistics." The paper shows
  no such thing. Change to: the architecture is domain-agnostic by construction, and we state
  what would be required to test that empirically.
- Add contribution 5: the divergence between attribution magnitude and predictive dependence,
  and what it implies for explanation surfaces that drive action.

### §2 Background and Related Work — fix and extend

**§2 opening, factual correction.** Currently: *"the IXAII framework proposed by Wang et al.
(2019) and extended by Speckmann et al. (2025)"*. IXAII is Speckmann, Nadj & Janiesch (2025).
Wang et al. (2019) proposed a theory-driven user-centric XAI design framework. Rewrite to
attribute each correctly. This is small and disproportionately damaging if left.

**Table 1.** Currently a bitmap of a table from Speckmann et al. Typeset as a real table and
paraphrase the six explanation types in our own words with the citation, which removes the
permission question entirely.

**§2.1** keep as is.

**§2.2 Derivation of the architecture — new, ~400 words.** Reviewer concern A9. Structured as:
what we adopt (IXAII's explanation taxonomy; Wang et al.'s counterfactual layer; He et al.'s
progressive disclosure), what limitation each leaves (all three stop at the model's own
register — none attaches a business consequence, a cost, or a ranked action), and what is
ours (the four-function layer; capability disclosure; costed option ranking; statement-level
provenance). Explicit sentence that no prior work proposed this architecture, and no claim
that it did.

### §3 The Business Language Layer — keep, extend

§3.1 and §3.2 are sound. Figures 3 and 4 keep their place.

**§3.3 Actionability and the lever distinction — new, ~350 words.** The implementation
distinguishes features that are levers from features that merely predict (`is_lever`,
`available_levers`, infeasible options shown rather than hidden). The paper never mentions
this, and the empirical result in §5.4 now justifies it: ranking drivers by attribution
magnitude surfaces seasonality, which no operations lead can act on. Capability disclosure
and the lever distinction stop the layer recommending what the data cannot support.

### §4 Data and Methods — new, ~1,200 words

Written entirely from the implementation and the verified run.

- **4.1 Dataset.** Olist Brazilian E-Commerce, nine tables, 99,441 orders, CC BY-NC-SA 4.0.
  Cleaning summary (geolocation collapsed 1,000,163 → 19,010 zip-prefix centroids; 551
  duplicate reviews removed; 166 impossible carrier pickups nulled). Order-grain construction,
  and why order-item grain inflates to 112,650 rows and double-counts multi-item orders.
- **4.2 Target and censoring.** `is_late_delivery`; delivered orders only; the 30-day
  right-censoring filter (96,470 → 96,171, late rate 8.11% → 8.14%) and why orders promised
  near the data boundary cannot yet be known late.
- **4.3 Leakage controls.** `LEAKAGE_COLUMNS`; the pre/post-dispatch separation and why
  `carrier_pickup_days` belongs only to the latter; `UNBOUNDED_TIME_COUNTERS` and the
  extrapolation argument. This is the section that most directly answers "quantitative
  validation absent", because it shows the evaluation was constructed to be honest rather than
  flattering.
- **4.4 Temporal split.** 2018-04-30. Train 70,883 (8.81% late) / test 25,288 (6.25% late).
  State the base-rate drop and note the test period is therefore harder, not easier.
- **4.5 Model and calibration.** Logistic regression inside an `imblearn` pipeline with SMOTE
  applied inside cross-validation; sigmoid calibration fitted on the last 14,177 training
  orders by date, not at random, so the mapping is learned on data later than the classifier
  saw.
- **4.6 Evaluation protocol.** ROC AUC, average precision, Brier, cost-optimal threshold with
  `cost_fn = 25`, `cost_fp = 4`. State that these are stipulated.
- **4.7 Reproducibility.** Seed 42; 159 tests; environment separation with artefact provenance
  keys; the `test_olist_unchanged` regression guard; `ASSUMPTIONS.md`. Closes with the brief's
  required caveat: reproducibility is not real-world validity.

### §5 Empirical Results — new, ~1,400 words

- **5.1 Predictive performance.** One table: AUC 0.8389, AP 0.3482, Brier 0.0514, threshold
  0.2074, recall 0.5608, precision 0.2585 — against the no-skill baselines (AUC 0.5, AP
  0.0625). The AP-versus-prevalence comparison is what shows 0.348 is a real result on a 6.25%
  problem rather than a weak-looking number.
- **5.2 Calibration.** The isolated ablation: same classifier, sigmoid wrapper only. AUC and
  AP unchanged to six decimals; Brier 0.3085 → 0.0514; threshold 0.8247 → 0.2074. Then the
  argument that matters: the BLL multiplies this probability by an order value, so an
  uncalibrated probability produces a monetary recommendation that is wrong by a factor, not
  by a rounding. Calibration is a precondition for the layer, not a refinement.
- **5.3 Delivery lateness and customer satisfaction.** −1.7291 stars, 95% bootstrap CI
  [−1.7680, −1.6925], n = 95,527, Mann-Whitney p < 1e-300, rank-biserial 0.5538. Stated as an
  association with no controls, explicitly not causal.
- **5.4 What drives lateness — and what can be acted on.** Permutation importance on the test
  set: dispatch 0.520, routing 0.118, basket 0.107, calendar 0.086. Then the SHAP/permutation
  divergence table and its implication for explanation surfaces that drive action. **This is
  the subsection I would push hardest in a rebuttal** — it is a methodological finding, not
  just a result, and it is specific to the paper's own contribution.
- **5.5 Feature-set ablation.** Removing `purchase_year` costs 0.019 AUC and drops recall from
  0.56 to 0.43. Reported with the deployment caveat: the test period falls inside the training
  years, so this evaluation cannot expose the degradation a genuinely unseen year would cause.

### §6 The Layer as a Decision Surface — existing §4, relabelled

Keep Figures 5, 6, 7 and the text. Three changes:

- Open the section by stating plainly that the worked example is an illustrative scenario in a
  Nigerian context, constructed to show the layer's behaviour, and that the empirical results
  are in §5. One sentence, before the first figure, not in a footnote.
- Remove "a live screen" and the browser-chrome URL from Figure 5, or replace the mockup with
  a screenshot of the running dashboard (still open — see §5 below).
- Fix the numerical errors listed in §4 of this plan.

### §7 Synthetic Nigerian Demonstration Environment — new, ~700 words

Follows the brief's §9 and §10 wording pattern.

Why it exists (public demonstration without a licensed dataset; Nigerian context for the
institution's own setting); why Olist remains the empirical basis; how separation is enforced
(separate directories, artefact provenance keys, no combined loader, the regression guard);
what is assumed (corridor friction, the 23 December peak, Eid periods, naira cost parameters);
and the integrity statement — synthetic SHAP values are correctly computed, but the
relationships they recover were put there by the generator, so they are not discoveries about
Nigerian commerce.

Any performance figure from this environment carries the brief's required sentence verbatim.

### §8 Discussion — existing §5, rebuilt

Organised on the evidence hierarchy the brief specifies. Four explicit tiers: what the Olist
data establishes; what the synthetic environment demonstrates; what rests on stipulated cost
assumptions; what remains proposed and unmeasured.

Figure 8 is reworked rather than replaced — "MEASURED AS" becomes "PROPOSED MEASURE", and the
four benefits computable from the implementation today are visually separated from the six
that require a user study.

Add the sensitivity analysis once it is built, and §6's persuasion warning promoted from
Limitations into Discussion, since it is one of the paper's better ideas and is currently
buried.

### §9 Limitations and Future Work — existing §6, strengthened

Keep the honest core. Add: no user study (with the specific design named — within-subjects,
matched orders, SHAP-only versus BLL, measuring time-to-decision, decision quality, and
appropriate reliance when the model is wrong); single domain; single dataset; association not
causation on the review effect; cost parameters stipulated; the calendar-feature deployment
caveat; synthetic environment is not Nigerian evidence.

### §10 Conclusion — rewrite

Four paragraphs on the brief's §18 structure: what Olist established empirically; what the
synthetic environment demonstrated; what the framework contributes; what remains unvalidated.
Remove every instance of "within seconds" and the three unmeasured human-outcome claims in the
final sentence.

### References — fix and extend

- Correct the IXAII attribution in text.
- Qureshi et al. (2025) and Ribeiro et al. (2016) appear nowhere in this manuscript. **Blocked
  pending the reviewer document.** If the reviewers read a draft containing them, I need that
  draft; if they were removed after review, the response letter should say so.
- Add references the new sections require: Olist dataset citation; SMOTE (Chawla et al. 2002);
  Platt (1999) or Niculescu-Mizil & Caruana (2005) for sigmoid calibration; Brier (1950);
  Lundberg & Lee (2017) for SHAP; Ribeiro et al. (2016) for LIME, which the paper cites in
  prose but not in the reference list.

### Appendix A — delete

Six pages duplicating Figures 3–8. One sentence is unique — that the figures share a Nigerian
worked example — and that moves into §6's opening.

---

## 4. Figure plan

| Figure | Action |
|---|---|
| Table 1 | Typeset; paraphrase; fix attribution |
| Fig 1, 2 | Unchanged |
| Fig 3 | Caption gains an illustrative label |
| Fig 4 | "OUTCOME (RESULT)" → "EXPECTED OUTCOME"; soften the causal claim; illustrative label |
| Fig 5 | Remove "live screen" and URL chrome, **or** replace with a real screenshot; fix the "freight is the largest driver" line |
| Fig 6 | Fix the arithmetic (contributions sum to 0.639 against a stated 0.710); reconcile `freight_value` +0.134 against +0.184 in Figs 3, 4, 7; settle the model name |
| Fig 7 | Remove "review score falls 4.6 → 2.8" — no model predicts per-order review score; illustrative label |
| Fig 8 | "MEASURED AS" → "PROPOSED MEASURE"; separate the four computable benefits; retitle 04 and 09 |
| All | Remove the "Yemi · …" footer |
| **New Fig 9** | Permutation importance by group — the dispatch-versus-routing result |
| **New Fig 10** | Calibration: reliability curve before and after, or the Brier/threshold shift |

Figures 9 and 10 are the only new artwork, and both plot measured values. I would generate
them from `reports/` rather than draw them.

---

## 5. Still open

1. **Reviewer document.** Blocks the reference corrections and may contain concerns I have not
   seen.
2. **Figure 5.** Screenshot or relabel. My recommendation remains the screenshot.
3. **Sensitivity analysis.** Approved; script to follow. Results feed §8.
4. **Journal and word limit.** If there is a target venue, its limit changes §4 and §5.

## 6. Requires new experiment

| Item | Why | Effort |
|---|---|---|
| Controlled user study | Every human-outcome claim depends on it. Six of Figure 8's ten benefits. | Large — stays future work |
| Figure 8 instrumentation | Benefits 05, 06, 09, 10 are computable now; computing them moves four cards from proposed to measured | ~1 day |
| Realised-outcome tracking | Benefit 08 needs deployment with outcomes recorded; the prediction store exists and is empty | Deployment-dependent |

---

## Approval

Phase 3 begins on your approval. Reply with changes to any section and I will revise the plan
before writing. The sensitivity-analysis script follows separately so it can run while you
read this.
