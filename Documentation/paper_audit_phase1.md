# Phase 1 Audit — BLL manuscript against the implementation

Date: 6 October 2026
Scope: `BLL-Paper.docx`, `BLL-FIGURES_OR_GRAPHICS.docx`, and the repository at `olist/`.
Status: **audit only.** No manuscript text has been changed. Phase 2 (revision plan) follows;
Phase 3 waits for approval.

---

## 0. The finding that reframes the task

**The attached manuscript contains no empirical results.** Not mislabelled ones — none.

There is no Methods section, no Data section, no evaluation table, no AUC, no Average
Precision, no Brier score, no sample size, no split description, no SHAP results. The word
"Olist" does not appear anywhere in the manuscript. Section 4, titled "Demonstration",
consists entirely of four figures and their prose descriptions. Section 6 states the position
plainly:

> "The demonstration is illustrative, drawn from a single e-commerce logistics scenario, and
> the figures presented use exemplary values to convey the layer's behaviour rather than
> results from a deployed system."

That sentence is honest, and it is also the reviewers' objection in the authors' own words.
The reviewers did not misread the paper. They read it correctly: as submitted, it is a
concept paper with designed illustrations.

So the revision is **not primarily a correction task**. Roughly 20% of the work is fixing
claims that overreach; roughly 80% is *adding* the empirical content that exists in the
implementation but never reached the manuscript. The brief's framing — "make the paper
accurately reflect what has actually been implemented and measured" — is right, but the gap
is one of absence, not of misstatement.

One consequence worth stating early: a reviewer who reads the revised paper will see a
manuscript that has gained an entire empirical section between rounds. That is defensible —
the work was done, it simply was not written up — but the Methods section has to be
unusually complete, because the empirical content has no prior round for the reviewers to
anchor on.

---

## 0.1 A blocking problem with the five headline numbers

The revision brief supplies five figures to present as real empirical results:

| Figure | Found in the repository? | What is actually there |
|---|---|---|
| 96,171 orders | Once, in a comment | `scripts/build_deploy_bundle.py:5`. A second figure, **96,470**, appears at `notebooks/olist_colab.ipynb:138`. The 299-order discrepancy is unexplained. |
| AUC = 0.839 | Once, in a binary artefact | `app/model_post_dispatch.joblib` → `metrics: {'roc_auc': 0.839}`. See the warning below. |
| Average Precision = 0.332 | Once, in a docstring | `src/olist/config.py:143`. In context it is the test AP of a **heavily regularised logistic regression under random CV folds**, cited to justify switching to temporal CV — not the headline model's AP. |
| Brier 0.28 → 0.06 | **Nowhere** | Absent. The nearest pair in the repo is 0.266 vs 0.051 (`README.md:99`, `src/olist/pipeline.py:357`), and that is **logistic regression vs XGBoost — two different models, not uncalibrated vs calibrated.** Presenting it as a calibration improvement would misattribute it. |
| −1.729 stars | **Nowhere at that precision** | The repo carries **1.73** (3 s.f.) at `notebooks/olist_corrected_bll.ipynb:421`. No file in the repo contains a third decimal place. |

Two further problems compound this.

**No evaluation output is stored anywhere.** `reports/` is empty but for a `.gitkeep`. All
three notebooks have zero stored cell outputs (verified: 51, 51 and 26 cells, 0 outputs
each). `.gitignore` excludes `reports/*.csv`, `reports/*.json` and `models/*.joblib` by
design. The code that computes every one of these metrics exists and is correct; the numbers
it produced were read off a screen and typed into prose, and the runs themselves were not
preserved.

**The one model artefact on disk was trained on test fixtures, not Olist.** Its fitted
encoders hold 10 states, 5 cities and 6 product categories — verbatim the constants in
`tests/synthetic.py:18-21`. Real Olist has 27 states, ~4,000 cities and 71 categories. Its
companion `app/sample_orders.parquet` has `order_id` values of the form `ord00000002`;
real Olist order IDs are 32-character hex hashes. Its `trained_at` is the placeholder
`2026-01-01T00:00:00`, identical to the fixture in `tests/test_api.py:44`, rather than the
real timestamp `train.py:199` would write. Its `metrics` dict holds a single hand-written
key. This artefact is the Streamlit Cloud demo bundle and was never meant to be the research
model — but it is the only `.joblib` present, and `app/dashboard.py:736` surfaces its 0.839
to the user as **"model AUC"**.

I want to be careful about what this does and does not mean. It does **not** mean the
empirical work was not done. Your local machine almost certainly holds the Olist CSVs (they
are a licensed Kaggle download that is correctly never committed) and the runs that produced
these numbers. It means that **as the repository stands, not one of the five figures can be
traced to a recorded experiment**, and two of them do not exist in it at all.

The brief forbids inventing model metrics and forbids changing empirical results without
evidence. Writing "AUC = 0.839" into a Results table, sourced from a hand-edited demo
artefact and a number in a docstring, would violate both — not because the number is
necessarily wrong, but because nothing can show it is right. A reviewer in round two who
asks "can you provide the evaluation output?" must receive a file, not a recollection.

**What this needs before Phase 3 can produce a Results section** is one clean run on the real
Olist data that writes its output to disk:

```bash
python -m olist.validate --build          # rebuild the modelling table from the real CSVs
python -m olist.train --scenario post_dispatch --save-metrics
pytest tests/test_olist_unchanged.py      # confirm the pipeline is unchanged
```

`train.py` already writes `metrics_{scenario}.csv`, `cost_curve_{scenario}.csv` and
`run_{scenario}.json`; none of these files currently exists. That run settles the row count,
both Brier scores, AUC, AP, the threshold, and the feature count in one pass — and gives you
artefacts to attach to the rebuttal. The review-score effect needs its notebook cell run once
with the output kept, which also yields the bootstrap CI and the Mann-Whitney p-value that
the paper will need and that currently exist nowhere.

Until that run exists, every empirical cell in the tables below is marked
**REQUIRES VERIFIED RUN** rather than given a value. I have not written any of the five
numbers into a draft.

---

## A. Reviewer correction table

| # | Reviewer concern | Current paper status | Actual evidence available | REAL / SYNTHETIC / ILLUSTRATIVE | Required correction | Requires new experiment? |
|---|---|---|---|---|---|---|
| A1 | Work appears to be an illustrative demonstration only | Accurate. §6 concedes it outright; no empirical section exists | Full empirical pipeline implemented: cleaning, leakage controls, temporal split, censoring, calibration, SHAP, cost-optimal threshold. 159 tests pass | REAL (code) / results **unverified** | Add §4 Data and Methods and §5 Results reporting the Olist evaluation. This is the single largest change | **No new experiment** — but **REQUIRES VERIFIED RUN** to recover the numbers |
| A2 | Quantitative validation absent | No dataset, split, preprocessing, model, protocol or metric is described anywhere | All six are fully specified in code: `config.py` (`SplitConfig`, `LEAKAGE_COLUMNS`, `POST_DISPATCH_FEATURES`, `UNBOUNDED_TIME_COUNTERS`), `features.py`, `history.py` (30-day censoring), `pipeline.py` (SMOTE inside `imblearn.Pipeline`, `TimeSeriesSplit`), `evaluate.py` | REAL | Write a Methods section from the code, and a results table with measured values only | No |
| A3 | Financial impact not substantiated | Figures show naira amounts with no basis stated; §4 text calls them "monetised business impact" | `BusinessCostModel` (`bll/brief.py:40-72`) with explicit defaults `refund_share=0.42`, `churn_share=0.46`, `support_cost=12.0`, `intervention_cost=4.0`. Its own docstring says: *"These are **assumptions, not measurements**"* | ASSUMPTION (the parameters) over REAL (the probability) | State all four parameters in a table; relabel every monetary figure "estimated financial impact under the stated assumptions"; add the sensitivity analysis | **Yes — sensitivity analysis is not implemented.** See §D1 |
| A4 | Review-score / lateness relationship | Not mentioned in the manuscript at all | Code exists: a two-group mean difference with a 2,000-sample bootstrap percentile CI and a one-sided Mann-Whitney U test (`olist_corrected_bll.ipynb:394-412`). **No output stored** | REAL (design) / **unverified** (value) | Report as an **association** with its CI and p-value, on the full modelling table, with no causal claim. The repo already hedges correctly (`README.md:387`: "No causal claim") | **REQUIRES VERIFIED RUN** |
| A5 | Synthetic Nigerian environment unexplained | Not mentioned in the manuscript at all | Fully built and documented: `environments.py` registry, `synthetic_ng/generate.py`, `ASSUMPTIONS.md`, 23 dedicated tests, measured at AUC 0.844 on a 24k build | SYNTHETIC | Add §6 "Synthetic Nigerian Demonstration Environment" using the brief's §9 wording pattern verbatim | No |
| A6 | Synthetic relationships may be generator artefacts | Not mentioned | `ASSUMPTIONS.md` already contains a "Conclusions that must NOT be drawn" section naming the corridor-friction multiplier and the dispatch effect as typed-in constants | SYNTHETIC | State explicitly that synthetic SHAP values are correctly computed but recover injected relationships, so they are not discoveries | No |
| A7 | No controlled user study | §6 correctly states this. §5 and Figure 8 partially undercut it | **Definitively none.** Exhaustive grep for study, participant, consent, telemetry, think-aloud, SUS, NASA-TLX, latency, dwell, session logging found nothing. The only log table (`api/store.py`) records model predictions, not human behaviour, and holds 0 rows | — (absent) | Keep the §6 limitation; strip outcome language from §5 and Figure 8 per A8 | **Yes — REQUIRES NEW EXPERIMENT.** See §D2 |
| A8 | Figure 8 benefits read as measured | Each card is headed "MEASURED AS" in capitals; titles are outcome claims ("Improved Decision Support", "Enhanced Transparency & Trust") | 4 of 10 are computable from the implementation today without human participants; 6 require a user study | Mixed — see §B6 for the per-benefit split | Change "MEASURED AS" → "PROPOSED MEASURE"; restate outcome titles as design intent per brief §12 | Partly — 6 of 10 require the study |
| A9 | Architecture derivation from prior work | §2 describes Wang et al. and He et al. but never says what was adopted, what was rejected, or what is new | Three mechanisms in the implementation are genuinely novel relative to the cited work: capability disclosure (infeasible levers shown, not hidden — `brief.py:300`), the costed option ranking (`brief.py:339`), and provenance from every business sentence back to its attribution | REAL (architecture) | Add §3.3 "Derivation" naming adopted concepts, the limitation that motivated each extension, and the specific delta | No |
| A10 | Reproducibility | Not mentioned | `RANDOM_STATE = 42`; fixed-seed generator with a `--verify` determinism check; environment registry with artefact provenance keys; `test_olist_unchanged.py` regression guard; 159 tests (**count verified by collection**); `ASSUMPTIONS.md` | REAL | State these accurately, and add the brief's caveat that reproducibility ≠ real-world validity | No |
| A11 | References — Qureshi et al. (2025), Ribeiro et al. (2016) | **Neither appears in the manuscript.** The reference list has exactly three entries: He et al. (2025), Speckmann et al. (2025), Wang et al. (2019) | — | — | **Resolve the version question first.** Either the reviewers read a different draft, or these citations were removed after review. I cannot correct references that are not there | No — but needs your input |
| A12 | IXAII attribution | §2 says IXAII was "proposed by Wang et al. (2019) and extended by Speckmann et al. (2025)" | This is backwards. IXAII is Speckmann, Nadj & Janiesch (2025). Wang et al. (2019) proposed a theory-driven user-centric XAI framework — a different contribution | — | Correct the attribution. This is the kind of error that costs credibility disproportionately | No |
| A13 | Appendices | Appendix A reproduces Figures 3–8 in full, duplicating ~6 pages | Nothing in the appendix is unique except one sentence: that the figures share a worked example in "Nigerian e-commerce fulfilment" | — | Remove Appendix A. Migrate that one disclosure into the figure captions, where it matters more | No |
| A14 | Table 1 | Reproduced as a **bitmap image** of a table from Speckmann et al. (2025) | — | — | Typeset it as a real table, and either obtain permission or paraphrase the six explanation types in the authors' own words with a citation | No |

---

## B. Figure audit

Numbering note: the manuscript labels these Figures 3–8; the figures document labels the same
six images Figures 1–6. **The two documents disagree about every figure number.** Fix before
submission. Below I use the manuscript's numbering, with the figures-document number in
brackets.

| Figure | Actual source | REAL / SYNTHETIC / ILLUSTRATIVE | Caption correct? | Problem | Correction |
|---|---|---|---|---|---|
| **Table 1** | Bitmap reproduction from Speckmann et al. (2025) | Third-party | Attribution note present | Reproduced as an image; copyright status unaddressed; main text misattributes IXAII to Wang et al. | Typeset; secure permission or paraphrase; fix the attribution in §2 |
| **Fig 1** (literature) | Wang et al. (2019), CHI | Third-party, REAL in origin | Yes, properly credited | None | Leave unchanged |
| **Fig 2** (literature) | He et al. (2025), IUI | Third-party, REAL in origin | Yes, properly credited | None | Leave unchanged |
| **Fig 3** [FD 1] "The Explainability Gap" | Hand-authored illustration | **ILLUSTRATIVE** | Caption does not say so | Shows a specific SHAP table (`freight_value +0.184`, `distance_km +0.121`, `seller_delay_days +0.098`, `order_hour +0.036`, `product_weight_g −0.042`, base 0.310) for "Order #7741" with "model: XGBoost". No such order, model or feature set exists in the implementation. A reviewer may read it as a real output. Note the arithmetic *does* close: 0.310 + 0.397 = 0.707 ≈ 0.71 | Add to caption: *"Illustrative example constructed to show the form of a technical explanation; values are exemplary and not drawn from the empirical model."* Figure itself can stay — it is making a conceptual point |
| **Fig 4** [FD 2] "The Business Language Layer" | Hand-authored architecture diagram | **ILLUSTRATIVE** (architecture is REAL and implemented) | Partly — AI-assisted-design note is present and correct | (a) The fifth panel is headed **"OUTCOME (RESULT)"** and reads "SLA met. Customer retained. Decision on record." — a hypothetical typeset as a recorded result. (b) "Better decisions lead to measurable outcomes" is an unvalidated causal claim. (c) `SHAP(freight_value) = +0.184` — see F1 below | Relabel the fifth panel **"EXPECTED OUTCOME"**; soften the causal claim to design intent; add *"The architecture shown is implemented; the worked example is illustrative."* |
| **Fig 5** [FD 3] "BLL Applied… Delivery-Delay Risk" | Hand-authored mockup | **ILLUSTRATIVE** | **No — materially misleading** | The subtitle says *"A live screen for order #7741"* and the browser chrome shows `app.datalab.com.ng/ops/orders/7741/risk`. **This is not a screenshot of a running system.** A real dashboard exists and could be screenshotted; this is a designed mockup of one that implies a deployed Nigerian production service. This is the single highest-risk item in the paper. Also: "High freight cost… the single largest driver" contradicts the implementation, whose actual top drivers are `working_days_in_window`, `estimated_delivery_days`, `non_working_days_in_window`, `carrier_pickup_days` | Either (a) **replace with a genuine screenshot** of the running dashboard — strongly preferred, and it would convert the paper's weakest figure into evidence — or (b) remove "live screen", remove the URL chrome, and caption it *"Illustrative interface design; not a screenshot of a deployed system."* |
| **Fig 6** [FD 4] "Translation in Practice" | Hand-authored | **ILLUSTRATIVE** | Footer says *"figures are exemplary"* — the only figure that labels itself, but in 7-point type, and the paper's own caption omits it | (a) **The SHAP arithmetic does not close**: 0.310 + 0.134 + 0.121 + 0.098 + 0.036 − 0.042 − 0.018 = **0.639**, but the figure states f(x) = 0.710. A 0.071 discrepancy. (b) `freight_value` is **+0.134** here but **+0.184** in Figures 3, 4 and 7 — same order, same model. (c) Additive SHAP notation is wrong for the implemented system: SHAP runs on the inner uncalibrated classifier (`bll/explain.py:55-67`), so attributions rank drivers but do not additively decompose the calibrated probability. (d) Model named "XGBoost"; the deployed model is logistic regression | Fix the arithmetic and the +0.134/+0.184 conflict; promote "illustrative" into the paper caption; add a sentence in the text on why attributions are ranked rather than summed under calibration; settle the model name |
| **Fig 7** [FD 5] "From Why to What Now" | Hand-authored | **ILLUSTRATIVE** | No label at all | (a) *"review score falls 4.6 → 2.8"* — a per-order point prediction of review score. **No model in the repository predicts review score.** The −1.73 figure is a group mean difference, not a per-order forecast, and 4.6→2.8 is −1.8, which matches neither. (b) "NGN 48,000 of order value is exposed" — assumption presented as fact | Remove or replace the 4.6 → 2.8 claim — it is the clearest instance of an unsupported number in the paper. Add an illustrative label |
| **Fig 8** [FD 6] "What the Layer Buys You" | Hand-authored | **ILLUSTRATIVE / PROPOSED** | Header says "how it would be measured" (conditional, good) but each card says **"MEASURED AS"** (past tense, bad) | Ten benefits, of which: **computable today** — 05 "share of explanations carrying a feasible action" (`brief.py:300-317` already marks infeasible options), 06 "option coverage" (partial), 09 "trace completeness" (every brief sentence carries provenance), 10 "protected-attribute disclosure rate". **Requires a user study** — 01, 02, 03, 04, 07. **Requires deployment with realised outcomes** — 08 "predicted naira variance vs. realised outcome" (the prediction store exists but holds 0 rows) | Change "MEASURED AS" → "PROPOSED MEASURE" throughout; retitle 04 → "Designed to Support Decision-Making" and 09 → "Traceability and Disclosure"; consider splitting the four computable benefits into a separate band so the honest ones are not tarred with the aspirational ones |
| **All figures** | — | — | — | Footer reads "Yemi · Business Language Layer research programme · 2026" — an informal personal name in a publication figure | Replace with the institutional attribution or remove |

---

## C. Paper inconsistencies

### Where the paper contradicts the implementation

**C1.** Figures name the model **XGBoost** (Figs 3, 6). The deployed artefact is
`logistic_regression`. The repo reports XGBoost at 0.862 AUC as an alternative
(`README.md:167`) but it is not the shipped model. Pick one and use it everywhere.

**C2.** Figure 5 asserts *"High freight cost on this lane — the single largest driver."* The
implementation's measured top drivers are `working_days_in_window`,
`estimated_delivery_days`, `non_working_days_in_window` and `carrier_pickup_days`.
Figures 3, 4, 6 and 7 all repeat freight as the top driver. Five figures agree with each
other and disagree with the model.

**C3.** Figure 7 claims a per-order review-score prediction (4.6 → 2.8). No such model
exists. The repository's review-score analysis is a two-group mean difference computed
*before* the train/test split, on the full modelling table.

**C4.** Figure 5's "Slow recent dispatch" is shown as a headline driver. From earlier
analysis, that feature contributes roughly **+0.002 AUC** — real, but far too small to carry
the weight the figure gives it.

**C5.** The paper never mentions calibration, yet calibration is the implementation's most
reviewer-relevant contribution: it is what makes the monetised expected-loss arithmetic
legitimate at all. An uncalibrated probability multiplied by an order value is not an
expected loss. This is a missing argument that would *strengthen* the paper.

### Where an illustrative claim looks measured

**C6.** "A live screen for order #7741" (Fig 5) — the most serious instance. See B/Fig 5.

**C7.** "OUTCOME (RESULT): SLA met. Customer retained." (Fig 4) — hypothetical as record.

**C8.** "MEASURED AS" on all ten Figure 8 cards.

**C9.** §4.1: *"a decision an operations lead can act on within seconds."* Repeated in §4.3,
§7 and the abstract. **"Within seconds" is a human-performance claim with no user study
behind it.** It appears five times. Each is a decision-latency assertion of exactly the kind
brief §11 forbids.

**C10.** §5: *"This improves the design of effective XAI interfaces, enabling users to… reach
personalised decisions with greater confidence. In turn, this can reduce scepticism toward,
and over-reliance on, AI-assisted decision-making."* Four unmeasured human-outcome claims in
two sentences. "Can reduce" is doing a great deal of hedging work.

### Where an assumption is presented as fact

**C11.** All naira amounts in Figures 4–7 derive from `refund_share=0.42`, `churn_share=0.46`,
`support_cost`, `intervention_cost`. None is stated in the paper. The implementation is
scrupulous about this and the paper is silent — the integrity discipline exists in the code
and was lost in the writing.

**C12.** Figure 5: *"Upgrading costs just NGN 2,100 and prevents the delay."* **Assumes
intervention efficacy of 1.0.** Every net-saving figure in the paper inherits this. No
evidence supports it; it is not even an assumption the code makes explicit.

**C13.** Figure 5 states *"there's a 71% chance this NGN 48,000 order is lost to a late
delivery."* Conflates P(late) with P(total order value lost). Lateness does not destroy order
value; it triggers a refund share, a churn share and a support cost — which is exactly what
`BusinessCostModel` encodes. The figure's own prose contradicts the model it was generated
from.

**C14.** **Exposure and order value are conflated.** Figure 5 presents NGN 20,000 + 22,000 +
6,000 = 48,000 as "revenue at stake"; Figure 6 calls the same 48,000 the *order value* and
computes expected loss as `p(late) × order value`. In the implementation these are different
quantities: exposure = 0.88 × V + support cost. If V = 48,000, exposure is 48,240. The two
figures agree only because the components were rounded.

### Where an empirical result is missing

**C15.** No sample size. **C16.** No train/test split description — the temporal split at
2018-04-30 and the 30-day censoring filter are among the paper's strongest methodological
choices and go unmentioned. **C17.** No leakage-control discussion, despite
`LEAKAGE_COLUMNS`, the pre/post-dispatch separation and `UNBOUNDED_TIME_COUNTERS` being
substantive contributions. **C18.** No AUC, AP, Brier or threshold. **C19.** No SHAP results
from the real model. **C20.** No mention of the synthetic Nigerian environment. **C21.** No
reproducibility section. **C22.** No baseline comparison of any kind — not even a majority-
class or always-predict-late baseline, which costs nothing to compute and which reviewers
routinely ask for.

### Where a caption is ambiguous

**C23.** Figures 3, 4, 5 and 7 carry no evidential label. Figure 6's is in the image footer,
not the caption. **C24.** The manuscript and figures document disagree on every figure
number. **C25.** Appendix A says the figures share a worked example in "Nigerian e-commerce
fulfilment" — the only place the Nigerian framing is disclosed, and it is in an appendix the
reviewers want deleted.

### Where a conclusion is stronger than the evidence

**C26.** Abstract: *"The demonstration shows how the BLL converts a technically faithful but
operationally silent explanation into guidance an operations lead can act on within
seconds."* No study supports "within seconds".

**C27.** §7: *"a monetised and actionable decision brief that a non-technical decision-maker
can act on within seconds."* Same claim, conclusion position.

**C28.** §1.1 contribution 4: *"We show that the layer generalises beyond logistics to other
high-stakes domains."* **The paper shows no such thing.** It asserts domain-agnosticism;
nothing outside logistics is instantiated. "We argue that the architecture is
domain-agnostic, and identify what would be required to test this" is defensible.

**C29.** §7: *"making Explainable AI more relatable, actionable, and trustworthy for the
people who must act on it"* — three unmeasured human outcomes in the final sentence.

**C30.** Abstract describes the case study as a demonstration "in which a late-delivery risk
prediction is translated into a monetised, actionable decision brief" without disclosing that
the figures are exemplary. The abstract is the only part many reviewers read closely.

---

## D. Items that require new work

### D1. Sensitivity analysis — REQUIRES NEW EXPERIMENT (small)

Brief §8 asks for one if supported. It is not implemented; `brief.py:48` recommends one
without providing it. This is the cheapest high-value addition in the whole list: the
recommendation flips between options at some threshold of `refund_share`, `churn_share` and
`intervention_cost`, and a one-way sweep over each — about 40 lines over the existing
`DecisionBrief` — would show *how robust the recommendation is to the assumptions it rests
on*. That directly answers the reviewer's financial-impact objection in a way no amount of
rewording can.

I can build this. It needs no new data.

### D2. Controlled user study — REQUIRES NEW EXPERIMENT (large)

Definitively absent, and not something to work around. Every claim in C9, C10, C26, C27 and
C29 depends on it. Until it exists the paper must say these benefits are intended and
unvalidated.

What it would take, so the Future Work section can be specific rather than vague: a
within-subjects design, operations and analytics staff, each participant seeing matched
orders under SHAP-only and BLL conditions, measuring time-to-decision, decision quality
against a held-out ground truth, and appropriate reliance (does the participant correctly
*reject* the recommendation when the model is wrong?). That last measure matters most, given
§6's own warning that a fluent costed recommendation is more persuasive than a raw
attribution — including when it is wrong. A study that measured only speed and confidence
would risk validating the persuasion rather than the decision quality.

### D3. Figure 8 instrumentation — computable now, not yet built

Benefits 05, 06, 09 and 10 can be measured from the implementation without any human
participant. Computing them would convert four of the ten cards from "proposed" to
"measured", which is a genuine strengthening rather than a rewording. Roughly a day's work.

### D4. One verified Olist run — REQUIRES VERIFIED RUN

See §0.1. Blocking for Phase 3's Results section.

---

## E. What is already right

Worth recording, because the revision should not disturb it:

- §6 is honest about the illustrative status. Most papers at this stage are not.
- §6's observation that a fluent costed recommendation is *more persuasive than a raw
  attribution, including when the model is wrong*, is the sharpest paragraph in the paper and
  anticipates a serious objection before a reviewer raises it. It should be promoted, not
  buried.
- The AI-assisted-design disclosure (§3 and Figure 4's note) is handled correctly.
- Figures 1 and 2 are properly attributed.
- The conceptual contribution — the translation gap, the four functions, the layer-above-XAI
  positioning — is coherent, and nothing in this audit undermines it. The problem is
  evidential, not conceptual.
- The implementation is in better shape than the paper. The honesty discipline the brief asks
  for already exists in the code (`BusinessCostModel`'s docstring, `ASSUMPTIONS.md`'s
  "Conclusions that must NOT be drawn", the environment provenance checks). The revision's
  job is largely to carry that discipline into the manuscript.

---

## F. Numerical errors, collected

| ID | Location | Error |
|---|---|---|
| F1 | Fig 6 vs Figs 3, 4, 7 | `freight_value` = **+0.134** vs **+0.184** for the same order and model |
| F2 | Fig 6 | SHAP contributions sum to 0.639, figure states f(x) = 0.710 — a 0.071 gap |
| F3 | Figs 5, 6 | Exposure (48,240 under the stated shares) and order value (48,000) conflated |
| F4 | Fig 7 | 4.6 → 2.8 is a −1.8 star drop; the repo's figure is −1.73, and it is a group mean, not a per-order prediction |
| F5 | Repo | 96,171 vs 96,470 orders, unexplained |
| F6 | Brief vs repo | −1.729 vs 1.73; no third decimal exists in any file |
| F7 | Brief vs repo | Brier 0.28 → 0.06 appears nowhere; nearest is 0.266 vs 0.051, a different comparison entirely |

---

## Phase 2 gate

Three things are needed before the revision plan can be finalised:

1. **The verified Olist run** (§0.1), or confirmation that the metrics exist on your machine
   with output files that can be attached.
2. **The reference question** (A11): did the reviewers read a draft containing Qureshi et al.
   (2025) and Ribeiro et al. (2016)? If so I need that draft.
3. **A decision on Figure 5**: replace the mockup with a real screenshot of the running
   dashboard, or relabel it as an interface design. I recommend the screenshot — it is the
   cheapest way to turn the paper's most exposed figure into actual evidence.
