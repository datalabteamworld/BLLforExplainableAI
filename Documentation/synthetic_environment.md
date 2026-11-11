# The two data environments

This project carries two datasets with different evidential status. Confusing
them would be a research-integrity failure, not a bug, so the separation is
enforced in the code rather than left to discipline.

---

## What each one is for

**Empirical Olist** — the Brazilian E-Commerce Public Dataset, roughly 96,000
real orders. Every measured result in the paper comes from here: the AUC, the
calibration finding, the −1.73 star review effect, the dispatch-versus-routing
conclusion. This is the evidence.

**Synthetic Nigerian** — generated data for public demonstration. Shows the
framework working in a Nigerian context, in naira, with no licensed dataset
required. This is **not** evidence.

The distinction in one line:

> Olist answers *does this framework work?*
> Synthetic Nigeria answers *what would it look like here?*

---

## Why not just use synthetic data everywhere

Because it would delete the paper's strongest contribution.

Section 6 of the paper concedes that its figures are illustrative. The Olist
case study closed that gap by replacing them with measurements. If the
deployed demonstration became the evidence base, every one of those numbers
would revert to being a property of a generator — more elaborately produced,
but no more empirical.

So both exist, with different jobs.

---

## How separation is enforced

Four mechanisms, in order of how hard they are to bypass.

**Separate directories.** Each environment owns its raw, processed and model
paths. There is no code path that writes synthetic output into an Olist
location.

```
data/raw/                                    Olist CSVs
data/processed/orders_features.parquet       Olist features
models/model_post_dispatch.joblib            Olist model

data/synthetic_ng/raw/                       generated CSVs
data/synthetic_ng/processed/…                synthetic features
models/synthetic_ng/…                        synthetic model
```

**Environment key in the artefact.** Every model records which environment
trained it. `PredictionService` refuses a mismatch:

```
ValueError: Model was trained in the 'synthetic_ng' environment but is being
loaded for 'olist'. The feature names match, so this would score silently and
produce meaningless results.
```

That check exists because the failure it prevents is invisible — the feature
names are identical across environments, so nothing else would catch it.

**No combined loader.** There is no function anywhere that returns both
datasets, so there is nothing to accidentally concatenate.

**A regression test on the Olist path.** `tests/test_olist_unchanged.py`
fingerprints the Olist feature table — row count, column list, late rate and a
SHA-256 of every numeric value. If any shared-code change alters what the
empirical pipeline produces, it fails.

That test is deliberately brittle. A failure means a published figure no
longer reproduces.

---

## What the two share

The entire analytical machinery, unmodified:

- cleaning rules
- feature engineering and grain handling
- leakage controls, including the pre/post-dispatch split
- temporal splitting and censoring
- the model, SMOTE handling and calibration
- SHAP
- the Business Language Layer, glossary and cost model
- the dashboard and REST API

**Shared architecture is not a shared dataset.** The code is one; the data is
two.

Only three things differ, and all three are supplied by the environment object
rather than branched on in the modelling code: the raw directory, the holiday
calendar, and the geographic bounding box.

---

## Switching

**In the dashboard** — a selector at the top of the sidebar. The banner on
every page states which environment is active, and the synthetic one carries a
disclosure that cannot be dismissed.

**In the API** — the `OLIST_ENVIRONMENT` variable, fixed per process:

```bash
uvicorn olist.api.app:app --reload                       # empirical (default)
OLIST_ENVIRONMENT=synthetic_ng uvicorn olist.api.app:app --reload
```

Fixed per process rather than per request on purpose. Holding both services in
one process would let a mis-set parameter answer an empirical question with a
synthetic model, and that failure would be silent.

Every metadata response carries an `environment` block with the provenance
line and, for synthetic, the disclosure — so a client cannot display a figure
without knowing what produced it.

---

## Building the synthetic environment

```bash
python scripts/build_synthetic_ng.py              # generate, clean, train
python scripts/build_synthetic_ng.py --verify     # confirm determinism
python scripts/build_synthetic_ng.py --orders 25000 --seed 7
```

Takes a few minutes for 80,000 orders. Writes only under `data/synthetic_ng/`
and `models/synthetic_ng/`.

Full generation assumptions: `src/olist/synthetic_ng/ASSUMPTIONS.md`.

---

## How the generator works

Nine Olist-shaped tables, so the existing pipeline consumes them with no
special casing. That schema parity is the whole architectural trick.

Lateness is **not imposed**. Transit time is generated causally from distance,
corridor friction and a noisy shock; lateness is whatever falls out of
comparing arrival against the promise. The timestamps and the target are
therefore consistent by construction.

An earlier version did the reverse — drew the label, then back-filled
timestamps — and silently forced about 20% of supposedly on-time orders late,
because no valid delivery date existed for them. The causal form has no such
failure mode.

Noise is deliberately large: a model cannot exceed roughly 0.85 AUC on this
data. A rule like `distance > 500 → late` would be learnable to 1.0 and would
make the explanation layer look far more decisive than any real deployment is.

**Measured on a 24,000-order build: AUC 0.844, average precision 0.42, Brier
0.061, late rate 7.45%.** Those describe how learnable the *simulated* problem
is. They are not evidence about Nigerian commerce.

---

## Using this in the paper

**Safe to claim:**

- The framework runs end to end on Nigerian-shaped data.
- The architecture is portable — one pipeline, two environments, no
  modification to the modelling path.
- A reproducible public demonstration exists requiring no licensed dataset.

**Not safe to claim:**

- Any performance figure from the synthetic environment as an empirical
  result.
- That dispatch speed dominates Nigerian lateness. It dominates here because
  the generator was written that way.
- That northern corridors are 1.45× slower. That number was typed into a
  configuration file.

If a synthetic figure appears in the paper at all, it needs the label beside
it, not in a footnote.
