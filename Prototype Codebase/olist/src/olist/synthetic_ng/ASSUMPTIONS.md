# Synthetic Nigerian environment — generation assumptions

Every relationship in this dataset was written by hand. This document records
which ones, so that nothing a model finds here can be mistaken for a discovery.

---

## 1. What is synthetic, and what is not

| Element | Status |
|---|---|
| City names, states, geopolitical zones | **Real** — checkable against any atlas |
| City coordinates | **Real** — distances reproduce actual straight-line distances |
| Public holidays, including both Eids | **Real** — from `workalendar.africa.Nigeria` |
| Customers, sellers, products, orders | **Synthetic** |
| Prices, freight, order values | **Synthetic** — plausible 2024 naira levels, not surveyed |
| Delivery times, lateness | **Synthetic** |
| Seller performance, review scores | **Synthetic** |
| Corridor difficulty | **Assumed** — see §4 |

Nothing in this dataset derives from the Brazilian Olist release. No Olist
record was copied, translated or currency-converted. The two datasets share a
*schema*, which is what lets one pipeline process both, and nothing else.

---

## 2. Why this environment exists

The paper's empirical evidence comes from the Olist dataset: ~96,000 real
Brazilian orders. That is where the measured AUC, Brier score, calibration
finding and review-score effect come from, and it stays that way.

This environment exists so the framework can be demonstrated publicly without
requiring a Kaggle licence, and so the demonstration speaks in a Nigerian
context relevant to DataLab's market. It is a **demonstration**, not evidence.

---

## 3. How lateness is generated

Lateness is **not imposed**. Transit time is generated causally, and lateness
is whatever falls out of comparing arrival against the promise:

```
transit_days = (1.2 + 0.0045 × distance_km) × corridor_friction × exp(0.42 × shock)
delivered    = carrier_handover + transit_days
is_late      = delivered > promised_date
```

An earlier version did the opposite — drew the label first, then back-filled
timestamps. That produced a contradiction: an order marked on-time could not
always be given a delivery date both after carrier handover and before the
promise, and the clamp silently forced about 20% of them late. The causal
form has no such failure mode, and it is also how the Olist target is defined.

### The shock term

```
shock = Σ βᵢ · z(featureᵢ) + ε,    ε ~ Normal(0, 1.35)
```

| Driver | β | Direction |
|---|---|---|
| Corridor friction | 0.55 | Harder corridor → slower |
| Distance | 0.40 | Further → slower |
| Non-working days in window | 0.30 | More → slower |
| Peak load (December, ember months) | 0.35 | Peak → slower |
| Items in order | 0.22 | More items → slower |
| Parcel weight | 0.18 | Heavier → slower |

Dispatch lag enters separately and more strongly: it shifts
`carrier_handover` directly, so a slow seller pushes the whole delivery later.
This mirrors the operational finding measured on Olist, **which is why it
was built in** — and therefore it is not independent confirmation of that
finding.

### Noise is deliberate

`noise_sd = 1.35` is large relative to the coefficients. A model cannot exceed
roughly 0.85 AUC on this data. That is the point: a rule such as
`distance > 500 → late` would be learnable to AUC 1.0 and would make the
explanation layer look far more decisive than any real deployment is.

Measured on a 24,000-order build: **ROC AUC 0.844, average precision 0.42,
Brier 0.061, late rate 7.45%.** These describe how learnable the simulated
problem is. Nothing more.

---

## 4. Nigerian context

### Geography

Eighteen commercial centres across all six geopolitical zones. Seller and
customer weights differ deliberately — Aba and Onitsha ship far more than they
receive; Abuja receives more than it ships.

Lagos carries 30% of seller volume rather than the 70%+ that a naive model of
Nigerian e-commerce would assume. Over-concentrating in one origin produces a
dataset with almost no route variation to learn from.

### Corridor friction — an assumption, not a measurement

| Zone | Friction |
|---|---|
| South West | 1.00 |
| South East | 1.08 |
| South South | 1.12 |
| North Central | 1.15 |
| North West | 1.28 |
| North East | 1.45 |

Plus 1.18× for crossing zones, and 1.10× for corridors crossing the Niger or
Benue.

These encode assumptions about road quality, carrier density and checkpoint
delay. **They are not measurements of Nigerian road conditions.** Any SHAP
result showing corridor friction as important is recovering this table.

### Calendar

Nigerian public holidays, including both Eids as two-day observances extended
by one day for backlog clearance. December 23 is the commercial peak rather
than Black Friday. October–December carry elevated load.

**Caveat:** `workalendar` warns that its Islamic date computation is
approximate. Eid dates may be a day out from those gazetted by the Nigerian
government, which announces on sighting of the moon. Acceptable for a
demonstration; not acceptable for a production system serving Nigerian
operations.

### Money

Prices are plausible 2024 Nigerian online-retail levels — phones ₦185,000,
groceries ₦8,500, generators ₦420,000. They are assumptions informed by
general familiarity with the market, **not a price survey**.

Freight: `₦700 + 0.95 × distance_km + 0.085 × weight_g`, with noise. Heavier
and more distant costs more, as a reader would expect.

---

## 5. Injected data defects

Mirroring the real Olist release, so the cleaning module is exercised rather
than bypassed:

| Defect | Rate |
|---|---|
| Inconsistent city spellings | 6% of customers |
| Missing product category | 2.5% of products |
| Impossible weight (0 g) | 0.04% of products |
| Carrier handover before purchase | 0.15% of orders |
| `not_defined` payment type | 3 rows |
| Zero payment value | 9 rows |
| Zero instalments | 2 rows |
| Coordinates outside Nigeria | ~0.2% of geolocation rows |
| Untranslated categories | 2 of 14 |

---

## 6. Reproducibility

```bash
python scripts/build_synthetic_ng.py                 # seed 42, 80,000 orders
python scripts/build_synthetic_ng.py --verify        # confirm determinism
```

Everything is controlled by `GeneratorConfig`, a frozen dataclass. Same seed,
same dataset, on any machine.

---

## 7. Conclusions that must NOT be drawn

Stated explicitly, because a fluent interface over simulated data is exactly
the failure mode the BLL paper warns about in Section 6.

**Do not conclude** that the framework achieves any particular accuracy on
Nigerian delivery data. It has never been tested on Nigerian delivery data.

**Do not conclude** that dispatch speed is the dominant driver of lateness in
Nigerian e-commerce. It dominates here because §3 made it dominate.

**Do not conclude** that northern corridors are 1.45× slower. That number is
in §4 because it was typed there.

**Do not conclude** that a late delivery costs 1.7 stars in Nigeria. That
effect size was measured on Olist and injected here so the cost model has
something to rest on.

**Do not report** any figure from this environment as an empirical result, or
place one in a table alongside Olist measurements without labelling it.

**Do not combine** these records with Olist records for any evaluation.

---

## 8. What may legitimately be claimed

- The Business Language Layer runs end to end on Nigerian-shaped data.
- The architecture is portable: the same pipeline, model, SHAP and BLL code
  serve both environments with no modification to the modelling path.
- A reproducible public demonstration exists that needs no licensed dataset.
- The capability-disclosure mechanism works identically in both environments:
  courier, warehouse, weather and seller rating are absent from both, and the
  app says so rather than inventing them.
