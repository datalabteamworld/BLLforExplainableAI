# Olist Brazilian E-Commerce Dataset

Reference for the dataset used in the late-delivery risk case study
(Oyeleke et al., 2026, Section 4).

**The CSVs are not committed to this repository.** They total ~120 MB and
GitHub rejects files above 100 MB. Download them from Kaggle and place them in
`Prototype Codebase/olist/data/raw/`.

---

## Source

**Brazilian E-Commerce Public Dataset by Olist**
<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>

Licence: **CC BY-NC-SA 4.0** — attribution required, non-commercial use only,
derivatives under the same licence.

The non-commercial clause matters for a published paper: the dataset may be
used for the research demonstration, but a commercial deployment of anything
trained on it would need a different data source.

Coverage: ~100,000 orders placed between September 2016 and October 2018 across
Brazilian marketplaces, anonymised by Olist.

---

## Expected files

| File | Rows | Grain |
|---|---|---|
| `olist_orders_dataset.csv` | 99,441 | one per order |
| `olist_customers_dataset.csv` | 99,441 | one per order (see note) |
| `olist_order_items_dataset.csv` | 112,650 | one per item |
| `olist_order_payments_dataset.csv` | 103,886 | one per payment instalment |
| `olist_order_reviews_dataset.csv` | 99,224 | roughly one per order |
| `olist_products_dataset.csv` | 32,951 | one per product |
| `olist_sellers_dataset.csv` | 3,095 | one per seller |
| `olist_geolocation_dataset.csv` | 1,000,163 | one per recorded address |
| `product_category_name_translation.csv` | 71 | one per category |

`python -m olist.validate` checks all nine are present, that schemas match, and
that row counts fall within 2% of the figures above — a truncated download or a
file re-saved by Excel then fails in seconds rather than mid-pipeline.

---

## Three traps in this dataset

**1. `customer_id` is not a person.** It is issued per *order*.
`customer_unique_id` identifies the individual. Confusing the two makes the
repeat-purchase rate look like zero and breaks any lifetime-value analysis.

**2. The tables are at three different grains.** Payments are per instalment,
items per line, reviews roughly per order. Joining them without aggregating
first silently multiplies rows: the 99,441 orders become 113,425 order-item
rows, and every item of a multi-item order then carries the same order-level
label.

**3. `geolocation` is not a lookup table.** It has ~1M rows for ~19k zip
prefixes, one per recorded address. Joining it directly causes a large fan-out;
it must be collapsed to zip centroids first.

---

## Known defects

Verified against the raw release and corrected in
`Prototype Codebase/olist/src/olist/cleaning.py`, each rule pinned by a test.

### `olist_order_payments_dataset.csv`
- 3 rows with `payment_type = "not_defined"`, all zero-value
- 9 rows with `payment_value = 0`
- 2 rows with `payment_installments = 0`

### `olist_products_dataset.csv`
- **610 rows** with no category, name length, description length or photo
  count — one block of unlisted products, not scattered nulls
- 2 rows with no weight or dimensions
- 4 products with `product_weight_g = 0` (physically impossible; all
  `cama_mesa_banho`, all 30×25×30 cm)
- Source column names misspell "length": `product_name_lenght`,
  `product_description_lenght`

### `olist_sellers_dataset.csv`
`seller_city` is free text, 611 distinct values. Lowercasing alone collapses
none of them, because the variation is structural:

- São Paulo appears as `sao paulo`, `são paulo`, `sao  paulo` (double space),
  `sao paulo - sp`, `sao paulo sp`, `sao paulo / sao paulo`, `sao paulop`
- Rio de Janeiro appears as `rio de janeiro`,
  `rio de janeiro / rio de janeiro`, `rio de janeiro \rio de janeiro`,
  `rio de janeiro, rio de janeiro, brasil`
- One city is recorded as a zip code: `04482255`
- Four sellers have city `sp` — a state code, not a city
- `sbc/sp` abbreviates São Bernardo do Campo
- `santa barbara d´oeste` uses an acute accent as an apostrophe

**40 rows carry malformed city strings**; normalisation reduces 611 distinct
values to 579, with 6 unrecoverable.

### `product_category_name_translation.csv`
- **UTF-8 BOM on the header line.** Read carelessly, the first column becomes
  `\ufeffproduct_category_name` and the merge silently produces all-NaN without
  raising.
- Two categories used by `products` have no translation row: `pc_gamer` and
  `portateis_cozinha_e_preparadores_de_alimentos`

### `olist_geolocation_dataset.csv`
- ~1M rows for ~19k zip prefixes
- 42 coordinates fall outside Brazil's bounding box

### `olist_orders_dataset.csv`
- 166 orders record `order_delivered_carrier_date` **before**
  `order_purchase_timestamp`

---

## After cleaning and merging

| Metric | Value |
|---|---|
| Orders in modelling table | 96,171 |
| Columns | 89 (68 usable features) |
| Late-delivery rate | 8.14% |
| Purchase date range | 2016-09-15 to 2018-08-29 |

Purchases stop in August rather than October because an order enters the table
only once **delivered**; orders bought in September were still in transit at
the data cutoff.

---

## Reproducing the modelling table

```bash
cd "Prototype Codebase/olist"
pip install -e ".[dev]"
python -m olist.validate           # preflight
python -m olist.validate --build   # clean, merge, write parquet
```

Writes `data/processed/orders_features.parquet`. Neither the raw CSVs nor this
parquet are committed — both are regenerable in a few minutes.
