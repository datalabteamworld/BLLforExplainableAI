# Data

The raw data is **not** committed to this repository. Download it and place
the nine CSVs in `data/raw/`.

## Source

Brazilian E-Commerce Public Dataset by Olist — Kaggle:
<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>

Licence: CC BY-NC-SA 4.0. Attribution required; non-commercial use only.

## Expected files

| File | Rows | Grain |
|---|---|---|
| `olist_orders_dataset.csv` | 99,441 | one per order |
| `olist_customers_dataset.csv` | 99,441 | one per order (not per person — see below) |
| `olist_order_items_dataset.csv` | 112,650 | one per item |
| `olist_order_payments_dataset.csv` | 103,886 | one per payment instalment |
| `olist_order_reviews_dataset.csv` | 99,224 | roughly one per order |
| `olist_products_dataset.csv` | 32,951 | one per product |
| `olist_sellers_dataset.csv` | 3,095 | one per seller |
| `olist_geolocation_dataset.csv` | 1,000,163 | one per recorded address |
| `product_category_name_translation.csv` | 71 | one per category |

`customer_id` is per *order*; `customer_unique_id` identifies the person.
Confusing the two makes the repeat-purchase rate look like zero.

## Known defects

Verified against the raw release and handled in `src/olist/cleaning.py`:

- **payments** — 3 rows typed `not_defined`, 9 with a zero value,
  2 with zero instalments
- **products** — 610 rows with no category or listing metadata; 4 products
  with `product_weight_g = 0`; 2 with no dimensions; source columns misspell
  "length" as `lenght`
- **sellers** — `seller_city` is free text with 611 variants, including
  `sao paulo`/`são paulo`/`sao  paulo`/`sao paulo - sp`/`sao paulop`,
  one city recorded as a zip code (`04482255`), and `sp` as a city name
- **translation** — carries a UTF-8 BOM, and omits two categories used by
  `products` (`pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos`)
- **geolocation** — ~1M rows for 19k zip prefixes; some coordinates fall
  outside Brazil
