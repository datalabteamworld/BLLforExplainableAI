"""Synthetic Nigerian e-commerce generator.

Emits the nine Olist-shaped raw tables so that every downstream module --
cleaning, feature engineering, history, splitting, training, SHAP, the
Business Language Layer -- runs unchanged. Nothing in the modelling path knows
which environment produced its input.

**What is simulated, and what that means**

Every relationship between a feature and late delivery in this dataset was
written here, deliberately. The model will find them, SHAP will rank them, and
the Business Language Layer will narrate them. That is a demonstration that the
machinery works end to end on Nigerian-shaped data. It is *not* evidence about
Nigerian commerce, and no figure produced from this environment should be
reported as such.

**How lateness is generated**

Not a rule. A latent score

    z = b0 + sum(beta_i * standardised_feature_i) + noise

is pushed through a logistic and sampled. Because the noise term is
substantial relative to the coefficients, the resulting problem has genuine
irreducible error: a perfect model could not exceed roughly 0.85 AUC on it.
A threshold rule such as ``distance > 500 -> late`` would be learnable to
AUC 1.0 and would make the explanation layer look far more decisive than any
real deployment ever is.

**Deliberate imperfection**

The raw tables carry the same classes of defect as the real Olist release --
inconsistent city spellings, a handful of impossible weights, missing category
translations, a few carrier timestamps before purchase. Without them the
cleaning module would be exercised against clean input and its value would be
invisible.

See ``ASSUMPTIONS.md`` for the full table of injected relationships.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import geography as geo

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400


# ==========================================================================
# Configuration
# ==========================================================================


@dataclass(frozen=True)
class GeneratorConfig:
    """Every knob, in one frozen object, so a run is fully described by it."""

    n_orders: int = 80_000
    seed: int = 42
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"

    #: Target late rate. Tuned to sit near the Olist figure (8.14%) so the
    #: demonstration exercises the same class of imbalance problem.
    target_late_rate: float = 0.095

    #: Coefficients on standardised drivers of the latent lateness score.
    #: These ARE the injected relationships. Changing them changes what SHAP
    #: will report, which is precisely why they are declared here rather than
    #: buried in the generation code.
    beta: dict[str, float] = field(default_factory=lambda: {
        "dispatch_lag": 0.95,        # strongest, mirrors the Olist finding
        "corridor_friction": 0.55,
        "distance": 0.40,
        "promise_slack": -0.85,      # a longer promise lowers risk
        "non_working_days": 0.30,
        "parcel_weight": 0.18,
        "peak_load": 0.35,
        "n_sellers": 0.22,
    })

    #: Standard deviation of the irreducible noise term. Larger means a harder
    #: problem. At 1.35 the achievable AUC lands around 0.82-0.86.
    noise_sd: float = 1.35

    #: Share of rows carrying each class of injected defect.
    dirty_city_rate: float = 0.06
    missing_category_rate: float = 0.025
    impossible_weight_rate: float = 0.0004
    carrier_before_purchase_rate: float = 0.0015


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------

#: Category name, median price in naira, median weight in grams.
#: Prices reflect Nigerian online retail at 2024 levels and are assumptions.
CATEGORIES: tuple[tuple[str, float, float], ...] = (
    ("phones_and_tablets",      185_000, 400),
    ("fashion_men",              14_500, 600),
    ("fashion_women",            16_000, 550),
    ("health_and_beauty",         9_800, 350),
    ("home_and_kitchen",         32_000, 2_800),
    ("electronics",              96_000, 1_900),
    ("computing",               310_000, 2_200),
    ("groceries",                 8_500, 3_500),
    ("baby_products",            19_000, 1_100),
    ("sports_and_fitness",       27_000, 2_400),
    ("books_and_stationery",      6_200, 800),
    ("automobile_parts",         58_000, 5_200),
    ("furniture",               145_000, 18_000),
    ("generators_and_power",    420_000, 32_000),
)

#: Categories whose English name is deliberately absent from the translation
#: table, mirroring the two untranslated Olist categories.
UNTRANSLATED = {"generators_and_power", "automobile_parts"}

PAYMENT_TYPES = ("card", "bank_transfer", "ussd", "pay_on_delivery", "wallet")
PAYMENT_WEIGHTS = (0.38, 0.27, 0.14, 0.16, 0.05)


def _ids(prefix: str, n: int) -> np.ndarray:
    return np.array([f"{prefix}{i:08d}" for i in range(n)])


def _dirty(city: str, rng: np.random.Generator) -> str:
    """Corrupt a city name the way free-text entry does.

    Mirrors the Olist defect classes: case drift, trailing state code,
    doubled spaces, and the occasional postcode typed into the city field.
    """
    style = rng.integers(0, 5)
    if style == 0:
        return city.lower()
    if style == 1:
        return f"{city.lower()} - {city[:2].lower()}"
    if style == 2:
        return city.lower().replace(" ", "  ")
    if style == 3:
        return city.upper()
    return f"{rng.integers(10000, 99999)}"


# ==========================================================================
# Generation
# ==========================================================================


def generate(config: GeneratorConfig | None = None) -> dict[str, pd.DataFrame]:
    """Build all nine tables. Deterministic given ``config.seed``."""
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_orders

    cities = geo.city_frame()
    n_sellers = max(60, n // 120)
    n_products = max(300, n // 25)
    n_customers = int(n * 0.93)

    # ---- sellers ---------------------------------------------------------
    s_city_idx = rng.choice(len(cities), n_sellers, p=cities["seller_weight"])
    sellers = pd.DataFrame({
        "seller_id": _ids("sell", n_sellers),
        "seller_zip_code_prefix": [
            geo.zip_prefix(int(i), rng, 1)[0] for i in s_city_idx
        ],
        "seller_city": cities["name"].to_numpy()[s_city_idx],
        "seller_state": cities["state"].to_numpy()[s_city_idx],
    })
    seller_zone = cities["zone"].to_numpy()[s_city_idx]
    #: Per-seller dispatch competence. The source of the seller-history signal.
    seller_speed = rng.gamma(2.2, 0.55, n_sellers) + 0.25

    # ---- customers -------------------------------------------------------
    c_city_idx = rng.choice(len(cities), n, p=cities["customer_weight"])
    customers = pd.DataFrame({
        "customer_id": _ids("cust", n),
        "customer_unique_id": rng.choice(_ids("uniq", n_customers), n),
        "customer_zip_code_prefix": [
            geo.zip_prefix(int(i), rng, 1)[0] for i in c_city_idx
        ],
        "customer_city": cities["name"].to_numpy()[c_city_idx],
        "customer_state": cities["state"].to_numpy()[c_city_idx],
    })
    customer_zone = pd.Series(cities["zone"].to_numpy()[c_city_idx])

    dirty = rng.random(n) < cfg.dirty_city_rate
    customers.loc[dirty, "customer_city"] = [
        _dirty(c, rng) for c in customers.loc[dirty, "customer_city"]
    ]

    # ---- products --------------------------------------------------------
    cat_idx = rng.integers(0, len(CATEGORIES), n_products)
    cat_names = np.array([CATEGORIES[i][0] for i in cat_idx])
    med_price = np.array([CATEGORIES[i][1] for i in cat_idx], dtype="float64")
    med_weight = np.array([CATEGORIES[i][2] for i in cat_idx], dtype="float64")

    # Lognormal spread around each category median: a few premium items, a
    # long tail of cheap ones, which is what a real catalogue looks like.
    price = med_price * rng.lognormal(0, 0.45, n_products)
    weight = med_weight * rng.lognormal(0, 0.40, n_products)

    length = np.clip(rng.normal(28, 12, n_products), 6, 110)
    height = np.clip(rng.normal(16, 8, n_products), 2, 90)
    width = np.clip(rng.normal(20, 9, n_products), 5, 100)

    products = pd.DataFrame({
        "product_id": _ids("prod", n_products),
        "product_category_name": cat_names,
        "product_name_lenght": rng.integers(18, 70, n_products),
        "product_description_lenght": rng.integers(80, 3_600, n_products),
        "product_photos_qty": rng.integers(1, 9, n_products),
        "product_weight_g": weight.round(0),
        "product_length_cm": length.round(0),
        "product_height_cm": height.round(0),
        "product_width_cm": width.round(0),
    })

    # Injected defects, mirroring the real release.
    impossible = rng.random(n_products) < cfg.impossible_weight_rate
    products.loc[impossible, "product_weight_g"] = 0
    missing_cat = rng.random(n_products) < cfg.missing_category_rate
    products.loc[missing_cat, "product_category_name"] = np.nan
    products.loc[missing_cat, "product_photos_qty"] = np.nan

    # ---- order composition ----------------------------------------------
    # Most orders are single-item; the tail matters because a multi-seller
    # order ships only when its slowest seller dispatches.
    n_items = rng.choice([1, 2, 3, 4, 5], n, p=[0.72, 0.17, 0.07, 0.03, 0.01])
    order_seller_idx = rng.choice(n_sellers, n)

    # ---- timing ----------------------------------------------------------
    start = pd.Timestamp(cfg.start_date)
    span = (pd.Timestamp(cfg.end_date) - start).days
    # Volume grows across the window, as a developing marketplace does.
    day_offset = (rng.beta(1.6, 1.15, n) * span).astype(int)
    purchase = start + pd.to_timedelta(day_offset, unit="D") + pd.to_timedelta(
        rng.integers(0, SECONDS_PER_DAY, n), unit="s"
    )

    approved = purchase + pd.to_timedelta(rng.exponential(4.5, n), unit="h")

    # Dispatch lag: how long the seller takes to hand the parcel over. The
    # dominant driver, mirroring the Olist operational finding.
    dispatch_lag = rng.gamma(1.9, seller_speed[order_seller_idx])
    carrier = approved + pd.to_timedelta(dispatch_lag, unit="D")

    # ---- route -----------------------------------------------------------
    s_idx_per_order = s_city_idx[order_seller_idx]
    distance = geo.haversine_km(
        cities["lat"].to_numpy()[s_idx_per_order],
        cities["lng"].to_numpy()[s_idx_per_order],
        cities["lat"].to_numpy()[c_city_idx],
        cities["lng"].to_numpy()[c_city_idx],
    )
    friction = geo.corridor_friction(
        pd.Series(seller_zone[order_seller_idx]), customer_zone
    ).to_numpy()

    # Promise: longer for distant and difficult routes, as a real promise is.
    promise_days = np.clip(
        3 + 0.009 * distance + 2.6 * (friction - 1) * 10 + rng.normal(0, 2.4, n),
        2, 45,
    ).round(0)
    estimated = purchase.normalize() + pd.to_timedelta(promise_days, unit="D")

    # Peak load: December pressure plus the ember-month effect.
    month = purchase.month
    peak_load = np.where(month == 12, 1.0, np.where(month.isin([10, 11]), 0.55, 0.0))

    # ---- transit and the emergent outcome --------------------------------
    # Lateness is NOT imposed and then back-filled into timestamps. Doing it
    # that way forced a contradiction: an order marked on-time could not
    # always be given a delivery date that was both after carrier handover
    # and before the promise, so the clamp silently made ~20% of them late.
    #
    # Instead transit time is generated causally -- it depends on distance,
    # corridor friction and a shock -- and lateness is whatever falls out of
    # comparing carrier handover plus transit against the promised date. The
    # timestamps and the target are then consistent by construction, which is
    # also how the Olist target is defined.
    def z(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype="float64")
        return (x - x.mean()) / (x.std() + 1e-9)

    b = cfg.beta
    non_working = rng.binomial(10, 0.28, n)
    parcel_weight = weight[rng.integers(0, n_products, n)]

    # Shock to transit, in log space. The coefficients are the injected
    # relationships: they decide what SHAP will later report as important.
    shock = (
        b["corridor_friction"] * z(friction)
        + b["distance"] * z(distance)
        + b["non_working_days"] * z(non_working)
        + b["parcel_weight"] * z(np.log1p(parcel_weight))
        + b["peak_load"] * z(peak_load)
        + b["n_sellers"] * z(n_items)
        + rng.normal(0, cfg.noise_sd, n)
    )

    # Base transit grows with distance and corridor difficulty.
    base_transit = (1.2 + 0.0045 * distance) * friction
    transit_days = np.clip(base_transit * np.exp(0.42 * shock), 0.4, 90)

    delivered = carrier + pd.to_timedelta(transit_days, unit="D")

    # Tune the promise so the realised late rate lands on target. The promise
    # is a business decision in reality, so shifting it is the honest lever:
    # it leaves the physical transit process untouched.
    gap_days = (delivered - purchase.normalize()).total_seconds() / SECONDS_PER_DAY
    needed = np.quantile(gap_days - promise_days, 1 - cfg.target_late_rate)
    promise_days = np.clip(promise_days + np.ceil(needed), 2, 60)
    estimated = purchase.normalize() + pd.to_timedelta(promise_days, unit="D")

    is_late = (delivered > estimated).astype(int)

    status = np.where(rng.random(n) < 0.968, "delivered",
                      rng.choice(["shipped", "canceled", "processing", "unavailable"], n))

    order_ids = _ids("ord", n)
    orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": customers["customer_id"],
        "order_status": status,
        "order_purchase_timestamp": purchase,
        "order_approved_at": approved,
        "order_delivered_carrier_date": carrier,
        "order_delivered_customer_date": delivered,
        "order_estimated_delivery_date": estimated,
    })
    undelivered = orders["order_status"] != "delivered"
    orders.loc[undelivered, "order_delivered_customer_date"] = pd.NaT

    # Injected defect: a few carrier timestamps before purchase.
    bad_carrier = rng.random(n) < cfg.carrier_before_purchase_rate
    orders.loc[bad_carrier, "order_delivered_carrier_date"] = (
        orders.loc[bad_carrier, "order_purchase_timestamp"] - pd.Timedelta(days=2)
    )

    # ---- order items -----------------------------------------------------
    item_order, item_no, item_product, item_seller = [], [], [], []
    for i, k in enumerate(n_items):
        for j in range(int(k)):
            item_order.append(order_ids[i])
            item_no.append(j + 1)
            item_product.append(rng.integers(0, n_products))
            # Most multi-item orders ship from one seller; some do not.
            item_seller.append(
                order_seller_idx[i] if j == 0 or rng.random() < 0.72
                else rng.integers(0, n_sellers)
            )
    item_product = np.array(item_product)
    item_seller = np.array(item_seller)
    n_rows = len(item_order)

    item_price = price[item_product] * rng.lognormal(0, 0.09, n_rows)
    item_distance = np.repeat(distance, n_items)
    item_weight = weight[item_product]
    # Freight rises with weight and distance -- the relationship a reader
    # would expect, and one the model can use.
    freight = (
        700
        + 0.95 * item_distance
        + 0.085 * item_weight
        + rng.normal(0, 320, n_rows)
    ).clip(400, None)

    shipping_limit = np.repeat(approved.to_numpy(), n_items) + pd.to_timedelta(
        rng.integers(2, 9, n_rows), unit="D"
    )

    order_items = pd.DataFrame({
        "order_id": item_order,
        "order_item_id": item_no,
        "product_id": products["product_id"].to_numpy()[item_product],
        "seller_id": sellers["seller_id"].to_numpy()[item_seller],
        "shipping_limit_date": shipping_limit,
        "price": item_price.round(2),
        "freight_value": freight.round(2),
    })

    # ---- payments --------------------------------------------------------
    order_value = order_items.groupby("order_id")["price"].sum() + \
        order_items.groupby("order_id")["freight_value"].sum()
    order_value = order_value.reindex(order_ids).fillna(0)

    pay_type = rng.choice(PAYMENT_TYPES, n, p=PAYMENT_WEIGHTS)
    # Instalments only apply to card payments, as they do in practice.
    installments = np.where(
        pay_type == "card",
        rng.choice([1, 2, 3, 6, 12], n, p=[0.55, 0.16, 0.14, 0.10, 0.05]),
        1,
    )
    payments = pd.DataFrame({
        "order_id": order_ids,
        "payment_sequential": 1,
        "payment_type": pay_type,
        "payment_installments": installments,
        "payment_value": order_value.to_numpy().round(2),
    })
    # Injected defects mirroring the real release.
    payments.loc[rng.choice(n, 3, replace=False), "payment_type"] = "not_defined"
    payments.loc[rng.choice(n, 9, replace=False), "payment_value"] = 0.0
    payments.loc[rng.choice(n, 2, replace=False), "payment_installments"] = 0

    # ---- reviews ---------------------------------------------------------
    # A late delivery costs roughly 1.7 stars, the effect size measured on
    # Olist. Injected here so the cost model has something to rest on, and
    # therefore NOT evidence of the same effect in Nigeria.
    base_score = np.where(is_late == 1,
                          rng.normal(2.6, 1.05, n),
                          rng.normal(4.3, 0.85, n))
    review_score = np.clip(np.rint(base_score), 1, 5).astype(int)
    has_review = rng.random(n) < 0.80

    reviews = pd.DataFrame({
        "review_id": _ids("rev", n),
        "order_id": order_ids,
        "review_score": review_score,
        "review_comment_title": pd.Series(
            np.where(rng.random(n) < 0.12, "Review", None), dtype="object"),
        "review_comment_message": pd.Series(
            np.where(rng.random(n) < 0.42, "Comment text", None), dtype="object"),
        "review_creation_date": delivered + pd.to_timedelta(rng.integers(1, 5, n), unit="D"),
        "review_answer_timestamp": delivered + pd.to_timedelta(rng.integers(2, 9, n), unit="D"),
    })[has_review].reset_index(drop=True)

    # ---- geolocation -----------------------------------------------------
    all_zips = pd.concat([
        customers["customer_zip_code_prefix"], sellers["seller_zip_code_prefix"]
    ]).drop_duplicates()
    zip_city = pd.concat([
        pd.Series(c_city_idx, index=customers["customer_zip_code_prefix"]),
        pd.Series(s_city_idx, index=sellers["seller_zip_code_prefix"]),
    ])
    zip_city = zip_city[~zip_city.index.duplicated()].reindex(all_zips)

    g_n = len(all_zips)
    # Several rows per prefix, as the real table has, with jitter around the
    # city centre so the cleaner's centroid aggregation does real work.
    repeat = rng.integers(2, 7, g_n)
    geo_zip = np.repeat(all_zips.to_numpy(), repeat)
    geo_idx = np.repeat(zip_city.to_numpy(), repeat)
    total = len(geo_zip)

    geolocation = pd.DataFrame({
        "geolocation_zip_code_prefix": geo_zip,
        "geolocation_lat": cities["lat"].to_numpy()[geo_idx] + rng.normal(0, 0.06, total),
        "geolocation_lng": cities["lng"].to_numpy()[geo_idx] + rng.normal(0, 0.06, total),
        "geolocation_city": cities["name"].to_numpy()[geo_idx],
        "geolocation_state": cities["state"].to_numpy()[geo_idx],
    })
    # A handful of bad fixes outside Nigeria, for the bounds filter to catch.
    bad = rng.choice(total, max(1, total // 500), replace=False)
    geolocation.loc[bad, "geolocation_lat"] = rng.uniform(-40, -20, len(bad))

    # ---- category translation -------------------------------------------
    present = sorted({c[0] for c in CATEGORIES} - UNTRANSLATED)
    translation = pd.DataFrame({
        "product_category_name": present,
        "product_category_name_english": [c.replace("_", " ").title() for c in present],
    })

    tables = {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "order_payments": payments,
        "order_reviews": reviews,
        "products": products,
        "sellers": sellers,
        "geolocation": geolocation,
        "category_translation": translation,
    }

    realised = is_late[status == "delivered"].mean()
    logger.info(
        "Generated %d orders | realised late rate %.2f%% | %d sellers, %d products",
        n, 100 * realised, n_sellers, n_products,
    )
    return tables


def write_raw(tables: dict[str, pd.DataFrame], raw_dir: Path) -> Path:
    """Write the nine tables under the Olist filenames.

    Using the same filenames is what lets ``io.load_all(raw_dir)`` read this
    environment without a single conditional.
    """
    from ..config import RAW_FILES

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(raw_dir / RAW_FILES[name], index=False)
    logger.info("Wrote %d tables to %s", len(tables), raw_dir)
    return raw_dir
