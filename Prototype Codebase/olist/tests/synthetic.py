"""Synthetic Olist-shaped data.

Generates all nine tables with the exact column names, dtypes and defect
patterns of the real release -- including the misspelled ``lenght`` columns,
``not_defined`` payment types, compound city strings and untranslated
categories -- so the pipeline can be exercised end to end without the real
CSVs, and so tests have deterministic fixtures.

This is **not** a substitute for running on real data. Relationships between
features and the target are injected deliberately and are not realistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "PE", "CE", "AM"]
CITIES = ["sao paulo", "rio de janeiro", "belo horizonte", "curitiba", "salvador"]
DIRTY_CITIES = ["sao paulo - sp", "são paulo", "sp", "rio de janeiro / rio de janeiro", "04482255"]
CATEGORIES = ["cama_mesa_banho", "beleza_saude", "esporte_lazer", "pc_gamer", "moveis_decoracao"]
TRANSLATABLE = ["cama_mesa_banho", "beleza_saude", "esporte_lazer", "moveis_decoracao"]


def _ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:08d}" for i in range(n)]


def make_tables(n_orders: int = 4000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    n_customers = int(n_orders * 0.97)
    n_sellers = max(20, n_orders // 40)
    n_products = max(50, n_orders // 8)

    customer_ids = _ids("cust", n_orders)
    unique_ids = rng.choice(_ids("uniq", n_customers), n_orders)
    seller_ids = _ids("sell", n_sellers)
    product_ids = _ids("prod", n_products)
    order_ids = _ids("ord", n_orders)

    zips = [f"{z:05d}" for z in rng.integers(1001, 99730, n_orders)]
    seller_zips = [f"{z:05d}" for z in rng.integers(1001, 99730, n_sellers)]

    # ---- customers -------------------------------------------------------
    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_unique_id": unique_ids,
        "customer_zip_code_prefix": zips,
        "customer_city": rng.choice(CITIES + DIRTY_CITIES, n_orders),
        "customer_state": rng.choice(STATES, n_orders),
    })

    # ---- orders ----------------------------------------------------------
    purchase = pd.to_datetime("2016-09-05") + pd.to_timedelta(
        rng.integers(0, 770, n_orders), unit="D"
    ) + pd.to_timedelta(rng.integers(0, 86400, n_orders), unit="s")

    approved = purchase + pd.to_timedelta(rng.exponential(6, n_orders), unit="h")
    pickup = approved + pd.to_timedelta(rng.gamma(2, 1.2, n_orders), unit="D")
    estimated = purchase.normalize() + pd.to_timedelta(
        rng.integers(8, 40, n_orders), unit="D"
    )

    # Actual delivery: base transit plus a shock, so lateness is learnable.
    transit = rng.gamma(3, 3, n_orders) + 2
    delivered = pickup + pd.to_timedelta(transit, unit="D")

    status = np.where(rng.random(n_orders) < 0.965, "delivered",
                      rng.choice(["shipped", "canceled", "unavailable", "processing"], n_orders))
    undelivered = status != "delivered"
    delivered = pd.Series(delivered)
    delivered[undelivered] = pd.NaT

    orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": customer_ids,
        "order_status": status,
        "order_purchase_timestamp": purchase,
        "order_approved_at": approved,
        "order_delivered_carrier_date": pd.Series(pickup).where(~undelivered),
        "order_delivered_customer_date": delivered,
        "order_estimated_delivery_date": estimated,
    })

    # ---- products (with real defect patterns) ----------------------------
    products = pd.DataFrame({
        "product_id": product_ids,
        "product_category_name": rng.choice(CATEGORIES, n_products),
        "product_name_lenght": rng.integers(5, 76, n_products).astype(float),
        "product_description_lenght": rng.integers(4, 3992, n_products).astype(float),
        "product_photos_qty": rng.integers(1, 20, n_products).astype(float),
        "product_weight_g": rng.gamma(2, 900, n_products).round(),
        "product_length_cm": rng.integers(7, 105, n_products).astype(float),
        "product_height_cm": rng.integers(2, 105, n_products).astype(float),
        "product_width_cm": rng.integers(6, 118, n_products).astype(float),
    })
    unlisted = rng.choice(n_products, max(1, n_products // 54), replace=False)
    products.loc[unlisted, ["product_category_name", "product_name_lenght",
                            "product_description_lenght", "product_photos_qty"]] = np.nan
    products.loc[rng.choice(n_products, 4, replace=False), "product_weight_g"] = 0

    # ---- sellers ---------------------------------------------------------
    sellers = pd.DataFrame({
        "seller_id": seller_ids,
        "seller_zip_code_prefix": seller_zips,
        "seller_city": rng.choice(CITIES + DIRTY_CITIES, n_sellers),
        "seller_state": rng.choice(STATES, n_sellers),
    })

    # ---- order items (deliberately one-to-many) --------------------------
    counts = rng.choice([1, 1, 1, 1, 2, 3], n_orders)
    rows = []
    for oid, ts, k in zip(order_ids, purchase, counts):
        for j in range(k):
            rows.append({
                "order_id": oid,
                "order_item_id": j + 1,
                "product_id": rng.choice(product_ids),
                "seller_id": rng.choice(seller_ids),
                "shipping_limit_date": ts + pd.Timedelta(days=int(rng.integers(2, 10))),
                "price": round(float(rng.gamma(2, 60)), 2),
                "freight_value": round(float(rng.gamma(2, 8)), 2),
            })
    order_items = pd.DataFrame(rows)

    # ---- payments (one-to-many, with defects) ----------------------------
    pay_rows = []
    for oid in order_ids:
        n_pay = 1 if rng.random() < 0.955 else 2
        for s in range(n_pay):
            pay_rows.append({
                "order_id": oid,
                "payment_sequential": s + 1,
                "payment_type": rng.choice(["credit_card", "boleto", "voucher", "debit_card"],
                                           p=[0.74, 0.19, 0.055, 0.015]),
                "payment_installments": int(rng.choice([1, 1, 1, 2, 3, 4, 6, 8, 10])),
                "payment_value": round(float(rng.gamma(2, 80)), 2),
            })
    order_payments = pd.DataFrame(pay_rows)
    bad = rng.choice(len(order_payments), 3, replace=False)
    order_payments.loc[bad, "payment_type"] = "not_defined"
    order_payments.loc[bad, "payment_value"] = 0.0
    order_payments.loc[rng.choice(len(order_payments), 2, replace=False), "payment_installments"] = 0

    # ---- reviews (with duplicates per order) -----------------------------
    reviewed = rng.random(n_orders) < 0.99
    rev = pd.DataFrame({
        "review_id": _ids("rev", int(reviewed.sum())),
        "order_id": np.array(order_ids)[reviewed],
        "review_score": rng.integers(1, 6, int(reviewed.sum())),
        "review_comment_title": np.nan,
        "review_comment_message": rng.choice([np.nan, "entrega atrasada", "produto otimo"],
                                             int(reviewed.sum())),
        "review_creation_date": pd.Series(purchase)[reviewed].values + pd.Timedelta(days=12),
        "review_answer_timestamp": pd.Series(purchase)[reviewed].values + pd.Timedelta(days=13),
    })
    order_reviews = pd.concat([rev, rev.head(20)], ignore_index=True)  # duplicate reviews

    # ---- geolocation (many rows per zip, some out of bounds) -------------
    all_zips = sorted(set(zips) | set(seller_zips))
    geo_rows = []
    for z in all_zips:
        for _ in range(int(rng.integers(1, 4))):
            geo_rows.append({
                "geolocation_zip_code_prefix": z,
                "geolocation_lat": float(rng.uniform(-30, -3)),
                "geolocation_lng": float(rng.uniform(-60, -38)),
                "geolocation_city": rng.choice(CITIES),
                "geolocation_state": rng.choice(STATES),
            })
    geolocation = pd.DataFrame(geo_rows)
    geolocation.loc[rng.choice(len(geolocation), 5, replace=False), "geolocation_lat"] = 45.0

    # ---- translation (BOM handled at load; two categories missing) -------
    category_translation = pd.DataFrame({
        "product_category_name": TRANSLATABLE,
        "product_category_name_english": ["bed_bath_table", "health_beauty",
                                          "sports_leisure", "furniture_decor"],
    })

    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "order_payments": order_payments,
        "order_reviews": order_reviews,
        "products": products,
        "sellers": sellers,
        "geolocation": geolocation,
        "category_translation": category_translation,
    }
