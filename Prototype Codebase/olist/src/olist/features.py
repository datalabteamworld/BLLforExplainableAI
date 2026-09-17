"""Feature engineering.

The central design decision here is **grain**. The original analysis merged
everything at order-*item* level, producing 113k rows for 99k orders. Because
the target (`is_late_delivery`) is a property of the order, every item of a
multi-item order carried an identical label and near-identical features. Any
random split then scattered sibling rows across train and test, which inflates
measured performance without improving the model.

This module aggregates to **one row per order** before modelling. Item-level
variation is preserved as explicit features -- item count, distinct sellers,
maximum weight, price spread -- several of which are plausibly predictive of
late delivery in their own right, since a multi-seller order cannot ship until
its slowest seller dispatches.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import calendar_br as cb
from . import history

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400


# ==========================================================================
# Table-level aggregation to order grain
# ==========================================================================


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Collapse payments to one row per order.

    ``payment_type`` is taken from the instalment with the largest value
    rather than the first row: a voucher + credit-card order sequenced with
    the voucher first would otherwise be labelled a voucher order even when
    the card carried almost all of the value.
    """
    dominant = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates(subset=["order_id"], keep="first")
        .set_index("order_id")["payment_type"]
        .rename("payment_type")
    )

    agg = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        n_payment_methods=("payment_type", "nunique"),
        n_payment_records=("payment_value", "size"),
    )
    return agg.join(dominant).reset_index()


def aggregate_items(items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Collapse order items to one row per order, retaining item-level spread.

    ``n_sellers`` matters: a two-seller order ships only when the slower
    seller dispatches, so its exposure to delay is the maximum of two
    processes, not the average of one.
    """
    merged = items.merge(products, on="product_id", how="left")

    agg = merged.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        n_products=("product_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        max_item_price=("price", "max"),
        mean_item_price=("price", "mean"),
        total_weight_g=("product_weight_g", "sum"),
        max_weight_g=("product_weight_g", "max"),
        total_volume_cm3=("product_volume_cm3", "sum"),
        max_volume_cm3=("product_volume_cm3", "max"),
        mean_density=("product_density_g_cm3", "mean"),
        mean_photos_qty=("product_photos_qty", "mean"),
        mean_description_length=("product_description_length", "mean"),
        shipping_limit_date=("shipping_limit_date", "min"),
        has_listing_metadata=("has_listing_metadata", "min"),
    )

    # The dominant seller and category are those of the highest-priced item:
    # the item most likely to drive handling and dispatch behaviour.
    dominant = (
        merged.sort_values("price", ascending=False)
        .drop_duplicates(subset=["order_id"], keep="first")
        .set_index("order_id")[["seller_id", "product_category_name"]]
        .rename(columns={"seller_id": "seller_id_primary"})
    )

    out = agg.join(dominant).reset_index()
    out["freight_ratio"] = out["total_freight"] / out["total_price"].replace(0, np.nan)
    out["is_multi_seller"] = (out["n_sellers"] > 1).astype("int8")
    return out


# ==========================================================================
# Merge chain
# ==========================================================================


def build_order_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble one row per order from the cleaned tables.

    Anchored on ``orders`` with left joins throughout, so the row count is
    invariant: it must equal ``len(orders)`` at every step. That invariant is
    asserted rather than merely printed, because a silent fan-out from a
    non-unique join key is the single most common defect in a merge chain
    this long.
    """
    orders = tables["orders"]
    n_orders = len(orders)

    df = orders.merge(tables["customers"], on="customer_id", how="left")
    _assert_grain(df, n_orders, "customers")

    df = df.merge(aggregate_items(tables["order_items"], tables["products"]),
                  on="order_id", how="left")
    _assert_grain(df, n_orders, "order_items")

    df = df.merge(tables["category_translation"], on="product_category_name", how="left")
    _assert_grain(df, n_orders, "category_translation")
    df["product_category"] = df["product_category_name_english"].fillna("unlisted")

    sellers = tables["sellers"][["seller_id", "seller_city", "seller_state", "seller_zip_code_prefix"]]
    df = df.merge(sellers.rename(columns={"seller_id": "seller_id_primary"}),
                  on="seller_id_primary", how="left")
    _assert_grain(df, n_orders, "sellers")

    df = df.merge(aggregate_payments(tables["order_payments"]), on="order_id", how="left")
    _assert_grain(df, n_orders, "order_payments")

    reviews = tables["order_reviews"][["order_id", "review_score", "has_review_comment"]]
    df = df.merge(reviews, on="order_id", how="left")
    _assert_grain(df, n_orders, "order_reviews")

    geo = tables["geolocation"]
    df = df.merge(
        geo.rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix",
                            "lat": "customer_lat", "lng": "customer_lng"}),
        on="customer_zip_code_prefix", how="left")
    df = df.merge(
        geo.rename(columns={"geolocation_zip_code_prefix": "seller_zip_code_prefix",
                            "lat": "seller_lat", "lng": "seller_lng"}),
        on="seller_zip_code_prefix", how="left")
    _assert_grain(df, n_orders, "geolocation")

    logger.info("Order table assembled: %d rows x %d cols", len(df), df.shape[1])
    return df


def _assert_grain(df: pd.DataFrame, expected: int, step: str) -> None:
    if len(df) != expected:
        raise ValueError(
            f"Merge with {step!r} changed the row count from {expected:,} to "
            f"{len(df):,}. The join key is not unique on the right-hand table; "
            f"aggregate it to order grain first."
        )


# ==========================================================================
# Derived features
# ==========================================================================


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add delivery durations and the binary late-delivery target."""
    out = df.copy()

    def _days(a: str, b: str) -> pd.Series:
        return (out[a] - out[b]).dt.total_seconds() / SECONDS_PER_DAY

    out["actual_delivery_days"] = _days("order_delivered_customer_date", "order_purchase_timestamp")
    out["estimated_delivery_days"] = _days("order_estimated_delivery_date", "order_purchase_timestamp")
    out["delivery_delay_days"] = _days("order_delivered_customer_date", "order_estimated_delivery_date")
    out["approval_time_hours"] = (
        (out["order_approved_at"] - out["order_purchase_timestamp"]).dt.total_seconds() / 3600
    )
    out["carrier_pickup_days"] = _days("order_delivered_carrier_date", "order_purchase_timestamp")
    #: Slack between the seller's contractual dispatch deadline and the
    #: customer's promised date: how much buffer the promise actually has.
    out["shipping_limit_slack_days"] = _days("order_estimated_delivery_date", "shipping_limit_date")

    out["is_late_delivery"] = (out["delivery_delay_days"] > 0).astype("int8")
    out.loc[out["delivery_delay_days"].isna(), "is_late_delivery"] = np.nan
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add purchase-time calendar features and transit-window event counts.

    The transit-window counts are the substantive addition over a
    purchase-date holiday flag: they measure how many non-working days the
    carrier must traverse to hit the promised date.
    """
    out = df.copy()
    ts = out["order_purchase_timestamp"]

    out["purchase_year"] = ts.dt.year
    out["purchase_month"] = ts.dt.month
    out["purchase_dow"] = ts.dt.dayofweek
    out["purchase_hour"] = ts.dt.hour
    out["purchase_quarter"] = ts.dt.quarter
    out["is_weekend_purchase"] = ts.dt.dayofweek.isin([5, 6]).astype("int8")

    lo = ts.min().normalize() - pd.Timedelta(days=30)
    hi = max(ts.max(), out["order_estimated_delivery_date"].max()).normalize() + pd.Timedelta(days=30)
    cal = cb.build_daily_calendar(lo, hi)
    cal["is_non_working"] = 1 - cal["is_working_day"]

    purchase_date = ts.dt.normalize()
    day_flags = cal.set_index("date")
    for flag in ("is_holiday", "is_black_friday", "is_cyber_monday", "is_carnival"):
        out[f"purchased_on_{flag[3:]}"] = (
            purchase_date.map(day_flags[flag]).fillna(0).astype("int8")
        )

    years = range(ts.min().year, ts.max().year + 1)
    out["days_from_black_friday"] = cb.days_to_nearest_black_friday(ts, years)
    out["is_black_friday_week"] = (out["days_from_black_friday"].abs() <= 7).astype("int8")

    # --- transit-window counts ---------------------------------------------
    est = out["order_estimated_delivery_date"]
    for flag, name in [
        ("is_non_working", "non_working_days_in_window"),
        ("is_holiday", "holidays_in_window"),
        ("is_carnival", "carnival_days_in_window"),
        ("is_weekend", "weekend_days_in_window"),
    ]:
        out[name] = cb.count_events_in_window(purchase_date, est, cal, flag)

    #: Working days available to hit the promise. A promise spanning a long
    #: holiday stretch has fewer usable days than its calendar length implies.
    out["working_days_in_window"] = (
        out["estimated_delivery_days"] - out["non_working_days_in_window"]
    )
    out["non_working_ratio"] = (
        out["non_working_days_in_window"] / out["estimated_delivery_days"].replace(0, np.nan)
    )
    return out


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Great-circle distance between seller and customer, plus route flags."""
    out = df.copy()
    lat1, lon1, lat2, lon2 = (
        np.radians(out["seller_lat"].to_numpy(dtype="float64")),
        np.radians(out["seller_lng"].to_numpy(dtype="float64")),
        np.radians(out["customer_lat"].to_numpy(dtype="float64")),
        np.radians(out["customer_lng"].to_numpy(dtype="float64")),
    )
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    out["seller_customer_distance_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    out["is_same_state"] = (out["customer_state"] == out["seller_state"]).astype("int8")
    out["is_same_city"] = (out["customer_city"] == out["seller_city"]).astype("int8")
    return out


def build_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Full feature pipeline: merge, target, calendar, geography."""
    df = build_order_table(tables)
    df = add_target(df)
    df = add_calendar_features(df)
    df = add_geo_features(df)
    logger.info("Feature table: %d rows x %d cols", len(df), df.shape[1])
    return df


def build_modelling_table(
    tables: dict[str, pd.DataFrame],
    censor_buffer_days: int = 30,
) -> pd.DataFrame:
    """Features, then delivered-only rows, then history, then censoring.

    Order matters. History is computed *after* restricting to resolved orders
    (only a delivered order has an observable outcome to contribute) but
    *before* censoring, so that orders dropped at the end of the window still
    inform the history of the orders that remain.
    """
    df = build_features(tables)
    df = select_modelling_rows(df)
    df = history.add_history_features(df)
    df = history.apply_censoring_filter(df, buffer_days=censor_buffer_days)
    logger.info("Modelling table: %d rows x %d cols", len(df), df.shape[1])
    return df


def select_modelling_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep delivered orders with a resolvable target.

    Undelivered orders have no observable outcome and cancelled orders have
    a different data-generating process, so both are excluded rather than
    imputed.
    """
    mask = df["order_status"].eq("delivered") & df["is_late_delivery"].notna()
    out = df.loc[mask].copy()
    out["is_late_delivery"] = out["is_late_delivery"].astype("int8")
    logger.info(
        "Modelling rows: %d of %d (%.1f%%), late rate %.2f%%",
        len(out), len(df), 100 * len(out) / len(df), 100 * out["is_late_delivery"].mean(),
    )
    return out
