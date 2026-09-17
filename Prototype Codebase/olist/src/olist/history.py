"""Backward-looking history features.

The pre-dispatch model's weakness is that nothing tells it *this seller has
been running late*. Adding that is the single largest available improvement,
because the post-dispatch comparison showed dispatch behaviour is where the
signal lives -- and a seller's past dispatch record, unlike the current
order's pickup time, is knowable the moment an order is placed.

It is also the easiest feature in the project to leak, in a way that looks
correct and inflates every metric.

**The trap.** The obvious implementation groups by seller, sorts by purchase
date, and takes an expanding mean of ``is_late``. That leaks. An order placed
on 1 June is not *resolved* on 1 June -- its lateness only becomes known when
it is delivered, perhaps 20 days later. Using it to score an order placed on
5 June means using an outcome from 21 June to predict something on 5 June.

Worse, the leak is selective: late orders resolve *later* than on-time ones,
so a purchase-time expanding mean systematically over-represents outcomes that
had not yet happened. The bias points toward exactly the thing being
predicted.

**The rule enforced here:** an order contributes to a seller's history only
once ``order_delivered_customer_date`` has passed. Every feature in this
module is computed from outcomes observable strictly before the scored order's
purchase timestamp.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Weight of the global prior when smoothing a rate. With ``alpha = 20``, a
#: seller needs ~20 resolved orders before their own rate dominates. Without
#: smoothing, a seller with one late order out of one reads as 100% risk.
SMOOTHING_ALPHA = 20.0


def _resolved_timeline(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Cumulative outcome counts per group, indexed by resolution time.

    Returns one row per resolved order carrying the running totals *including*
    that order, sorted by the moment the outcome became observable.
    """
    events = (
        df.loc[
            df["order_delivered_customer_date"].notna() & df["is_late_delivery"].notna(),
            [key, "order_delivered_customer_date", "is_late_delivery"],
        ]
        .dropna(subset=[key])
        .sort_values("order_delivered_customer_date")
        .reset_index(drop=True)
    )
    grouped = events.groupby(key, sort=False)["is_late_delivery"]
    events["_n_resolved"] = grouped.cumcount() + 1
    events["_n_late"] = grouped.cumsum()
    return events


def _global_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Marketplace-wide cumulative late rate, by resolution time.

    Used as the prior that sparse seller rates are smoothed toward. It is
    itself expanding rather than a fixed constant, so no future information
    enters through the prior.
    """
    events = (
        df.loc[
            df["order_delivered_customer_date"].notna() & df["is_late_delivery"].notna(),
            ["order_delivered_customer_date", "is_late_delivery"],
        ]
        .sort_values("order_delivered_customer_date")
        .reset_index(drop=True)
    )
    events["_global_n"] = np.arange(1, len(events) + 1)
    events["_global_late"] = events["is_late_delivery"].cumsum()
    return events[["order_delivered_customer_date", "_global_n", "_global_late"]]


def _asof_join(
    df: pd.DataFrame,
    timeline: pd.DataFrame,
    by: str | None,
    value_cols: list[str],
) -> pd.DataFrame:
    """As-of backward join: newest outcome resolved strictly before purchase.

    ``allow_exact_matches=False`` matters. An order delivered at the exact
    instant another is purchased must not count, and more importantly it stops
    an order from contributing to its own history when timestamps coincide.
    """
    left = df[["order_purchase_timestamp"] + ([by] if by else [])].copy()
    left["_row"] = np.arange(len(left))
    left = left.sort_values("order_purchase_timestamp")

    right = timeline.sort_values("order_delivered_customer_date").copy()

    # merge_asof requires identical datetime resolution on both keys. pandas 3
    # may hold one column at microsecond and another at nanosecond precision
    # depending on how each was parsed, which raises a MergeError rather than
    # coercing.
    unit = "datetime64[ns]"
    left["order_purchase_timestamp"] = left["order_purchase_timestamp"].astype(unit)
    right["order_delivered_customer_date"] = right["order_delivered_customer_date"].astype(unit)

    merged = pd.merge_asof(
        left,
        right[["order_delivered_customer_date"] + ([by] if by else []) + value_cols],
        left_on="order_purchase_timestamp",
        right_on="order_delivered_customer_date",
        by=by,
        direction="backward",
        allow_exact_matches=False,
    )
    return merged.sort_values("_row")[value_cols].to_numpy()


def add_history_features(df: pd.DataFrame, alpha: float = SMOOTHING_ALPHA) -> pd.DataFrame:
    """Add seller, route and category history known at purchase time.

    Every rate is smoothed toward the running marketplace rate, so a seller
    with two resolved orders does not read as 0% or 100% risk.
    """
    out = df.sort_values("order_purchase_timestamp").copy()

    global_tl = _global_timeline(out)
    global_stats = _asof_join(out, global_tl, None, ["_global_n", "_global_late"])
    g_n = pd.Series(global_stats[:, 0], index=out.index).fillna(0.0)
    g_late = pd.Series(global_stats[:, 1], index=out.index).fillna(0.0)
    #: Marketplace late rate as of this purchase. Also a useful feature in its
    #: own right: it carries the platform-wide trend the model would otherwise
    #: have to infer from purchase_month.
    prior = (g_late / g_n.replace(0, np.nan)).fillna(out["is_late_delivery"].mean())
    out["market_late_rate_to_date"] = prior

    specs = [
        ("seller_id_primary", "seller"),
        ("customer_state", "customer_state"),
        ("product_category", "category"),
    ]
    for key, name in specs:
        if key not in out.columns:
            continue
        timeline = _resolved_timeline(out, key)
        stats = _asof_join(out, timeline, key, ["_n_resolved", "_n_late"])
        n = pd.Series(stats[:, 0], index=out.index).fillna(0.0)
        late = pd.Series(stats[:, 1], index=out.index).fillna(0.0)

        out[f"{name}_orders_to_date"] = n
        out[f"{name}_late_rate_to_date"] = (late + alpha * prior) / (n + alpha)
        #: How much evidence the rate rests on. Lets a tree learn to distrust
        #: the rate for new sellers rather than treating all rates alike.
        out[f"{name}_history_confidence"] = n / (n + alpha)

    # Route-level history: a seller may be reliable in general but slow on one
    # long inter-state lane.
    if {"seller_state", "customer_state"} <= set(out.columns):
        out["_route"] = out["seller_state"].astype(str) + ">" + out["customer_state"].astype(str)
        timeline = _resolved_timeline(out, "_route")
        stats = _asof_join(out, timeline, "_route", ["_n_resolved", "_n_late"])
        n = pd.Series(stats[:, 0], index=out.index).fillna(0.0)
        late = pd.Series(stats[:, 1], index=out.index).fillna(0.0)
        out["route_orders_to_date"] = n
        out["route_late_rate_to_date"] = (late + alpha * prior) / (n + alpha)
        out = out.drop(columns=["_route"])

    # Seller tenure: days between a seller's first observed order and this one.
    first_seen = out.groupby("seller_id_primary")["order_purchase_timestamp"].transform("min")
    out["seller_tenure_days"] = (
        (out["order_purchase_timestamp"] - first_seen).dt.total_seconds() / 86_400
    )

    added = [c for c in out.columns if c not in df.columns]
    logger.info("Added %d history features: %s", len(added), ", ".join(sorted(added)))
    return out.loc[df.index]


# ==========================================================================
# Right-censoring
# ==========================================================================


def apply_censoring_filter(df: pd.DataFrame, buffer_days: int = 30) -> pd.DataFrame:
    """Drop orders that had not had a fair chance to be observed as late.

    An order enters the modelling table only once it has been delivered. Late
    orders are delivered later than on-time ones, so near the end of the data
    the late ones are disproportionately still in transit and therefore
    missing. The observed late rate falls for reasons that have nothing to do
    with performance.

    This is visible in the split: 8.81% late in training against 6.18% in the
    test period. Some of that gap is genuine improvement by Olist; some is
    this artefact. Requiring the promised date to fall at least
    ``buffer_days`` before the last observed delivery separates the two.
    """
    cutoff = df["order_delivered_customer_date"].max()
    horizon = cutoff - pd.Timedelta(days=buffer_days)

    keep = df["order_estimated_delivery_date"] <= horizon
    dropped = int((~keep).sum())

    out = df.loc[keep].copy()
    logger.info(
        "Censoring filter: dropped %d orders promised after %s (last delivery %s). "
        "Late rate %.2f%% -> %.2f%%",
        dropped, horizon.date(), cutoff.date(),
        100 * df["is_late_delivery"].mean(), 100 * out["is_late_delivery"].mean(),
    )
    return out
