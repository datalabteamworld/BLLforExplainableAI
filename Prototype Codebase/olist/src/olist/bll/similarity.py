"""Similar historical orders.

Answers "what happened to orders like this one?" — evidence a business user can
weigh independently of the model's own reasoning. When the model says 78% and
the twenty most similar historical orders were late 16 times, those are two
distinct pieces of evidence pointing the same way.

**Leakage constraint.** Similarity is computed only on features knowable at
prediction time. Matching on ``actual_delivery_days`` or ``delivery_delay_days``
would retrieve orders that are similar *in outcome*, making the outcome
statistics circular — the retrieved orders would be late because lateness was
the search key. Those columns appear in the *result*, never in the query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Features used to judge similarity. Each is knowable when the order is placed
#: (or, for carrier handover, when the post-dispatch model runs). Ordered by
#: how much a logistics reader would expect them to matter.
SIMILARITY_FEATURES = [
    "seller_customer_distance_km",
    "estimated_delivery_days",
    "working_days_in_window",
    "total_freight",
    "total_price",
    "total_weight_g",
    "n_items",
    "n_sellers",
    "carrier_pickup_days",
    "freight_ratio",
]

#: Categorical features that must match exactly rather than by distance. A
#: São Paulo -> Bahia order is not "60% similar" to a São Paulo -> Paraná one;
#: the route either matches or it does not.
EXACT_MATCH_FEATURES = ["customer_state", "seller_state"]


@dataclass
class SimilarOrders:
    """Retrieved neighbours and their outcome statistics."""

    orders: pd.DataFrame
    mean_similarity: float
    late_rate: float
    mean_delay_days: float
    n_found: int
    n_requested: int
    features_used: list[str]
    sufficient: bool
    note: str = ""


def _scaled(frame: pd.DataFrame, cols: list[str], centre: pd.Series,
            spread: pd.Series) -> np.ndarray:
    """Robust-scale to the reference distribution, filling gaps with the centre."""
    block = (frame[cols] - centre[cols]) / spread[cols]
    return block.fillna(0.0).to_numpy(dtype="float64")


def find_similar_orders(
    row: pd.Series,
    history: pd.DataFrame,
    k: int = 10,
    require_same_route: bool = False,
    min_results: int = 5,
    exclude_order_id: object = None,
) -> SimilarOrders:
    """Retrieve the ``k`` most similar historical orders.

    Similarity is ``1 / (1 + mean scaled distance)`` over the available
    features — bounded in (0, 1], monotone decreasing in distance, and
    interpretable as "1.0 means identical on every feature used".

    Median and IQR are used for scaling rather than mean and standard
    deviation, because freight, weight and distance are heavily right-skewed
    and a few very large orders would otherwise dominate the metric.
    """
    # An order is trivially its own nearest neighbour. Left in, it occupies the
    # top slot at 100% similarity, makes the comparison table read all zeros,
    # and inflates the outcome statistics with the very order being asked
    # about. Its own outcome is not evidence about itself.
    if exclude_order_id is not None and "order_id" in history.columns:
        history = history[history["order_id"] != exclude_order_id]

    cols = [c for c in SIMILARITY_FEATURES if c in history.columns and c in row.index]
    if not cols or history.empty:
        return SimilarOrders(
            orders=pd.DataFrame(), mean_similarity=0.0, late_rate=float("nan"),
            mean_delay_days=float("nan"), n_found=0, n_requested=k,
            features_used=[], sufficient=False,
            note="No comparable features available in the historical data.",
        )

    pool = history
    note = ""
    if require_same_route:
        mask = pd.Series(True, index=history.index)
        for col in EXACT_MATCH_FEATURES:
            if col in history.columns and col in row.index:
                mask &= history[col] == row[col]
        if mask.sum() >= min_results:
            pool = history[mask]
        else:
            note = (
                f"Only {int(mask.sum())} historical orders share this exact route, "
                f"too few to summarise. Showing the closest orders across all routes."
            )

    centre = pool[cols].median()
    spread = (pool[cols].quantile(0.75) - pool[cols].quantile(0.25)).replace(0, np.nan)
    spread = spread.fillna(pool[cols].std().replace(0, 1.0)).fillna(1.0)

    reference = _scaled(pool, cols, centre, spread)
    point = _scaled(row.to_frame().T, cols, centre, spread)[0]

    distances = np.sqrt(((reference - point) ** 2).mean(axis=1))
    order = np.argsort(distances)[: min(k, len(distances))]

    found = pool.iloc[order].copy()
    found["similarity"] = 1.0 / (1.0 + distances[order])

    late = found["is_late_delivery"] if "is_late_delivery" in found else pd.Series(dtype=float)
    delays = found.loc[found.get("is_late_delivery", 0) == 1, "delivery_delay_days"] \
        if "delivery_delay_days" in found else pd.Series(dtype=float)

    sufficient = len(found) >= min_results
    if not sufficient and not note:
        note = (
            f"Only {len(found)} comparable historical orders were found "
            f"(at least {min_results} are needed for a reliable summary)."
        )

    return SimilarOrders(
        orders=found,
        mean_similarity=float(found["similarity"].mean()) if len(found) else 0.0,
        late_rate=float(late.mean()) if len(late) else float("nan"),
        mean_delay_days=float(delays.mean()) if len(delays) else 0.0,
        n_found=len(found),
        n_requested=k,
        features_used=cols,
        sufficient=sufficient,
        note=note,
    )


def compare_to_nearest(row: pd.Series, similar: SimilarOrders,
                       features: list[str] | None = None) -> pd.DataFrame:
    """Feature-by-feature comparison against the single closest order.

    Differences are reported in original units, not scaled ones, because the
    reader needs "180 km further" rather than "0.4 standard deviations".
    """
    if similar.orders.empty:
        return pd.DataFrame()

    nearest = similar.orders.iloc[0]
    cols = features or similar.features_used

    rows = []
    for col in cols:
        current, other = row.get(col), nearest.get(col)
        if pd.isna(current) and pd.isna(other):
            continue
        difference = (
            current - other
            if pd.notna(current) and pd.notna(other)
            and isinstance(current, (int, float, np.number)) else np.nan
        )
        rows.append({
            "feature": col,
            "current_order": current,
            "most_similar": other,
            "difference": difference,
        })
    return pd.DataFrame(rows)
