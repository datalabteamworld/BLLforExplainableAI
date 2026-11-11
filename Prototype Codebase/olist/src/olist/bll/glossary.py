"""Feature-to-business-language glossary.

The BLL's *interpret* function turns a feature name and its attribution into a
sentence an operations lead can read. That mapping is domain knowledge, not
something a model can infer, so it lives here as an explicit, auditable table
rather than being generated on the fly.

Each entry carries the phrasing for both directions of influence, because
``seller_customer_distance_km`` raising risk and lowering it are different
business statements, not one statement with a sign attached.

Keeping this table separate is also what makes the layer domain-agnostic in
the sense the paper claims: porting BLL to credit risk or clinical triage
means writing a new glossary, not rewriting the reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureNarrative:
    """How one feature is described to a business reader.

    ``raises`` and ``lowers`` are complete sentences describing what the
    feature's value means for this order. ``lever`` marks features the
    business can actually change, which is what separates an explanation
    from a recommendation.
    """

    label: str
    raises: str
    lowers: str
    unit: str = ""
    lever: bool = False


GLOSSARY: dict[str, FeatureNarrative] = {
    # --- Promise and timing -------------------------------------------------
    "estimated_delivery_days": FeatureNarrative(
        label="Delivery promise length",
        raises="The promised window is tight for this route, leaving little slack for delay.",
        lowers="The promised window is generous, absorbing normal variation in transit.",
        unit="days",
        lever=True,
    ),
    "working_days_in_window": FeatureNarrative(
        label="Working days available",
        raises="Few working days fall inside the promised window.",
        lowers="The promised window contains plenty of working days.",
        unit="days",
    ),
    "non_working_days_in_window": FeatureNarrative(
        label="Holidays and weekends in transit",
        raises="Weekends or public holidays fall inside the delivery window, pausing the network.",
        lowers="The delivery window is largely clear of holidays and weekends.",
        unit="days",
    ),
    "carnival_days_in_window": FeatureNarrative(
        label="Carnival period",
        raises="Carnival falls inside the delivery window; logistics capacity drops sharply.",
        lowers="The delivery window avoids the Carnival shutdown.",
        unit="days",
    ),
    "shipping_limit_slack_days": FeatureNarrative(
        label="Dispatch deadline slack",
        raises="The seller's dispatch deadline leaves little margin before the customer promise.",
        lowers="There is comfortable margin between the dispatch deadline and the promise.",
        unit="days",
        lever=True,
    ),
    "carrier_pickup_days": FeatureNarrative(
        label="Time to carrier handover",
        raises="This parcel is slow leaving the warehouse.",
        lowers="This parcel reached the carrier quickly.",
        unit="days",
        lever=True,
    ),
    "approval_time_hours": FeatureNarrative(
        label="Payment approval delay",
        raises="Payment approval took longer than usual, delaying the start of fulfilment.",
        lowers="Payment cleared quickly, so fulfilment could start immediately.",
        unit="hours",
    ),

    # --- Route and geography ------------------------------------------------
    "seller_customer_distance_km": FeatureNarrative(
        label="Shipping distance",
        raises="The seller is far from the customer on this lane.",
        lowers="Seller and customer are close together.",
        unit="km",
    ),
    "is_same_state": FeatureNarrative(
        label="Interstate shipment",
        raises="This is an interstate shipment, crossing more of the carrier network.",
        lowers="Seller and customer are in the same state.",
    ),
    # Coordinates and administrative areas stand in for corridor conditions the
    # data does not record directly -- road quality, carrier density, river and
    # border crossings. They are described as corridor effects rather than as
    # properties of a place, because the model has learned where delays cluster,
    # not why. None is a lever: an operations lead cannot move a city.
    "customer_state": FeatureNarrative(
        label="Destination region",
        raises="Deliveries into this region have run late more often than average.",
        lowers="This region has a better-than-average delivery record.",
    ),
    "seller_state": FeatureNarrative(
        label="Origin region",
        raises="Dispatches from this region have run late more often than average.",
        lowers="This region dispatches more reliably than average.",
    ),
    "customer_city": FeatureNarrative(
        label="Destination city",
        raises="Last-mile delivery into this city has been slower than average.",
        lowers="Last-mile delivery into this city has been faster than average.",
    ),
    "seller_city": FeatureNarrative(
        label="Origin city",
        raises="Collections from this city have been slower than average.",
        lowers="Collections from this city have been faster than average.",
    ),
    "customer_lat": FeatureNarrative(
        label="Destination position (north-south)",
        raises="The destination sits on a corridor that has run slower than average.",
        lowers="The destination sits on a corridor that has run faster than average.",
        unit="degrees",
    ),
    "customer_lng": FeatureNarrative(
        label="Destination position (east-west)",
        raises="The destination sits on a corridor that has run slower than average.",
        lowers="The destination sits on a corridor that has run faster than average.",
        unit="degrees",
    ),
    "seller_lat": FeatureNarrative(
        label="Origin position (north-south)",
        raises="The origin sits on a corridor that has run slower than average.",
        lowers="The origin sits on a corridor that has run faster than average.",
        unit="degrees",
    ),
    "seller_lng": FeatureNarrative(
        label="Origin position (east-west)",
        raises="The origin sits on a corridor that has run slower than average.",
        lowers="The origin sits on a corridor that has run faster than average.",
        unit="degrees",
    ),
    "is_same_city": FeatureNarrative(
        label="Same-city delivery",
        raises="Seller and customer are in different cities, so the parcel enters the long-haul network.",
        lowers="Seller and customer are in the same city, so the parcel stays on local routes.",
    ),
    "route_late_rate_to_date": FeatureNarrative(
        label="Route track record",
        raises="This lane has run late more often than average.",
        lowers="This lane has a good on-time record.",
    ),

    # --- Seller behaviour ---------------------------------------------------
    "seller_late_rate_to_date": FeatureNarrative(
        label="Seller dispatch record",
        raises="This seller's recent orders have left the warehouse late.",
        lowers="This seller has a strong recent dispatch record.",
        lever=True,
    ),
    "seller_history_confidence": FeatureNarrative(
        label="Evidence behind the seller record",
        raises="The seller has enough completed orders for their record to be meaningful.",
        lowers="This seller has few completed orders, so their record is weak evidence.",
    ),
    "is_multi_seller": FeatureNarrative(
        label="Multiple sellers",
        raises="The order spans several sellers and ships only when the slowest dispatches.",
        lowers="A single seller fulfils the whole order.",
    ),
    "n_sellers": FeatureNarrative(
        label="Sellers involved",
        raises="Several sellers must each dispatch before this order completes.",
        lowers="Few sellers are involved in fulfilment.",
    ),

    # --- Parcel -------------------------------------------------------------
    "total_weight_g": FeatureNarrative(
        label="Parcel weight",
        raises="This is a heavy consignment, attracting slower handling.",
        lowers="Standard parcel weight, with no heavy-item handling surcharge.",
        unit="g",
    ),
    "max_weight_g": FeatureNarrative(
        label="Heaviest item",
        raises="The order contains a heavy item requiring special handling.",
        lowers="No single item is heavy enough to need special handling.",
        unit="g",
    ),
    "total_volume_cm3": FeatureNarrative(
        label="Parcel volume",
        raises="The consignment is bulky for its weight, limiting vehicle options.",
        lowers="The consignment is compact.",
        unit="cm3",
    ),
    "n_products": FeatureNarrative(
        label="Distinct products",
        raises="The order contains several different products, each needing separate picking.",
        lowers="Few distinct products to pick and consolidate.",
    ),
    "mean_density": FeatureNarrative(
        label="Parcel density",
        raises="Dense, heavy-for-size items attract slower handling.",
        lowers="Density is unremarkable for this parcel.",
    ),
    "max_volume_cm3": FeatureNarrative(
        label="Largest item volume",
        raises="One item is bulky enough to limit vehicle and handling options.",
        lowers="No single item is unusually bulky.",
        unit="cm3",
    ),
    "mean_item_price": FeatureNarrative(
        label="Average item price",
        raises="Higher-value items, typical of slower specialist fulfilment.",
        lowers="Routine item values.",
    ),
    "max_item_price": FeatureNarrative(
        label="Most expensive item",
        raises="A high-value item that may need signature or insured handling.",
        lowers="No item requires special value handling.",
    ),
    "market_late_rate_to_date": FeatureNarrative(
        label="Marketplace late rate",
        raises="Platform-wide lateness was running high when this order was placed.",
        lowers="The platform was performing well when this order was placed.",
    ),
    "category_late_rate_to_date": FeatureNarrative(
        label="Category track record",
        raises="This product category has run late more often than average.",
        lowers="This category has a good on-time record.",
    ),
    "customer_state_late_rate_to_date": FeatureNarrative(
        label="Destination track record",
        raises="This destination state has run late more often than average.",
        lowers="This destination has a good on-time record.",
    ),
    "n_items": FeatureNarrative(
        label="Items in order",
        raises="A multi-item order needs consolidation before dispatch.",
        lowers="A single-item order ships directly.",
    ),
    "freight_ratio": FeatureNarrative(
        label="Freight cost share",
        raises="Freight is high relative to order value, typical of difficult routes.",
        lowers="Freight is proportionate to order value.",
    ),
    "total_freight": FeatureNarrative(
        label="Freight cost",
        raises="High freight cost on this lane.",
        lowers="Freight cost is unremarkable for this lane.",
    ),

    # --- Commercial ---------------------------------------------------------
    "total_price": FeatureNarrative(
        label="Order value",
        raises="A high-value order, typical of slower specialist fulfilment.",
        lowers="A routine order value.",
    ),
    "purchase_month": FeatureNarrative(
        label="Time of year",
        raises="This order falls in a month when the network historically runs behind.",
        lowers="This order falls in a quieter month for the delivery network.",
    ),
    "purchase_year": FeatureNarrative(
        label="Platform era",
        raises="Placed during a period of weaker platform delivery performance.",
        lowers="Placed during a period of stronger platform performance.",
    ),
    "days_from_black_friday": FeatureNarrative(
        label="Proximity to Black Friday",
        raises="Close to the Black Friday peak, when the network is under strain.",
        lowers="Well away from the Black Friday peak.",
        unit="days",
    ),
    "weekend_days_in_window": FeatureNarrative(
        label="Weekends in transit",
        raises="Weekends fall inside the delivery window, when the network slows.",
        lowers="The delivery window is largely free of weekends.",
        unit="days",
    ),
    "holidays_in_window": FeatureNarrative(
        label="Public holidays in transit",
        raises="Public holidays fall inside the delivery window.",
        lowers="No public holidays fall inside the delivery window.",
        unit="days",
    ),
    "non_working_ratio": FeatureNarrative(
        label="Share of window that is non-working",
        raises="Much of the promised window falls on non-working days.",
        lowers="Most of the promised window consists of working days.",
    ),
    "is_same_city": FeatureNarrative(
        label="Same-city delivery",
        raises="Seller and customer are in different cities.",
        lowers="Seller and customer are in the same city.",
    ),
    "route_orders_to_date": FeatureNarrative(
        label="Route volume to date",
        raises="This lane has carried many prior orders.",
        lowers="This lane is lightly used.",
    ),
    "purchase_dow": FeatureNarrative(
        label="Day of purchase",
        raises="Ordered late in the week, so fulfilment starts after the weekend.",
        lowers="Ordered early in the week, with full working days ahead.",
    ),
    "purchase_hour": FeatureNarrative(
        label="Time of purchase",
        raises="Ordered outside working hours, delaying pick-and-pack by a shift.",
        lowers="Ordered during working hours and picked up in the same cycle.",
    ),
    "is_black_friday_week": FeatureNarrative(
        label="Black Friday peak",
        raises="Placed during Black Friday week, when the network is at peak load.",
        lowers="Placed outside the Black Friday peak.",
    ),
}


def describe(feature: str, raises_risk: bool) -> tuple[str, str]:
    """Return ``(label, sentence)`` for a feature and direction of influence.

    Unknown features fall back to their raw name. That is deliberate: a
    silent generic phrase would let an unmapped feature masquerade as
    business language it has not actually been given, which would break the
    traceability the framework requires.
    """
    entry = GLOSSARY.get(feature)
    if entry is None:
        direction = "increases" if raises_risk else "reduces"
        return feature, f"`{feature}` {direction} the predicted risk (no business mapping defined)."
    return entry.label, (entry.raises if raises_risk else entry.lowers)


def is_lever(feature: str) -> bool:
    """True when the business can act on this feature before delivery."""
    entry = GLOSSARY.get(feature)
    return bool(entry and entry.lever)


def coverage(features: list[str]) -> float:
    """Share of features with a business mapping.

    The paper requires every business statement to trace back to its
    originating attribution. Low coverage means the layer is silently
    dropping drivers it cannot express, so this is reported rather than
    hidden.
    """
    if not features:
        return 1.0
    return sum(f in GLOSSARY for f in features) / len(features)
