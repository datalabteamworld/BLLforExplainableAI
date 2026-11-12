"""Nigerian commercial geography for the synthetic demonstration environment.

Coordinates and state assignments are real; everything built on top of them is
simulated. The distinction matters: a reader should be able to check that Kano
is where this module says it is, while understanding that the order volumes and
delivery outcomes attached to it are generated.

Three design choices worth stating.

**Volume is not concentrated in Lagos alone.** Lagos dominates Nigerian
e-commerce, but a dataset where 80% of orders share one origin produces a
model that has learned almost nothing about routes. Seller weights here spread
across the four genuine commercial clusters -- Lagos, Onitsha/Aba in the
south-east, Kano in the north, and Abuja -- which is both closer to reality
and a more useful demonstration.

**Corridor difficulty is a modelled quantity, not a measured one.** The
``friction`` multiplier on each corridor encodes assumptions about road
quality, security checkpoints and carrier density. It is an assumption. It is
documented in ASSUMPTIONS.md and must not be read as a measurement of Nigerian
road conditions.

**Distance is great-circle, not road distance.** Road distance in Nigeria can
exceed great-circle distance considerably, particularly across the Niger and
Benue. The corridor friction multiplier absorbs some of that, but the distance
column itself is straight-line, same as the Olist pipeline computes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class City:
    """One commercial centre.

    ``seller_weight`` and ``customer_weight`` differ deliberately: Aba and
    Onitsha are manufacturing and wholesale hubs that ship far more than they
    receive, while Abuja is a consumption centre with comparatively little
    outbound trade.
    """

    name: str
    state: str
    zone: str          # geopolitical zone
    lat: float
    lng: float
    seller_weight: float
    customer_weight: float


#: Eighteen centres covering all six geopolitical zones.
CITIES: tuple[City, ...] = (
    # --- South West ---------------------------------------------------------
    City("Lagos",         "Lagos",     "South West",  6.5244,  3.3792, 0.30, 0.22),
    City("Ibadan",        "Oyo",       "South West",  7.3775,  3.9470, 0.05, 0.07),
    City("Abeokuta",      "Ogun",      "South West",  7.1475,  3.3619, 0.03, 0.04),
    City("Akure",         "Ondo",      "South West",  7.2571,  5.2058, 0.01, 0.03),
    # --- South East ---------------------------------------------------------
    City("Onitsha",       "Anambra",   "South East",  6.1414,  6.8023, 0.11, 0.04),
    City("Aba",           "Abia",      "South East",  5.1066,  7.3667, 0.10, 0.03),
    City("Enugu",         "Enugu",     "South East",  6.5244,  7.5186, 0.03, 0.05),
    City("Owerri",        "Imo",       "South East",  5.4836,  7.0333, 0.02, 0.04),
    # --- South South --------------------------------------------------------
    City("Port Harcourt", "Rivers",    "South South", 4.8156,  7.0498, 0.05, 0.08),
    City("Benin City",    "Edo",       "South South", 6.3350,  5.6037, 0.03, 0.05),
    City("Uyo",           "Akwa Ibom", "South South", 5.0377,  7.9128, 0.01, 0.03),
    City("Warri",         "Delta",     "South South", 5.5160,  5.7500, 0.01, 0.03),
    # --- North Central ------------------------------------------------------
    City("Abuja",         "FCT",       "North Central", 9.0579, 7.4951, 0.08, 0.14),
    City("Ilorin",        "Kwara",     "North Central", 8.4966, 4.5421, 0.02, 0.04),
    City("Jos",           "Plateau",   "North Central", 9.8965, 8.8583, 0.02, 0.03),
    # --- North West ---------------------------------------------------------
    City("Kano",          "Kano",      "North West",  12.0022, 8.5920, 0.08, 0.07),
    City("Kaduna",        "Kaduna",    "North West",  10.5105, 7.4165, 0.03, 0.05),
    # --- North East ---------------------------------------------------------
    City("Maiduguri",     "Borno",     "North East",  11.8311, 13.1510, 0.02, 0.03),
)

#: Multiplier on expected transit time for a shipment leaving each zone.
#: An assumption about carrier density and road quality, not a measurement.
ZONE_FRICTION: dict[str, float] = {
    "South West": 1.00,
    "South East": 1.08,
    "South South": 1.12,
    "North Central": 1.15,
    "North West": 1.28,
    "North East": 1.45,
}

#: Additional friction when a shipment crosses between zones, which typically
#: means a trunk-route handover rather than a local delivery.
INTERZONE_PENALTY = 1.18

#: Crossing the Niger or Benue adds a bridge bottleneck. South-eastern and
#: northern traffic to the south west passes through a small number of
#: crossings, which is a recognised source of delay.
RIVER_CROSSING_ZONES = {
    frozenset({"South West", "South East"}),
    frozenset({"South West", "South South"}),
    frozenset({"North West", "South East"}),
    frozenset({"North East", "South West"}),
}
RIVER_CROSSING_PENALTY = 1.10


def city_frame() -> pd.DataFrame:
    """The city table as a DataFrame, with weights normalised to sum to one."""
    df = pd.DataFrame([vars(c) for c in CITIES])
    df["seller_weight"] /= df["seller_weight"].sum()
    df["customer_weight"] /= df["customer_weight"].sum()
    df["friction"] = df["zone"].map(ZONE_FRICTION)
    return df


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Great-circle distance in kilometres.

    Same formula the Olist feature pipeline uses, so the resulting
    ``seller_customer_distance_km`` column is computed identically in both
    environments.
    """
    lat1, lng1, lat2, lng2 = map(np.radians, (np.asarray(lat1, dtype="float64"),
                                              np.asarray(lng1, dtype="float64"),
                                              np.asarray(lat2, dtype="float64"),
                                              np.asarray(lng2, dtype="float64")))
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def corridor_friction(seller_zone: pd.Series, customer_zone: pd.Series) -> pd.Series:
    """Combined difficulty multiplier for each seller-to-customer corridor.

    Three components, multiplied: the origin zone's own friction, a penalty
    for crossing zones at all, and a further penalty for corridors that must
    cross a major river.
    """
    base = seller_zone.map(ZONE_FRICTION).astype("float64")

    crosses = (seller_zone != customer_zone)
    base = base * np.where(crosses, INTERZONE_PENALTY, 1.0)

    pairs = [frozenset({s, c}) for s, c in zip(seller_zone, customer_zone)]
    river = np.array([p in RIVER_CROSSING_ZONES for p in pairs])
    base = base * np.where(river, RIVER_CROSSING_PENALTY, 1.0)

    return pd.Series(base, index=seller_zone.index)


def zip_prefix(city_index: int, rng: np.random.Generator, n: int) -> list[str]:
    """Synthetic five-digit postcode prefixes, clustered by city.

    Nigeria uses six-digit postcodes rather than Brazil's five, but the
    pipeline treats the prefix purely as a categorical join key between the
    address tables and the geolocation table. A five-digit synthetic code
    keeps the schema identical without implying these are real NIPOST codes.
    """
    base = 10_000 + city_index * 5_000
    return [f"{base + int(o):05d}" for o in rng.integers(0, 4_000, n)]
