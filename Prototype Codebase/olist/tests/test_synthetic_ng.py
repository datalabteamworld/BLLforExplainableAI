"""Tests for the synthetic Nigerian demonstration environment.

Two things are under test here. The first is ordinary: the generator produces
relationally consistent data that the existing pipeline can consume. The
second matters more — that the two environments stay separate, and that the
synthetic one cannot be mistaken for the empirical one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olist import calendar_ng, cleaning, environments, features
from olist.cleaning import NIGERIA_BOUNDS
from olist.synthetic_ng import geography as geo
from olist.synthetic_ng.generate import GeneratorConfig, generate

CFG = GeneratorConfig(n_orders=2500, seed=7)


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return generate(CFG)


@pytest.fixture(scope="module")
def modelling(tables) -> pd.DataFrame:
    cleaned, _ = cleaning.clean_all(tables)
    cleaned["geolocation"], _ = cleaning.clean_geolocation(
        tables["geolocation"], bounds=NIGERIA_BOUNDS, region="Nigeria"
    )
    return features.build_modelling_table(cleaned, calendar=calendar_ng)


# --- Schema compatibility -----------------------------------------------------


def test_emits_all_nine_olist_tables(tables):
    """Schema parity is what lets one pipeline serve both environments."""
    from olist.config import RAW_FILES

    assert set(tables) == set(RAW_FILES)


def test_pipeline_consumes_it_unmodified(modelling):
    """The whole Olist feature pipeline runs on this with no special casing."""
    assert len(modelling) > 1_000
    assert "is_late_delivery" in modelling.columns
    assert "seller_customer_distance_km" in modelling.columns


# --- Relational integrity -----------------------------------------------------


def test_no_orphan_records(tables):
    orders = set(tables["orders"]["order_id"])
    customers = set(tables["customers"]["customer_id"])
    products = set(tables["products"]["product_id"])
    sellers = set(tables["sellers"]["seller_id"])

    assert set(tables["orders"]["customer_id"]) <= customers
    assert set(tables["order_items"]["order_id"]) <= orders
    assert set(tables["order_items"]["product_id"]) <= products
    assert set(tables["order_items"]["seller_id"]) <= sellers
    assert set(tables["order_payments"]["order_id"]) <= orders
    assert set(tables["order_reviews"]["order_id"]) <= orders


def test_every_order_has_at_least_one_item(tables):
    with_items = set(tables["order_items"]["order_id"])
    assert set(tables["orders"]["order_id"]) <= with_items


# --- Temporal coherence -------------------------------------------------------


def test_delivery_never_precedes_carrier_handover(tables):
    """The bug the first generator design had, pinned so it cannot return."""
    o = tables["orders"]
    delivered = o[o["order_delivered_customer_date"].notna()]
    impossible = (
        delivered["order_delivered_customer_date"]
        < delivered["order_delivered_carrier_date"]
    )
    # Only the deliberately injected bad-timestamp rows may violate this.
    assert impossible.mean() < 0.005


def test_undelivered_orders_have_no_delivery_date(tables):
    o = tables["orders"]
    assert o.loc[o["order_status"] != "delivered",
                 "order_delivered_customer_date"].isna().all()


# --- The prediction problem ---------------------------------------------------


def test_late_rate_is_realistic(modelling):
    """Near the Olist base rate, so the same imbalance problem is exercised."""
    assert 0.04 < modelling["is_late_delivery"].mean() < 0.16


def test_target_is_not_deterministic(modelling):
    """A threshold rule would make the demonstration dishonestly decisive.

    Orders in the top distance decile must not be uniformly late, and orders
    in the bottom decile must not be uniformly on time.
    """
    d = modelling["seller_customer_distance_km"]
    far = modelling.loc[d >= d.quantile(0.9), "is_late_delivery"]
    near = modelling.loc[d <= d.quantile(0.1), "is_late_delivery"]

    assert 0 < far.mean() < 0.9, "distant orders are nearly always late"
    assert near.mean() > 0, "near orders are never late"


def test_injected_relationships_have_the_intended_sign(modelling):
    """Dispatch lag should raise lateness. If it does not, the generator broke."""
    late = modelling["is_late_delivery"] == 1
    assert (modelling.loc[late, "carrier_pickup_days"].mean()
            > modelling.loc[~late, "carrier_pickup_days"].mean())


# --- Nigerian context ---------------------------------------------------------


def test_uses_nigerian_states(tables):
    states = set(tables["customers"]["customer_state"])
    assert {"Lagos", "Kano", "FCT"} <= states
    assert not {"SP", "RJ", "MG"} & states, "Brazilian states leaked in"


def test_volume_is_not_concentrated_in_one_city(tables):
    """A single-origin dataset has no route variation to learn from."""
    share = tables["sellers"]["seller_city"].value_counts(normalize=True)
    assert share.iloc[0] < 0.45


def test_all_six_zones_represented():
    assert len(set(c.zone for c in geo.CITIES)) == 6


def test_distances_match_reality():
    """Coordinates are real, so distances should be checkable."""
    df = geo.city_frame().set_index("name")
    lagos, kano = df.loc["Lagos"], df.loc["Kano"]
    km = float(geo.haversine_km(lagos.lat, lagos.lng, kano.lat, kano.lng))
    assert 780 < km < 880, "Lagos-Kano should be roughly 825 km straight-line"


def test_calendar_includes_both_eids():
    """Eid matters for northern corridors and moves each year."""
    for year in (2023, 2024):
        assert len(calendar_ng.eid_periods(year)) >= 4


def test_nigerian_calendar_matches_the_brazilian_interface():
    """Shared feature code calls both, so the columns must be identical."""
    from olist import calendar_br

    ng = calendar_ng.build_daily_calendar("2024-01-01", "2024-06-30")
    br = calendar_br.build_daily_calendar("2024-01-01", "2024-06-30")
    assert list(ng.columns) == list(br.columns)


# --- Reproducibility ----------------------------------------------------------


def test_generation_is_deterministic():
    a = generate(GeneratorConfig(n_orders=500, seed=99))
    b = generate(GeneratorConfig(n_orders=500, seed=99))
    pd.testing.assert_frame_equal(a["orders"], b["orders"])


def test_different_seeds_give_different_data():
    a = generate(GeneratorConfig(n_orders=500, seed=1))
    b = generate(GeneratorConfig(n_orders=500, seed=2))
    assert not a["orders"]["order_delivered_customer_date"].equals(
        b["orders"]["order_delivered_customer_date"]
    )


# --- Environment isolation ----------------------------------------------------


def test_environments_use_separate_directories():
    """Isolation is structural, not a matter of discipline."""
    olist, ng = environments.get("olist"), environments.get("synthetic_ng")
    assert olist.raw_dir != ng.raw_dir
    assert olist.processed_path != ng.processed_path
    assert olist.models_dir != ng.models_dir
    assert ng.raw_dir.resolve() not in olist.raw_dir.resolve().parents


def test_mismatched_model_is_refused():
    """Feature names match across environments, so nothing else would catch it."""
    artefact = {environments.ENVIRONMENT_KEY: "synthetic_ng"}
    with pytest.raises(ValueError, match="trained in"):
        environments.check_artefact(artefact, environments.get("olist"))


def test_unmarked_artefact_is_treated_as_olist():
    """Models trained before the key existed are Olist models."""
    environments.check_artefact({}, environments.get("olist"))


def test_synthetic_environment_carries_a_disclosure():
    ng, olist = environments.get("synthetic_ng"), environments.get("olist")
    assert ng.is_synthetic and ng.disclosure
    assert "simulated" in ng.disclosure.lower()
    assert olist.disclosure is None


def test_default_environment_is_the_empirical_one():
    """A caller that names nothing gets the data the paper's results rest on."""
    assert environments.get().key == "olist"
    assert environments.DEFAULT.kind == "empirical"


def test_synthetic_environment_uses_naira():
    assert environments.get("synthetic_ng").currency == "NGN"
    assert environments.get("olist").currency == "BRL"
