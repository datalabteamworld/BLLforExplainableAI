"""Regression guard for the empirical Olist pipeline.

The project now carries two data environments: the empirical Olist pipeline,
which is the measured evidence behind the paper, and a synthetic Nigerian
environment used only for public demonstration.

The risk that creates is silent drift. A refactor made to accommodate the
synthetic environment could alter the Olist feature table by a fraction, and
nothing else in the suite would notice -- the published AUC, Brier score and
review-effect figures would quietly stop matching the code that claims to
produce them.

This module pins the Olist path by fingerprint. The values below were captured
from the pipeline *before* any environment abstraction existed. If a change to
shared code alters what the Olist pipeline produces, these fail.

They are deliberately brittle. A failure here is not a flaky test: it means a
published empirical result no longer reproduces, and the change needs
reverting or the paper needs rechecking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olist import cleaning, features
from tests.synthetic import make_tables

#: Captured from the pre-abstraction pipeline. Do not update these to make a
#: failing test pass -- investigate what changed in the Olist path instead.
BASELINE = {
    "rows": 2811,
    "columns": 89,
    "late_rate": 0.2170046247,
    "numeric_columns": 68,
}

#: Per-column (mean, std, min, max) for every numeric feature.
#:
#: An earlier version of this guard hashed the raw float64 bytes of the whole
#: table. That was too strict in the wrong way: a numpy patch release changes
#: the last bit of a floating-point sum, so the hash moved between machines
#: while nothing meaningful had changed, and the guard cried wolf.
#:
#: Column statistics compared at a tight relative tolerance catch everything
#: the hash caught -- any real change to cleaning, feature engineering, the
#: grain or the merge chain moves a mean by orders of magnitude more than
#: 1e-6 -- without failing on arithmetic noise.
COLUMN_STATS: dict[str, tuple[float, float, float, float]] = {
    "actual_delivery_days": (13.737353809978394, 5.6207017072832794, 3.233098141541215, 51.20074118584094),
    "approval_time_hours": (5.858354032781213, 5.909191122096506, 4.4482585833333333e-05, 47.52583206666278),
    "carnival_days_in_window": (0.27748132337246534, 1.0989423203269166, 0.0, 5.0),
    "carrier_pickup_days": (2.619894203276658, 1.6755487458359895, 0.07945746995809028, 13.234659213057732),
    "category_history_confidence": (0.8548093708505227, 0.1886447729098088, 0.0, 0.9676375404530745),
    "category_late_rate_to_date": (0.2035926207445001, 0.029218504234678897, 0.0, 0.5),
    "category_orders_to_date": (262.9302739238705, 162.56832818453594, 0.0, 598.0),
    "customer_lat": (-16.31902156908153, 6.666729883539557, -29.99257846802505, -3.0558543930512485),
    "customer_lng": (-48.924545474480766, 5.301573677544434, -59.97936980512249, -38.07768877620968),
    "customer_state_history_confidence": (0.7891569214262953, 0.20103603139167342, 0.0, 0.937888198757764),
    "customer_state_late_rate_to_date": (0.20425309481006382, 0.0371348245198008, 0.0, 0.5238095238095238),
    "customer_state_orders_to_date": (135.2856634649591, 81.698041401051, 0.0, 302.0),
    "days_from_black_friday": (0.16293134115972963, 105.03932389789699, -181.0, 182.0),
    "delivery_delay_days": (-9.132767283411331, 10.970552191626366, -34.034583941967256, 42.296296741396496),
    "estimated_delivery_days": (22.870121093389727, 9.258838705445047, 7.005011574074074, 38.9996875),
    "freight_ratio": (0.2458791696820209, 0.51201870546883, 0.0008785416209092906, 11.848416289592759),
    "has_listing_metadata": (0.9772323016720028, 0.14918863278400335, 0.0, 1.0),
    "has_review_comment": (1.0, 0.0, 1.0, 1.0),
    "holidays_in_window": (0.6108146567057986, 0.7990722455047442, 0.0, 3.0),
    "is_black_friday_week": (0.03557452863749555, 0.18525979367280576, 0.0, 1.0),
    "is_late_delivery": (0.21700462468872286, 0.4122791348491765, 0.0, 1.0),
    "is_multi_seller": (0.33938100320170755, 0.47358349305234687, 0.0, 1.0),
    "is_same_city": (0.14692280327285664, 0.3540919328162104, 0.0, 1.0),
    "is_same_state": (0.10032017075773746, 0.3004798723408337, 0.0, 1.0),
    "is_weekend_purchase": (0.2760583422269655, 0.4471255471989523, 0.0, 1.0),
    "market_late_rate_to_date": (0.2035247396912694, 0.01760605821582588, 0.0, 0.5),
    "max_installments": (4.031661330487371, 3.1680220566552744, 1.0, 10.0),
    "max_item_price": (139.36587335467803, 90.53429780110079, 1.47, 813.9),
    "max_volume_cm3": (236245.21202419067, 218417.95693368546, 2112.0, 1036386.0),
    "max_weight_g": (2088.237634408602, 1354.3389964876203, 81.0, 7546.0),
    "mean_density": (0.04551057967078123, 0.1287645362090042, 0.00022017790374622695, 1.5572916666666667),
    "mean_description_length": (1921.510071942446, 1001.1691714421406, 6.0, 3987.0),
    "mean_item_price": (119.89771611526146, 77.49140690595327, 1.47, 813.9),
    "mean_photos_qty": (10.318705035971224, 4.96065704011958, 1.0, 19.0),
    "n_items": (1.520811099252935, 0.7799837005993177, 1.0, 3.0),
    "n_payment_methods": (1.0174315190323728, 0.13881277947800982, 0.0, 2.0),
    "n_payment_records": (1.0508715759516187, 0.21977452424817565, 1.0, 2.0),
    "n_products": (1.5168979011028103, 0.7764201429165487, 1.0, 3.0),
    "n_sellers": (1.5094272500889363, 0.7682535951308953, 1.0, 3.0),
    "non_working_days_in_window": (7.267520455353966, 3.034957926363223, 2.0, 15.0),
    "non_working_ratio": (0.31991213799841867, 0.058730147414673126, 0.16686317509335358, 0.7724670666942547),
    "purchase_dow": (2.967627178939879, 1.9953731242760353, 0.0, 6.0),
    "purchase_hour": (11.381714692280328, 6.924900146644027, 0.0, 23.0),
    "purchase_month": (6.585200996086802, 3.406313186370733, 1.0, 12.0),
    "purchase_quarter": (2.51832088224831, 1.1099367547248102, 1.0, 4.0),
    "purchase_year": (2017.1970828886517, 0.6788141427946145, 2016.0, 2018.0),
    "purchased_on_black_friday": (0.000711490572749911, 0.026669033976839677, 0.0, 1.0),
    "purchased_on_carnival": (0.01351832088224831, 0.11550031012648751, 0.0, 1.0),
    "purchased_on_cyber_monday": (0.0032017075773745998, 0.05650302996884021, 0.0, 1.0),
    "purchased_on_holiday": (0.03023834934187122, 0.17127296095747027, 0.0, 1.0),
    "review_score": (2.9913854989231874, 1.42984264893339, 1.0, 5.0),
    "route_late_rate_to_date": (0.20354575808965805, 0.0447019398318902, 0.0, 0.5238095238095238),
    "route_orders_to_date": (15.569192458199929, 11.714384078125942, 0.0, 55.0),
    "seller_customer_distance_km": (1191.4105514917135, 591.6289969080945, 26.653617861965444, 3156.784433866218),
    "seller_history_confidence": (0.4183480159556499, 0.18957288176008655, 0.0, 0.7142857142857143),
    "seller_lat": (-15.9366257283831, 6.669550142818217, -28.572005660700015, -5.408248794125022),
    "seller_late_rate_to_date": (0.2025982335315012, 0.0467515881891239, 0.0, 0.5),
    "seller_lng": (-48.66616168782903, 5.665871656569597, -59.89967078485214, -38.883640433207745),
    "seller_orders_to_date": (17.93881181074351, 11.421299257841467, 0.0, 50.0),
    "seller_tenure_days": (353.0569346449794, 213.01346738191125, 0.0, 737.5416319444445),
    "shipping_limit_slack_days": (17.885062395417478, 9.61038691895821, -1.819189814814815, 36.9996875),
    "total_freight": (24.568612593383136, 18.445655941922983, 0.15, 109.61),
    "total_payment_value": (166.49849519743864, 123.03621020588731, 0.0, 1185.23),
    "total_price": (181.182924226254, 138.08436526509053, 1.47, 1224.2),
    "total_volume_cm3": (291482.7858413376, 288478.8137143084, 2112.0, 2304552.0),
    "total_weight_g": (2684.778726431875, 2095.714680191813, 0.0, 13661.0),
    "weekend_days_in_window": (6.663109213802917, 2.707835239017028, 2.0, 12.0),
    "working_days_in_window": (15.60260063803576, 6.496895899548325, 2.038530092592593, 28.9996875),
}

#: Relative tolerance for the statistics above. Loose enough to absorb
#: last-bit differences between library versions, far tighter than any
#: genuine pipeline change could hide under.
TOLERANCE = 1e-6

SEED = 12345
N_ORDERS = 3000


@pytest.fixture(scope="module")
def olist_table() -> pd.DataFrame:
    """The Olist modelling table, built exactly as the empirical path builds it."""
    tables = cleaning.clean_all(make_tables(n_orders=N_ORDERS, seed=SEED))[0]
    return features.build_modelling_table(tables)


def _stats(df: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    """Per-column mean, standard deviation, minimum and maximum."""
    num = df.select_dtypes("number")
    return {
        c: (float(num[c].mean()), float(num[c].std()),
            float(num[c].min()), float(num[c].max()))
        for c in sorted(num.columns)
    }


def test_olist_row_count_unchanged(olist_table):
    assert len(olist_table) == BASELINE["rows"]


def test_olist_column_count_unchanged(olist_table):
    """A new column in the Olist table means the feature space moved."""
    assert olist_table.shape[1] == BASELINE["columns"]


def test_olist_late_rate_unchanged(olist_table):
    assert olist_table["is_late_delivery"].mean() == pytest.approx(
        BASELINE["late_rate"], abs=1e-9
    )


def test_olist_numeric_columns_unchanged(olist_table):
    """The exact set of numeric features, by name."""
    found = _stats(olist_table)
    assert len(found) == BASELINE["numeric_columns"]
    assert set(found) == set(COLUMN_STATS), (
        "The numeric feature set changed.\n"
        f"  added:   {sorted(set(found) - set(COLUMN_STATS))}\n"
        f"  removed: {sorted(set(COLUMN_STATS) - set(found))}"
    )


def test_olist_numeric_values_unchanged(olist_table):
    """The strongest check: the distribution of every numeric feature.

    Reports every column that moved rather than stopping at the first, so a
    failure says what actually changed instead of only that something did.
    """
    found = _stats(olist_table)
    labels = ("mean", "std", "min", "max")

    drifted = []
    for column, expected in COLUMN_STATS.items():
        if column not in found:
            continue
        for label, want, got in zip(labels, expected, found[column]):
            if want != pytest.approx(got, rel=TOLERANCE, abs=1e-9):
                drifted.append(f"  {column}.{label}: {want!r} -> {got!r}")

    assert not drifted, (
        "The Olist feature table has changed. The paper's empirical figures "
        "were measured on the previous version. Revert the change, or "
        "re-measure and update the paper.\n\n" + "\n".join(drifted[:25])
    )


def test_olist_uses_the_brazilian_calendar_by_default(olist_table):
    """Calendar selection must default to Brazil with no argument passed.

    The synthetic environment supplies its own calendar. If that parameter
    ever defaults to anything else, Olist features silently change.
    """
    from olist import calendar_br

    tables = cleaning.clean_all(make_tables(n_orders=400, seed=7))[0]
    default = features.build_modelling_table(tables)
    explicit = features.build_modelling_table(tables, calendar=calendar_br)

    shared = [c for c in default.columns if c in explicit.columns]
    pd.testing.assert_frame_equal(default[shared], explicit[shared])


def test_olist_geolocation_uses_brazilian_bounds():
    """Coordinate filtering must default to Brazil's bounding box."""
    from olist.cleaning import BRAZIL_BOUNDS, clean_geolocation

    geo = pd.DataFrame({
        "geolocation_zip_code_prefix": ["01001", "02002", "03003"],
        "geolocation_lat": [-23.5, 6.5, -22.9],      # middle one is Nigeria
        "geolocation_lng": [-46.6, 3.4, -43.2],
    })
    cleaned, _ = clean_geolocation(geo)
    assert len(cleaned) == 2, "a Nigerian coordinate survived the Brazil filter"

    lat_lo, lat_hi, lng_lo, lng_hi = BRAZIL_BOUNDS
    assert lat_lo < -30 and lng_hi < 0, "BRAZIL_BOUNDS is not Brazil"
