"""Tests for cleaning rules, calendar logic and leakage guarantees.

Run with ``pytest`` from the project root.

The leakage tests matter most. Every defect corrected in the original
notebook -- carrier timings used as pre-dispatch features, SMOTE applied
outside the CV folds, sibling order-item rows split across train and test --
was invisible in the output and produced a *better*-looking score. Only an
assertion catches that class of bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olist import calendar_br as cb
from olist import cleaning, features, history, pipeline
from olist.config import CONFIG, LEAKAGE_COLUMNS, POST_DISPATCH_FEATURES, TARGET
from tests.synthetic import make_tables


@pytest.fixture(scope="module")
def tables():
    return make_tables(n_orders=2500, seed=7)


@pytest.fixture(scope="module")
def cleaned(tables):
    return cleaning.clean_all(tables)[0]


@pytest.fixture(scope="module")
def feature_table(cleaned):
    """The full modelling table, including history features and censoring."""
    return features.build_modelling_table(cleaned)


# ==========================================================================
# City normalisation
# ==========================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("sao paulo - sp", "sao paulo"),           # trailing state code
        ("são paulo", "sao paulo"),                 # accent folding
        ("sao  paulo", "sao paulo"),                # collapsed whitespace
        ("sao paulop", "sao paulo"),                # known typo
        ("santo andre/sao paulo", "santo andre"),   # city/state compound
        ("rio de janeiro / rio de janeiro", "rio de janeiro"),
        ("novo hamburgo, rio grande do sul, brasil", "novo hamburgo"),
        ("sbc/sp", "sao bernardo do campo"),        # abbreviation
        ("barbacena/ minas gerais", "barbacena"),
        ("santa barbara d´oeste", "santa barbara d'oeste"),
        ("rio de janeiro", "rio de janeiro"),       # state name that is a real city
        ("sao paulo", "sao paulo"),
    ],
)
def test_normalise_city(raw, expected):
    assert cleaning.normalise_city(raw) == expected


@pytest.mark.parametrize("raw", ["sp", "sp / sp", "04482255", "", "   ", None, np.nan])
def test_normalise_city_unrecoverable(raw):
    assert pd.isna(cleaning.normalise_city(raw))


def test_state_names_that_are_cities_survive():
    """São Paulo and Rio are both state and city names.

    Filtering segments by state *name* rather than state code discards the
    city itself -- the bug that turned "sao paulo - sp" into "sao paulo -".
    """
    for city in ("sao paulo", "rio de janeiro"):
        assert cleaning.normalise_city(f"{city} - sp") == city


# ==========================================================================
# Cleaning rules
# ==========================================================================


def test_payments_defects_corrected(tables):
    out, rep = cleaning.clean_payments(tables["order_payments"])
    assert not out["payment_type"].eq("not_defined").any()
    assert out["payment_installments"].min() >= 1
    assert len(out) == len(tables["order_payments"]), "cleaning must not drop rows"
    assert rep.actions


def test_products_columns_renamed_and_zero_weight_nulled(tables):
    out, _ = cleaning.clean_products(tables["products"])
    assert "product_name_length" in out.columns
    assert "product_name_lenght" not in out.columns
    assert not out["product_weight_g"].eq(0).any()


def test_translation_covers_every_product_category(cleaned):
    """The two untranslated categories silently produce NaN after the merge."""
    used = set(cleaned["products"]["product_category_name"].dropna())
    known = set(cleaned["category_translation"]["product_category_name"])
    assert used <= known, f"untranslated categories: {sorted(used - known)}"


def test_geolocation_is_unique_per_zip(cleaned):
    geo = cleaned["geolocation"]
    assert not geo["geolocation_zip_code_prefix"].duplicated().any()


# ==========================================================================
# Calendar
# ==========================================================================


def test_black_friday_is_fourth_friday_of_november():
    for year, expected in [(2016, "2016-11-25"), (2017, "2017-11-24"), (2018, "2018-11-23")]:
        assert cb.black_friday(year) == pd.Timestamp(expected)
        assert cb.black_friday(year).dayofweek == 4


def test_window_count_includes_start_day():
    """Half-open [start, end): an exclusive prefix sum, or New Year is missed."""
    cal = cb.build_daily_calendar("2016-12-01", "2017-02-01")
    starts = pd.Series(pd.to_datetime(["2017-01-01"]))
    ends = pd.Series(pd.to_datetime(["2017-01-15"]))
    assert cb.count_events_in_window(starts, ends, cal, "is_holiday").iloc[0] == 1


def test_carnival_window_captures_all_five_days():
    cal = cb.build_daily_calendar("2018-01-01", "2018-03-31")
    starts = pd.Series(pd.to_datetime(["2018-02-08"]))
    ends = pd.Series(pd.to_datetime(["2018-02-20"]))
    assert cb.count_events_in_window(starts, ends, cal, "is_carnival").iloc[0] == 5


def test_days_to_black_friday_uses_calendar_days():
    """Guards the ns/us unit mismatch that produced offsets of -17,000 days."""
    dates = pd.Series(pd.to_datetime(["2017-11-20", "2017-11-24", "2017-11-30"]))
    offsets = cb.days_to_nearest_black_friday(dates, [2016, 2017, 2018])
    assert offsets.tolist() == [-4.0, 0.0, 6.0]


def test_window_count_handles_nulls():
    cal = cb.build_daily_calendar("2017-01-01", "2017-12-31")
    starts = pd.Series(pd.to_datetime(["2017-01-01", None]))
    ends = pd.Series(pd.to_datetime([None, "2017-02-01"]))
    assert cb.count_events_in_window(starts, ends, cal, "is_holiday").isna().all()


# ==========================================================================
# Grain and leakage
# ==========================================================================


def test_merge_preserves_order_grain(cleaned):
    table = features.build_order_table(cleaned)
    assert len(table) == len(cleaned["orders"])
    assert not table["order_id"].duplicated().any()


def test_merge_raises_on_fanout(cleaned):
    """A non-unique join key must fail loudly, not silently duplicate rows."""
    broken = dict(cleaned)
    broken["customers"] = pd.concat([cleaned["customers"], cleaned["customers"].head(5)])
    with pytest.raises(ValueError, match="changed the row count"):
        features.build_order_table(broken)


def test_pre_dispatch_excludes_carrier_features(feature_table):
    cols = pipeline.feature_columns(feature_table, "pre_dispatch")
    for col in POST_DISPATCH_FEATURES:
        assert col not in cols, f"{col} is unknown when the order is placed"


def test_no_scenario_leaks_the_target(feature_table):
    for scenario in ("pre_dispatch", "post_dispatch"):
        cols = pipeline.feature_columns(feature_table, scenario)
        assert not set(cols) & set(LEAKAGE_COLUMNS)
        assert TARGET not in cols


def test_temporal_split_does_not_overlap(feature_table):
    train, test = pipeline.temporal_split(feature_table, "2018-03-31")
    assert train["order_purchase_timestamp"].max() < test["order_purchase_timestamp"].min()
    assert not set(train["order_id"]) & set(test["order_id"])


def test_smote_is_inside_the_pipeline(feature_table):
    """SMOTE must be a pipeline step so CV refits it per fold."""
    cols = pipeline.feature_columns(feature_table, "pre_dispatch")
    pipe = pipeline.build_pipeline(feature_table, cols, "logistic_regression")
    assert "smote" in dict(pipe.steps)
    assert list(dict(pipe.steps)).index("smote") < list(dict(pipe.steps)).index("clf")


def test_pipeline_fits_and_predicts_probabilities(feature_table):
    cols = pipeline.feature_columns(feature_table, "pre_dispatch")
    train, test = pipeline.temporal_split(feature_table, "2018-03-31")
    pipe = pipeline.build_pipeline(train, cols, "logistic_regression")
    pipe.fit(train[cols], train[TARGET])
    proba = pipe.predict_proba(test[cols])[:, 1]
    assert proba.shape == (len(test),)
    assert ((proba >= 0) & (proba <= 1)).all()


# ==========================================================================
# Threshold selection
# ==========================================================================


def test_chosen_threshold_beats_the_default_on_cost():
    from olist.evaluate import choose_threshold, expected_cost_curve

    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.08, 5000)
    proba = np.clip(rng.beta(2, 8, 5000) + 0.3 * y, 0, 1)

    choice, curve = choose_threshold(y, proba, CONFIG.cost)
    at_default = curve.iloc[(curve["threshold"] - 0.5).abs().argmin()]
    assert choice.expected_cost <= at_default["expected_cost"]
    assert 0.0 <= choice.threshold <= 1.0


def test_cost_curve_prefers_recall_when_misses_are_expensive():
    """Raising the cost of a miss must not lower the selected recall."""
    from dataclasses import replace

    from olist.evaluate import choose_threshold

    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.08, 5000)
    proba = np.clip(rng.beta(2, 8, 5000) + 0.3 * y, 0, 1)

    cheap, _ = choose_threshold(y, proba, replace(CONFIG.cost, cost_fn=10.0))
    dear, _ = choose_threshold(y, proba, replace(CONFIG.cost, cost_fn=200.0))
    assert dear.recall >= cheap.recall
    assert dear.threshold <= cheap.threshold


# ==========================================================================
# History features -- leakage is the whole risk here
# ==========================================================================


def _history_frame() -> pd.DataFrame:
    """Orders with deliberately awkward timing.

    Order B is purchased *after* A is purchased but *before* A is delivered.
    A naive expanding mean over purchase dates would let A's outcome inform
    B; only A's delivery date makes it observable, and that comes later.

    Order D belongs to a second seller and arrives on time, so the running
    marketplace prior is not degenerate at 1.0 — without it, smoothing has
    nothing to pull a sparse seller rate toward.
    """
    return pd.DataFrame({
        "order_id": ["A", "B", "D", "C"],
        "seller_id_primary": ["s1", "s1", "s2", "s1"],
        "customer_state": ["SP", "SP", "RJ", "SP"],
        "seller_state": ["SP", "SP", "RJ", "SP"],
        "product_category": ["x", "x", "y", "x"],
        "order_purchase_timestamp": pd.to_datetime(
            ["2018-01-01", "2018-01-05", "2018-01-06", "2018-03-01"]
        ),
        "order_delivered_customer_date": pd.to_datetime(
            ["2018-01-20", "2018-01-25", "2018-01-28", "2018-03-20"]
        ),
        "order_estimated_delivery_date": pd.to_datetime(
            ["2018-01-10", "2018-01-15", "2018-02-10", "2018-03-25"]
        ),
        "is_late_delivery": [1, 1, 0, 0],
    })


def test_history_ignores_outcomes_not_yet_observable():
    """Order B must not see order A: A was still in transit when B was placed."""
    out = history.add_history_features(_history_frame(), alpha=0.0)
    by_id = out.set_index("order_id")

    assert by_id.loc["A", "seller_orders_to_date"] == 0
    assert by_id.loc["B", "seller_orders_to_date"] == 0, (
        "B saw A's outcome, but A was delivered 2018-01-20, after B was "
        "purchased on 2018-01-05"
    )
    # C is placed after both A and B were delivered, so it sees both.
    assert by_id.loc["C", "seller_orders_to_date"] == 2
    assert by_id.loc["C", "seller_late_rate_to_date"] == pytest.approx(1.0)


def test_history_never_includes_the_order_itself():
    df = _history_frame()
    df["order_delivered_customer_date"] = df["order_purchase_timestamp"]
    out = history.add_history_features(df, alpha=0.0)
    assert out["seller_orders_to_date"].iloc[0] == 0


def test_history_smoothing_pulls_sparse_rates_toward_the_prior():
    """One late order out of one must not read as 100% risk."""
    out = history.add_history_features(_history_frame(), alpha=20.0)
    rate = out.set_index("order_id").loc["C", "seller_late_rate_to_date"]
    assert 0.0 < rate < 1.0


def test_history_confidence_grows_with_evidence(feature_table):
    out = history.add_history_features(feature_table)
    conf = out["seller_history_confidence"]
    assert conf.between(0, 1).all()
    assert conf.corr(out["seller_orders_to_date"]) > 0.5


def test_history_adds_no_rows_and_preserves_order(feature_table):
    out = history.add_history_features(feature_table)
    assert len(out) == len(feature_table)
    assert out.index.equals(feature_table.index)


def test_censoring_removes_orders_promised_near_the_cutoff(feature_table):
    out = history.apply_censoring_filter(feature_table, buffer_days=30)
    cutoff = feature_table["order_delivered_customer_date"].max()
    assert len(out) < len(feature_table)
    assert (out["order_estimated_delivery_date"] <= cutoff - pd.Timedelta(days=30)).all()


def test_calibration_improves_brier_score(feature_table):
    """The point of calibration: probabilities that mean what they say."""
    from sklearn.metrics import brier_score_loss

    cols = pipeline.feature_columns(feature_table, "pre_dispatch")
    train, test = pipeline.temporal_split(feature_table, "2018-03-31")
    if train[TARGET].sum() < 100 or test[TARGET].sum() < 20:
        pytest.skip("fixture too small for a meaningful calibration comparison")

    raw = pipeline.build_pipeline(train, cols, "logistic_regression")
    raw.fit(train[cols], train[TARGET])
    brier_raw = brier_score_loss(test[TARGET], raw.predict_proba(test[cols])[:, 1])

    cal = pipeline.fit_calibrated(
        pipeline.build_pipeline(train, cols, "logistic_regression"), train, cols
    )
    brier_cal = brier_score_loss(test[TARGET], cal.predict_proba(test[cols])[:, 1])

    assert brier_cal < brier_raw, f"calibration made it worse: {brier_cal} vs {brier_raw}"


def test_unbounded_time_counters_are_excluded(feature_table):
    """Counters that grow with time leave the training range under a temporal split."""
    from olist.config import UNBOUNDED_TIME_COUNTERS

    cols = pipeline.feature_columns(feature_table, "pre_dispatch")
    assert not set(cols) & set(UNBOUNDED_TIME_COUNTERS)
    # The bounded equivalents survive: they carry the same signal on a stable scale.
    assert "seller_late_rate_to_date" in cols
    assert "seller_history_confidence" in cols


def test_no_history_flag_drops_history_features(feature_table):
    with_hist = pipeline.feature_columns(feature_table, "pre_dispatch", use_history=True)
    without = pipeline.feature_columns(feature_table, "pre_dispatch", use_history=False)
    assert len(without) < len(with_hist)
    assert not any(c.endswith(("_to_date", "_history_confidence")) for c in without)


def test_sigmoid_calibration_preserves_ranking(feature_table):
    """Platt scaling is strictly monotone, so AUC must not move.

    Isotonic is a step function and its ties reduce AUC; that is why sigmoid
    is the default.
    """
    from sklearn.metrics import roc_auc_score

    cols = pipeline.feature_columns(feature_table, "pre_dispatch")
    train, test = pipeline.temporal_split(feature_table, "2018-03-31")
    if train[TARGET].sum() < 100 or test[TARGET].sum() < 20:
        pytest.skip("fixture too small")

    ordered = train.sort_values("order_purchase_timestamp")
    fit_part = ordered.iloc[: int(len(ordered) * 0.8)]

    raw = pipeline.build_pipeline(train, cols, "logistic_regression")
    raw.fit(fit_part[cols], fit_part[TARGET])
    auc_raw = roc_auc_score(test[TARGET], raw.predict_proba(test[cols])[:, 1])

    cal = pipeline.fit_calibrated(
        pipeline.build_pipeline(train, cols, "logistic_regression"), train, cols
    )
    auc_cal = roc_auc_score(test[TARGET], cal.predict_proba(test[cols])[:, 1])
    assert auc_cal == pytest.approx(auc_raw, abs=1e-6)


def test_temporal_cv_is_the_tuning_default():
    """Random CV folds reward memorising seasonality a forward test cannot use."""
    assert CONFIG.tuning.temporal_cv is True
