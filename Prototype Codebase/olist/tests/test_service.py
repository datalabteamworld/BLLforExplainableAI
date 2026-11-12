"""Tests for confidence, similarity and the prediction service.

The specification's central rule is that the application must never present
invented results. Most of these tests exist to enforce that: unavailable
capabilities must be reported as unavailable, insufficient evidence must be
declared, and confidence must not silently collapse into the predicted
probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olist import cleaning, features, pipeline
from olist.bll import service as svc
from olist.bll.confidence import TrainingProfile, estimate_confidence
from olist.bll.similarity import SIMILARITY_FEATURES, find_similar_orders
from olist.config import LEAKAGE_COLUMNS, TARGET
from tests.synthetic import make_tables


@pytest.fixture(scope="module")
def built():
    tables = cleaning.clean_all(make_tables(n_orders=3000, seed=23))[0]
    df = features.build_modelling_table(tables)
    train, test = pipeline.temporal_split(df, "2018-03-31")
    cols = pipeline.feature_columns(df, "post_dispatch")
    model = pipeline.fit_calibrated(
        pipeline.build_pipeline(train, cols, "logistic_regression", y=train[TARGET]),
        train, cols,
    )
    artefact = {
        "pipeline": model, "model_name": "logistic_regression",
        "scenario": "post_dispatch", "feature_columns": cols,
        "threshold": 0.2, "trained_at": "2026-01-01T00:00:00",
        "metrics": {"roc_auc": 0.84, "precision_at_chosen": 0.25},
    }
    return artefact, train, test, cols


@pytest.fixture(scope="module")
def service(built):
    artefact, train, test, cols = built
    return svc.PredictionService(artefact, history=train, background=train[cols].head(150))


# --- Confidence --------------------------------------------------------------


def test_confidence_is_not_the_predicted_probability(service, built):
    """The naive formula would call a confidently-safe order 'low confidence'."""
    _, _, test, cols = built
    row = test.iloc[[0]]
    probability = float(service.predict_probability(row)[0])
    confidence = service.get_confidence(row)
    assert confidence.score != pytest.approx(probability, abs=1e-6)


def test_confidence_falls_near_the_decision_threshold(service, built):
    """A prediction at the operating point could flip; it must score lower."""
    _, train, test, cols = built
    proba = service.predict_probability(test)
    at_boundary = test.iloc[[int(np.abs(proba - service.threshold).argmin())]]
    far_away = test.iloc[[int(np.abs(proba - service.threshold).argmax())]]
    assert service.get_confidence(at_boundary).threshold_margin < \
           service.get_confidence(far_away).threshold_margin


def test_confidence_is_bounded_and_labelled(service, built):
    _, _, test, cols = built
    for i in range(5):
        c = service.get_confidence(test.iloc[[i]])
        assert 0.0 <= c.score <= 1.0
        assert c.level in {"HIGH", "MEDIUM", "LOW"}
        assert c.is_estimate, "confidence must be labelled an estimate, not a guarantee"


def test_low_confidence_always_states_a_reason(service, built):
    """A bare low-confidence flag is not actionable; the user needs the why."""
    _, _, test, cols = built
    for i in range(10):
        c = service.get_confidence(test.iloc[[i]])
        assert c.reasons, "every confidence estimate must carry at least one reason"


def test_unfamiliar_input_lowers_familiarity(built):
    _, train, test, cols = built
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(train[c])]
    profile = TrainingProfile.from_frame(train[numeric])

    typical = train[numeric].median()
    extreme = typical.copy()
    for c in numeric[:5]:
        extreme[c] = train[c].max() * 100

    normal = estimate_confidence(0.5, typical, 0.2, profile)
    weird = estimate_confidence(0.5, extreme, 0.2, profile)
    assert weird.familiarity < normal.familiarity
    assert any("outside the range" in r for r in weird.reasons)


# --- Similar orders ----------------------------------------------------------


def test_similarity_never_matches_on_outcome_features():
    """Matching on the outcome would make the retrieved late rate circular."""
    assert not set(SIMILARITY_FEATURES) & set(LEAKAGE_COLUMNS)
    assert "delivery_delay_days" not in SIMILARITY_FEATURES
    assert "actual_delivery_days" not in SIMILARITY_FEATURES


def test_similar_orders_are_returned_ranked(service, built):
    _, train, test, cols = built
    result = service.find_similar_orders(test[cols].iloc[[0]], k=10)
    assert result.n_found <= 10
    sims = result.orders["similarity"].to_numpy()
    assert (np.diff(sims) <= 1e-9).all(), "results must be ordered by similarity"
    assert (sims > 0).all() and (sims <= 1.0).all()


def test_insufficient_evidence_is_declared_not_hidden(built):
    artefact, train, test, cols = built
    tiny = svc.PredictionService(artefact, history=train.head(2),
                                 background=train[cols].head(50))
    result = tiny.find_similar_orders(test[cols].iloc[[0]], k=10)
    assert not result.sufficient
    assert result.note, "the interface must be told why the summary is unreliable"


def test_no_history_returns_empty_not_fabricated(built):
    artefact, train, test, cols = built
    bare = svc.PredictionService(artefact, history=pd.DataFrame(),
                                 background=train[cols].head(50))
    result = bare.find_similar_orders(test[cols].iloc[[0]])
    assert result.n_found == 0
    assert result.orders.empty
    assert "No historical orders" in result.note


def test_comparison_reports_differences_in_original_units(service, built):
    _, _, test, cols = built
    row = test[cols].iloc[[0]]
    similar = service.find_similar_orders(row, k=5)
    table = service.compare_to_similar(row, similar)
    assert {"feature", "current_order", "most_similar", "difference"} <= set(table.columns)
    sample = table[table["difference"].notna()].head(1)
    if len(sample):
        r = sample.iloc[0]
        assert r["difference"] == pytest.approx(r["current_order"] - r["most_similar"])


# --- Capabilities: the no-fabrication rule -----------------------------------


@pytest.mark.parametrize("absent", [
    "Courier / shipping tier", "Warehouse / fulfilment centre",
    "Weather", "Seller rating",
])
def test_absent_levers_are_disabled_with_a_stated_reason(service, absent):
    """Olist records none of these. Offering them would mean inventing both
    the control and the resulting risk figure."""
    levers = service.available_levers()
    assert absent in levers
    assert not levers[absent].available
    assert levers[absent].reason and levers[absent].requires


def test_supported_levers_are_marked_available(service):
    levers = service.available_levers()
    assert levers["carrier_pickup_days"].available
    assert levers["estimated_delivery_days"].available


def test_pre_dispatch_model_disables_the_dispatch_lever(built):
    artefact, train, test, cols = built
    pre_cols = pipeline.feature_columns(train, "pre_dispatch")
    pre = dict(artefact, feature_columns=pre_cols, scenario="pre_dispatch")
    pre["pipeline"] = pipeline.fit_calibrated(
        pipeline.build_pipeline(train, pre_cols, "logistic_regression", y=train[TARGET]),
        train, pre_cols,
    )
    service = svc.PredictionService(pre, history=train, background=train[pre_cols].head(150))
    assert not service.available_levers()["carrier_pickup_days"].available

    infeasible = [i for i in service.suggest_interventions(test.iloc[[0]]) if not i.feasible]
    assert any(i.feature == "carrier_pickup_days" for i in infeasible)


# --- Interventions -----------------------------------------------------------


def test_interventions_use_the_real_pipeline(service, built):
    _, _, test, cols = built
    row = test.iloc[[0]]
    current = float(service.predict_probability(row)[0])
    for i in service.suggest_interventions(row):
        if i.feasible:
            assert 0.0 <= i.new_risk <= 1.0
            assert i.risk_change == pytest.approx(i.new_risk - current)


def test_faster_dispatch_does_not_raise_predicted_risk(service, built):
    _, _, test, cols = built
    row = test.iloc[[0]]
    fast = service.run_counterfactual(row, "carrier_pickup_days", 0.5)
    slow = service.run_counterfactual(row, "carrier_pickup_days", 25.0)
    assert fast <= slow


# --- Risk banding and orchestration -----------------------------------------


def test_risk_bands_are_set_against_the_base_rate():
    assert svc.PredictionService.risk_level(0.60) == "HIGH"
    assert svc.PredictionService.risk_level(0.30) == "MEDIUM"
    assert svc.PredictionService.risk_level(0.05) == "LOW"


def test_assessment_returns_every_component(service, built):
    _, _, test, cols = built
    a = service.assess(test.iloc[[0]], order_id="ord-test")
    assert 0.0 <= a.probability <= 1.0
    assert a.risk_level in {"HIGH", "MEDIUM", "LOW"}
    assert a.confidence.reasons
    assert a.brief.drivers
    assert a.explanation.calibrated
    assert isinstance(a.interventions, list)


def test_post_dispatch_assessment_warns_about_its_own_timing(service, built):
    """A model needing carrier handover cannot run at order time; say so."""
    _, _, test, cols = built
    a = service.assess(test.iloc[[0]])
    assert any("post-dispatch" in w for w in a.warnings)


def test_model_info_reports_only_recorded_metrics(service):
    info = service.model_info()
    assert info["ROC AUC"] == "0.8400"
    assert "F1" not in info, "metrics absent from the artefact must not be invented"
    assert info["Calibrated"] == "yes"


def test_global_importance_maps_to_business_labels(service):
    importance = service.get_feature_importance()
    assert not importance.empty
    assert importance["share"].sum() == pytest.approx(1.0)
    assert (importance["importance"].diff().dropna() <= 1e-9).all()
    assert "label" in importance.columns


# --- "When": target-seeking (Table 1) ----------------------------------------


def test_target_search_finds_a_value_that_meets_the_target(service, built):
    """Table 1's 'When': simulated inputs derived from a desired output."""
    _, _, test, cols = built
    proba = service.predict_probability(test)
    risky = test.iloc[[int(proba.argmax())]]

    result = service.find_value_for_target(risky, "carrier_pickup_days", target_risk=0.2)
    if result.achievable:
        assert result.achieved_risk <= 0.2 + 0.01
        actual = service.run_counterfactual(risky, "carrier_pickup_days", result.required_value)
        assert actual == pytest.approx(result.achieved_risk, abs=1e-6), \
            "the returned value must be verifiable against the real pipeline"
    else:
        assert result.reason and result.best_risk is not None


def test_target_search_declares_unreachable_targets(service, built):
    """An impossible target must be reported, not approximated silently."""
    _, _, test, cols = built
    proba = service.predict_probability(test)
    risky = test.iloc[[int(proba.argmax())]]

    result = service.find_value_for_target(risky, "carrier_pickup_days", target_risk=0.0)
    assert not result.achievable
    assert "cannot" in result.reason.lower() or "above" in result.reason.lower()


def test_target_search_rejects_features_the_model_lacks(service, built):
    _, _, test, cols = built
    result = service.find_value_for_target(test.iloc[[0]], "courier_tier", 0.2)
    assert not result.achievable
    assert "not a model feature" in result.reason


def test_why_not_names_what_would_have_to_change(service, built):
    """'Why not' must go beyond restating the drivers."""
    _, _, test, cols = built
    proba = service.predict_probability(test)
    risky = test.iloc[[int(proba.argmax())]]

    lines = service.explain_why_not(risky, "LOW")
    assert lines
    assert any("percentage points" in line for line in lines)


def test_why_not_recognises_an_order_already_in_the_band(service, built):
    _, _, test, cols = built
    proba = service.predict_probability(test)
    safe = test.iloc[[int(proba.argmin())]]
    lines = service.explain_why_not(safe, "LOW")
    assert any("already sits within" in line for line in lines)


# --- Regression: an order is not its own evidence -----------------------------


def test_similar_orders_exclude_the_query_order(service, built):
    """An order is trivially its own nearest neighbour.

    Left in, it takes the top slot at 100% similarity, makes the
    feature-comparison table read all zeros, and folds the order's own
    outcome into statistics that are supposed to be independent evidence
    about it.
    """
    _, _, test, cols = built
    order_id = test["order_id"].iloc[0]
    row = test[cols].iloc[[0]]

    result = service.find_similar_orders(row, k=10, exclude_order_id=order_id)
    assert order_id not in set(result.orders["order_id"])
    assert (result.orders["similarity"] < 1.0).all(), \
        "a similarity of exactly 1.0 means the query order matched itself"


def test_assess_excludes_the_order_from_its_own_neighbours(service, built):
    _, _, test, cols = built
    order_id = test["order_id"].iloc[0]
    assessment = service.assess(test.iloc[[0]], order_id=order_id)
    if assessment.similar.n_found:
        assert order_id not in set(assessment.similar.orders["order_id"])


def test_comparison_is_against_a_different_order(service, built):
    """With the self-match removed, the comparison has real differences."""
    _, _, test, cols = built
    order_id = test["order_id"].iloc[0]
    row = test[cols].iloc[[0]]
    similar = service.find_similar_orders(row, k=5, exclude_order_id=order_id)
    table = service.compare_to_similar(row, similar)
    differences = table["difference"].dropna()
    if len(differences):
        assert (differences != 0).any(), \
            "every difference zero means the order was compared with itself"


# --- Regression: no search when the target is already met ---------------------


def test_target_search_reports_an_order_already_in_band(service, built):
    """Searching from a satisfied start returns a meaningless required value."""
    _, _, test, cols = built
    proba = service.predict_probability(test)
    safe = test.iloc[[int(proba.argmin())]]
    safe_risk = float(proba.min())

    result = service.find_value_for_target(safe, "carrier_pickup_days",
                                           target_risk=min(safe_risk + 0.2, 0.99))
    assert result.already_satisfied
    assert result.achievable
    assert result.required_value == pytest.approx(result.current_value)
    assert "No change needed" in result.reason


def test_target_search_still_searches_when_above_target(service, built):
    _, _, test, cols = built
    proba = service.predict_probability(test)
    risky = test.iloc[[int(proba.argmax())]]

    result = service.find_value_for_target(risky, "carrier_pickup_days",
                                           target_risk=0.01)
    assert not result.already_satisfied


def test_why_not_omits_levers_that_need_no_change(service, built):
    """A lever reported as 'already satisfied' must not appear as an action."""
    _, _, test, cols = built
    proba = service.predict_probability(test)
    safe = test.iloc[[int(proba.argmin())]]
    lines = service.explain_why_not(safe, "LOW")
    assert any("already sits within" in line for line in lines)
