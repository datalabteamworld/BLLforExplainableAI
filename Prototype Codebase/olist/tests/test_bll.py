"""Tests for the Business Language Layer.

The framework's own stated requirements are what these tests enforce:
traceability from every business statement to its attribution, and honesty
about uncertainty. A fluent, costed recommendation is more persuasive than a
raw attribution score, so a BLL that silently invents or drops evidence is
more dangerous than no BLL at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olist import bll, cleaning, features, pipeline
from olist.bll import glossary
from olist.bll.brief import BusinessCostModel
from olist.config import TARGET
from tests.synthetic import make_tables


@pytest.fixture(scope="module")
def fitted():
    tables = cleaning.clean_all(make_tables(n_orders=3000, seed=13))[0]
    df = features.build_modelling_table(tables)
    train, test = pipeline.temporal_split(df, "2018-03-31")
    cols = pipeline.feature_columns(df, "post_dispatch")
    model = pipeline.fit_calibrated(
        pipeline.build_pipeline(train, cols, "logistic_regression", y=train[TARGET]),
        train, cols,
    )
    return model, cols, train, test


@pytest.fixture(scope="module")
def brief(fitted):
    model, cols, train, test = fitted
    row = test[cols].iloc[[0]]
    exp = bll.explain_order(model, row, background=train[cols].head(150))
    return bll.build_brief(model, row, exp, order_value=250.0, model_precision=0.25)


# --- Explanation faithfulness ------------------------------------------------


def test_shap_contributions_map_to_original_features(fitted):
    model, cols, train, test = fitted
    exp = bll.explain_order(model, test[cols].iloc[[0]], background=train[cols].head(150))
    assert set(exp.contributions.index) == set(cols), (
        "one-hot columns must be summed back onto their source feature"
    )


def test_displayed_probability_comes_from_the_deployed_model(fitted):
    """The brief must show the model's own number, not one reconstructed from SHAP."""
    model, cols, train, test = fitted
    row = test[cols].iloc[[0]]
    exp = bll.explain_order(model, row, background=train[cols].head(150))
    assert exp.calibrated_probability == pytest.approx(model.predict_proba(row)[0, 1])


def test_explanation_records_that_the_model_is_calibrated(fitted):
    model, cols, train, test = fitted
    exp = bll.explain_order(model, test[cols].iloc[[0]], background=train[cols].head(150))
    assert exp.calibrated is True


def test_counterfactual_runs_the_real_pipeline(fitted):
    """What-if figures must be the system's own output, not an approximation."""
    model, cols, train, test = fitted
    row = test[cols].iloc[[0]]
    base = model.predict_proba(row)[0, 1]
    faster = bll.counterfactual(model, row, "carrier_pickup_days", 0.5)
    slower = bll.counterfactual(model, row, "carrier_pickup_days", 30.0)
    assert 0 <= faster <= 1 and 0 <= slower <= 1
    assert slower > faster, "dispatching later should not reduce predicted risk"
    assert base != slower


def test_counterfactual_rejects_unknown_features(fitted):
    model, cols, train, test = fitted
    with pytest.raises(KeyError):
        bll.counterfactual(model, test[cols].iloc[[0]], "courier_tier", 1.0)


# --- Traceability ------------------------------------------------------------


def test_every_statement_traces_to_an_attribution(brief):
    audit = brief.audit()
    assert len(audit) == len(brief.drivers)
    assert set(audit.columns) >= {"statement", "feature", "shap_value", "direction"}
    assert audit["feature"].notna().all()
    assert audit["shap_value"].notna().all()


def test_driver_direction_matches_its_shap_sign(brief):
    for d in brief.drivers:
        assert d.raises_risk == (d.shap_value > 0)


def test_unmapped_features_are_flagged_not_disguised():
    """A feature with no glossary entry must not masquerade as business language."""
    label, sentence = glossary.describe("some_unmapped_feature", raises_risk=True)
    assert "some_unmapped_feature" in sentence
    assert "no business mapping" in sentence


# --- Honesty about uncertainty -----------------------------------------------


def test_brief_discloses_its_cost_assumptions(brief):
    assert any("assumed cost" in c.lower() for c in brief.caveats)


def test_brief_discloses_model_precision(brief):
    assert any("25%" in c for c in brief.caveats)


def test_uncalibrated_model_is_declared_loudly(fitted):
    """Section 6 names calibration as load-bearing; an uncalibrated model must say so."""
    model, cols, train, test = fitted
    raw = pipeline.build_pipeline(train, cols, "logistic_regression", y=train[TARGET])
    raw.fit(train[cols], train[TARGET])
    row = test[cols].iloc[[0]]
    exp = bll.explain_order(raw, row, background=train[cols].head(150))
    assert exp.calibrated is False
    b = bll.build_brief(raw, row, exp, order_value=250.0)
    assert any("UNCALIBRATED" in c for c in b.caveats)


def test_risk_bands_are_set_against_the_base_rate():
    """A 30% chance of lateness is elevated when the typical order sits near 8%."""
    assert bll.risk_level(0.30) == "Elevated"
    assert bll.risk_level(0.60) == "High"
    assert bll.risk_level(0.02) == "Low"


# --- Quantify and recommend --------------------------------------------------


def test_expected_loss_is_exposure_weighted_by_probability():
    costs = BusinessCostModel()
    components, exposure, expected = bll.quantify(0.5, 1000.0, costs)
    assert exposure == pytest.approx(sum(components.values()))
    assert expected == pytest.approx(0.5 * exposure)


def test_recommendation_minimises_total_cost(brief):
    feasible = [o for o in brief.options if o.feasible and not np.isnan(o.total_cost)]
    assert brief.recommendation.total_cost == pytest.approx(min(o.total_cost for o in feasible))
    assert brief.recommendation.recommended


def test_ship_as_is_wins_when_intervention_cannot_pay(fitted):
    """A layer that always recommends acting is a sales pitch, not decision support."""
    model, cols, train, test = fitted
    proba = model.predict_proba(test[cols])[:, 1]
    row = test[cols].iloc[[int(proba.argmin())]]
    exp = bll.explain_order(model, row, background=train[cols].head(150))
    b = bll.build_brief(model, row, exp, order_value=5.0)
    assert b.recommendation.name == "Ship as-is"
    assert b.net_saving == pytest.approx(0.0)


def test_infeasible_levers_are_shown_not_hidden(fitted):
    """A silently dropped option looks like an option that never existed.

    The pre-dispatch model genuinely has no carrier-handover feature, so
    "Expedite dispatch" cannot be evaluated for it. It must still appear in
    the options table, marked unavailable.
    """
    _, _, train, test = fitted
    pre_cols = pipeline.feature_columns(train, "pre_dispatch")
    assert "carrier_pickup_days" not in pre_cols

    pre_model = pipeline.fit_calibrated(
        pipeline.build_pipeline(train, pre_cols, "logistic_regression", y=train[TARGET]),
        train, pre_cols,
    )
    row = test[pre_cols].iloc[[0]]
    options, best, _ = bll.recommend(pre_model, row, 0.5, 250.0, BusinessCostModel())

    unavailable = [o for o in options if not o.feasible]
    assert unavailable, "the dispatch lever should be reported as unavailable"
    assert any(o.name == "Expedite dispatch" for o in unavailable)
    assert best.feasible, "the recommendation must come from an evaluable option"


def test_brief_renders_without_error(brief):
    text = bll.render_text(brief)
    assert "DELIVERY RISK ASSESSMENT" in text
    assert "RECOMMENDATION" in text
    assert "CAVEATS" in text


def test_audit_is_arrow_serialisable(brief):
    """Feature values are heterogeneous; a mixed-type column breaks Arrow.

    Every dataframe viewer serialises through Arrow, so an audit trail that
    cannot be converted is an audit trail the user never sees.
    """
    import pyarrow as pa

    audit = brief.audit()
    pa.Table.from_pandas(audit, preserve_index=False)   # raises if mixed-type

    assert audit["feature_value"].map(type).eq(str).all()
    assert pd.api.types.is_numeric_dtype(audit["shap_value"])


def test_audit_formats_missing_values_readably(brief):
    from olist.bll.brief import DecisionBrief

    assert DecisionBrief._format_value(float("nan")) == "missing"
    assert DecisionBrief._format_value(None) == "missing"
    assert DecisionBrief._format_value(1234.5) == "1,234.50"
    assert DecisionBrief._format_value("sao paulo") == "sao paulo"
