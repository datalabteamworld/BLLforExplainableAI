"""Tests for the REST API.

Most of these guard the same commitments the package makes elsewhere:
uncertainty travels with the number, unavailable capabilities are disclosed
rather than omitted, and monetary figures never appear without the
assumptions that produced them.

The type-coercion tests exist because JSON has no numeric distinction. A
client sending ``"12.5"`` for a float feature is behaving legitimately, and
the boundary — not the model code — is responsible for sorting that out.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from olist import cleaning, features, pipeline
from olist.config import TARGET
from tests.synthetic import make_tables


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    """A trained artefact and feature table on disk, as a real deployment has."""
    root = tmp_path_factory.mktemp("deployment")
    (root / "models").mkdir()
    (root / "data" / "processed").mkdir(parents=True)

    tables = cleaning.clean_all(make_tables(n_orders=2500, seed=71))[0]
    df = features.build_modelling_table(tables)
    train, _ = pipeline.temporal_split(df, "2018-03-31")
    cols = pipeline.feature_columns(df, "post_dispatch")
    model = pipeline.fit_calibrated(
        pipeline.build_pipeline(train, cols, "logistic_regression", y=train[TARGET]),
        train, cols,
    )
    joblib.dump({
        "pipeline": model, "model_name": "logistic_regression",
        "scenario": "post_dispatch", "feature_columns": cols, "threshold": 0.2,
        "trained_at": "2026-01-01T00:00:00",
        "metrics": {"roc_auc": 0.84, "precision_at_chosen": 0.25},
        "environment": {"scikit-learn": "1.9.0"},
    }, root / "models" / "model_post_dispatch.joblib")

    df.to_parquet(root / "data" / "processed" / "orders_features.parquet", index=False)
    train[cols].head(150).to_parquet(
        root / "data" / "processed" / "background.parquet", index=False)
    return root, df, cols


@pytest.fixture(scope="module")
def client(deployment, monkeypatch_module):
    root, _, _ = deployment
    # Import the module, not the re-exported FastAPI instance in __init__.
    import olist.api.app as api_module

    monkeypatch_module.setattr(api_module, "MODEL_DIRS", [root / "models"])
    monkeypatch_module.setattr(api_module, "DATA_DIRS", [root / "data" / "processed"])
    monkeypatch_module.setenv("OLIST_DB", str(root / "predictions.db"))

    with TestClient(api_module.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def payload(deployment):
    """One order's features, sent as JSON strings — as a real client would."""
    _, df, cols = deployment
    row = df[cols].iloc[0]
    return {k: (None if pd.isna(v) else str(v)) for k, v in row.items()}


@pytest.fixture(scope="module")
def order_id(deployment):
    _, df, _ = deployment
    return df["order_id"].iloc[0]


# --- Meta --------------------------------------------------------------------


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] and body["history_rows"] > 0


def test_model_card_reports_only_recorded_metrics(client):
    body = client.get("/model").json()
    assert body["metrics"]["roc_auc"] == pytest.approx(0.84)
    assert "f1" not in body["metrics"], "metrics absent from the artefact must not be invented"
    assert body["calibrated"] is True


def test_model_card_pins_the_training_environment(client):
    """scikit-learn can load a pickle from another version with altered behaviour."""
    assert client.get("/model").json()["environment"]["scikit-learn"]


def test_threshold_is_cost_optimal_not_half(client):
    assert client.get("/model").json()["threshold"] != 0.5


def test_unavailable_capabilities_are_disclosed(client):
    """A client must distinguish 'the data lacks this' from 'not implemented'."""
    capabilities = client.get("/model").json()["capabilities"]
    unavailable = [c for c in capabilities if not c["available"]]
    assert unavailable
    for capability in unavailable:
        assert capability["reason"], f"{capability['name']} disabled without a reason"


# --- Scoring -----------------------------------------------------------------


def test_score_returns_probability_and_confidence(client, payload, order_id):
    body = client.post("/score", json={"features": payload, "order_id": order_id}).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_level"] in {"HIGH", "MEDIUM", "LOW"}
    assert body["flagged"] == (body["probability"] >= body["threshold"])


def test_confidence_is_never_the_probability(client, payload):
    """Confidence answers how much to trust the estimate, not how likely lateness is."""
    body = client.post("/score", json={"features": payload}).json()
    assert body["confidence"]["score"] != pytest.approx(body["probability"], abs=1e-9)
    assert body["confidence"]["reasons"], "confidence must always carry its reasons"


def test_numeric_features_may_arrive_as_json_strings(client, payload):
    """JSON has no numeric type; "12.5" is a legitimate float from a client."""
    assert client.post("/score", json={"features": payload}).status_code == 200


def test_missing_features_are_imputed_not_rejected(client, payload):
    partial = dict(list(payload.items())[:5])
    assert client.post("/score", json={"features": partial}).status_code == 200


def test_post_dispatch_model_warns_about_its_own_timing(client, payload):
    body = client.post("/score", json={"features": payload}).json()
    assert any("post-dispatch" in w.lower() for w in body["warnings"])


def test_batch_matches_single_scoring(client, payload, order_id):
    single = client.post("/score", json={"features": payload, "order_id": order_id}).json()
    batch = client.post("/score/batch",
                        json={"orders": [{"features": payload, "order_id": order_id}] * 3}).json()
    assert batch["count"] == 3
    for result in batch["results"]:
        assert result["probability"] == pytest.approx(single["probability"], abs=1e-9)


# --- Explanation -------------------------------------------------------------


def test_explanation_returns_business_language(client, order_id):
    body = client.get(f"/orders/{order_id}/explanation").json()
    assert body["drivers"]
    for driver in body["drivers"]:
        assert driver["sentence"] and driver["label"]
        assert isinstance(driver["feature_value"], str)


def test_explanation_declares_the_calibration_caveat(client, order_id):
    """SHAP explains the base classifier; the probability passes through calibration."""
    body = client.get(f"/orders/{order_id}/explanation").json()
    assert body["calibrated"] is True
    assert body["caveats"]


def test_brief_echoes_its_cost_assumptions(client, order_id):
    """A client must never be able to display money without its assumptions."""
    body = client.get(f"/orders/{order_id}/brief?refund_share=0.6&currency=NGN").json()
    assert body["cost_assumptions"]["refund_share"] == pytest.approx(0.6)
    assert body["impact"]["currency"] == "NGN"


def test_brief_expected_loss_is_probability_times_exposure(client, order_id):
    body = client.get(f"/orders/{order_id}/brief").json()
    assert body["impact"]["expected_loss"] == pytest.approx(
        body["probability"] * body["impact"]["exposure"], rel=1e-6)


def test_cost_assumptions_change_the_recommendation_inputs(client, order_id):
    cheap = client.get(f"/orders/{order_id}/brief?refund_share=0.05&churn_share=0.05").json()
    dear = client.get(f"/orders/{order_id}/brief?refund_share=0.9&churn_share=0.9").json()
    assert dear["impact"]["exposure"] > cheap["impact"]["exposure"]


def test_unknown_order_returns_404(client):
    assert client.get("/orders/does-not-exist/brief").status_code == 404


# --- Evidence ----------------------------------------------------------------


def test_similar_orders_declare_sufficiency(client, order_id):
    body = client.get(f"/orders/{order_id}/similar?k=10").json()
    assert "sufficient" in body
    if not body["sufficient"]:
        assert body["note"], "an unreliable summary must say why"


def test_similar_orders_never_match_on_the_outcome(client, order_id):
    """Matching on lateness would make the returned late rate circular."""
    body = client.get(f"/orders/{order_id}/similar").json()
    assert "is_late_delivery" not in body["features_used"]
    assert "delivery_delay_days" not in body["features_used"]


# --- Simulation --------------------------------------------------------------


def test_what_if_runs_the_real_pipeline(client, payload):
    fast = client.post("/what-if", json={"features": payload,
                                         "feature": "carrier_pickup_days", "value": 0.5}).json()
    slow = client.post("/what-if", json={"features": payload,
                                         "feature": "carrier_pickup_days", "value": 30.0}).json()
    assert slow["new_risk"] >= fast["new_risk"], "later dispatch should not lower risk"
    assert fast["risk_change"] == pytest.approx(fast["new_risk"] - fast["current_risk"])


def test_what_if_rejects_an_absent_lever_by_name(client, payload):
    """A silent zero would let a client believe the lever exists."""
    response = client.post("/what-if", json={"features": payload,
                                             "feature": "courier_tier", "value": 1.0})
    assert response.status_code == 422
    assert "courier_tier" in response.json()["detail"]


def test_target_search_returns_a_verifiable_value(client, payload):
    body = client.post("/target", json={"features": payload,
                                        "feature": "carrier_pickup_days",
                                        "target_risk": 0.2}).json()
    if body["achievable"]:
        check = client.post("/what-if", json={"features": payload,
                                              "feature": "carrier_pickup_days",
                                              "value": body["required_value"]}).json()
        assert check["new_risk"] == pytest.approx(body["achieved_risk"], abs=1e-6)
    else:
        assert body["reason"]


def test_why_not_explains_what_would_have_to_change(client, order_id):
    body = client.get(f"/orders/{order_id}/why-not?target_level=LOW").json()
    assert body["explanations"]


# --- Monitoring --------------------------------------------------------------


def test_predictions_are_logged_for_audit(client, payload):
    before = len(client.get("/monitoring/predictions?limit=1000").json())
    client.post("/score", json={"features": payload, "order_id": "audit-test"})
    after = client.get("/monitoring/predictions?limit=1000").json()
    assert len(after) == before + 1


def test_outcome_reporting_feeds_calibration(client, payload):
    client.post("/score", json={"features": payload, "order_id": "outcome-test"})
    recorded = client.post("/outcomes?order_id=outcome-test&was_late=true").json()
    assert recorded["predictions_updated"] >= 1

    calibration = client.get("/monitoring/calibration").json()
    assert calibration["n_orders"] >= 1
    assert "drifted" in calibration


def test_outcome_for_unknown_order_returns_404(client):
    assert client.post("/outcomes?order_id=never-scored&was_late=true").status_code == 404


def test_calibration_reports_its_tolerance(client):
    """Drift is a judgement call, so the threshold that produced it travels too."""
    body = client.get("/monitoring/calibration?tolerance=0.05").json()
    assert body["tolerance"] == pytest.approx(0.05)
