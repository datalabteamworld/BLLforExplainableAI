"""FastAPI service over the Business Language Layer.

A thin wrapper. All modelling, explanation and business logic lives in
``olist.bll.PredictionService`` and is covered by the package's test suite;
this module translates HTTP to that class and back. Nothing here reimplements
a feature, a threshold or a cost rule, because the leakage rules those
encode are subtle and break silently when reimplemented under time pressure.

Three conventions worth knowing before extending it.

**Fast and slow paths are separated.** ``/score`` is milliseconds; ``/explain``
and ``/brief`` run SHAP and take hundreds of milliseconds. Bundling them would
make every caller pay for an explanation most do not want.

**Capabilities are disclosed, not omitted.** ``/model`` returns the levers this
dataset cannot support along with the field each would require. A client that
only sees available options cannot distinguish "the data has no courier field"
from "not implemented", and that distinction is what separates an honest
system from a demo.

**Monetary figures never travel without their assumptions.** Every response
carrying money echoes the cost parameters that produced it.

Run with::

    uvicorn olist.api.app:app --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ..bll import PredictionService, glossary
from ..bll.brief import BusinessCostModel
from . import schemas as S
from .store import PredictionStore, calibration_report

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIRS = [ROOT / "models", ROOT / "deploy", Path.cwd() / "models", Path.cwd()]
DATA_DIRS = [ROOT / "data" / "processed", ROOT / "deploy",
             Path.cwd() / "data" / "processed", Path.cwd()]

_state: dict = {"service": None, "store": None, "error": "", "importance": None}


def _find(patterns: list[str], dirs: list[Path]) -> Path | None:
    for directory in dirs:
        if directory.exists():
            for pattern in patterns:
                hits = sorted(directory.glob(pattern))
                if hits:
                    return hits[0]
    return None


def load_service() -> PredictionService | None:
    """Locate and load the trained artefact. Returns None and records why."""
    model_path = _find(["model_post_dispatch.joblib", "model_*.joblib"], MODEL_DIRS)
    if model_path is None:
        _state["error"] = ("No model artefact found. Train one with "
                           "`python -m olist.train --scenario post_dispatch`.")
        return None
    try:
        service = PredictionService.from_files(
            model_path,
            _find(["orders_features.parquet", "sample_orders.parquet"], DATA_DIRS),
            _find(["background.parquet"], DATA_DIRS),
        )
        logger.info("Loaded %s: %d features, threshold %.4f",
                    model_path.name, len(service.feature_columns), service.threshold)
        return service
    except Exception as exc:                       # noqa: BLE001 — surfaced via /health
        _state["error"] = f"Failed to load model: {exc}"
        logger.exception("Model load failed")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["service"] = load_service()
    _state["store"] = PredictionStore(os.getenv("OLIST_DB", "data/predictions.db"))
    yield


app = FastAPI(
    title="Olist Delivery Risk API",
    description=(
        "Late-delivery risk prediction with Business Language Layer explanations.\n\n"
        "Probabilities are calibrated. The decision threshold is cost-optimal, not "
        "0.5. Monetary figures rest on the cost assumptions supplied per request — "
        "estimates, not measured finances — and every money-bearing response echoes "
        "the assumptions that produced it."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("OLIST_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------


def get_service() -> PredictionService:
    service = _state["service"]
    if service is None:
        raise HTTPException(503, _state["error"] or "Model not loaded")
    return service


def get_store() -> PredictionStore:
    return _state["store"]


def model_version(service: PredictionService) -> str:
    """Stable identifier for the loaded artefact, used to scope stored records."""
    trained = str(service.artefact.get("trained_at", "unknown"))[:10]
    return f"{service.artefact.get('model_name', 'model')}-{service.scenario}-{trained}"


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


def _frame(features: dict, service: PredictionService) -> pd.DataFrame:
    """Build a one-row frame in the model's column order, with training dtypes.

    Unknown keys are ignored and absent features left as NaN for the
    pipeline's imputer, which was fitted on training data. Rejecting a
    superset of fields would make the API brittle for no safety gain.

    Dtypes are coerced to match the training frame. JSON has no numeric type
    distinction, so a caller may legitimately send ``"12.5"`` for a float
    column; without coercion that string reaches the confidence check and
    fails on a float comparison. Coercing here keeps the boundary responsible
    for type discipline rather than pushing it into the model code.
    """
    row = {c: features.get(c, np.nan) for c in service.feature_columns}
    frame = pd.DataFrame([row])

    reference = service.history
    if reference is not None and len(reference):
        for col in frame.columns:
            if col not in reference.columns:
                continue
            if pd.api.types.is_numeric_dtype(reference[col]):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            else:
                frame[col] = frame[col].astype("object")
    else:
        for col in frame.columns:
            converted = pd.to_numeric(frame[col], errors="coerce")
            if converted.notna().all():
                frame[col] = converted
    return frame


def _confidence(estimate) -> S.Confidence:
    return S.Confidence(
        score=estimate.score, level=estimate.level,
        threshold_margin=estimate.threshold_margin,
        familiarity=estimate.familiarity,
        evidence_density=estimate.evidence_density,
        reasons=estimate.reasons,
    )


def _drivers(drivers) -> list[S.Driver]:
    """Convert drivers for transport.

    ``feature_value`` is rendered as a string by ``DecisionBrief._format_value``
    — the same formatter the audit trail uses. Feature values are
    heterogeneous (floats, counts, category names, missing), and a mixed-type
    field is awkward for both JSON schema and Arrow. Formatting once, in one
    place, keeps the API and the dashboard consistent.
    """
    from ..bll.brief import DecisionBrief

    return [S.Driver(feature=d.feature, label=d.label, sentence=d.sentence,
                     shap_value=d.shap_value,
                     feature_value=DecisionBrief._format_value(d.feature_value),
                     raises_risk=d.raises_risk, is_lever=d.is_lever)
            for d in drivers]


def _clean(value):
    """NaN is not valid JSON; absent is more honest than zero."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)


def _options(options) -> list[S.Option]:
    """Convert options for transport.

    An infeasible option is returned with an explanatory note rather than
    dropped. A client that only ever receives evaluable options cannot tell
    the difference between "this lever does not exist in the data" and "not
    implemented".
    """
    return [S.Option(
        name=o.name, description=o.description, risk=_clean(o.risk),
        action_cost=o.action_cost, expected_loss=_clean(o.expected_loss),
        recommended=o.recommended, feasible=o.feasible,
        note="" if o.feasible else "Not evaluable: this lever is absent from the model.",
    ) for o in options]


def _cost_model(a: S.CostAssumptions) -> BusinessCostModel:
    return BusinessCostModel(currency=a.currency, refund_share=a.refund_share,
                             churn_share=a.churn_share, support_cost=a.support_cost,
                             intervention_cost=a.intervention_cost)


def _warnings(service: PredictionService, confidence) -> list[str]:
    out = []
    if service.scenario == "post_dispatch":
        out.append("Post-dispatch model: it uses carrier handover timing, so it can "
                   "only run once the parcel has left the seller.")
    if confidence.level == "LOW":
        out.append("Low confidence: review the contributing factors before acting.")
    return out


def _order_row(service: PredictionService, order_id: str) -> pd.DataFrame:
    history = service.history
    if not len(history) or "order_id" not in history.columns:
        raise HTTPException(404, "No historical orders are loaded in this deployment.")
    match = history[history["order_id"] == order_id]
    if match.empty:
        raise HTTPException(404, f"Order {order_id!r} not found.")
    return match.iloc[[0]]


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------


@app.get("/health", response_model=S.HealthResponse, tags=["meta"])
def health() -> S.HealthResponse:
    service = _state["service"]
    if service is None:
        return S.HealthResponse(status="degraded", model_loaded=False,
                                model_version=None, history_rows=0,
                                api_version=API_VERSION)
    return S.HealthResponse(status="ok", model_loaded=True,
                            model_version=model_version(service),
                            history_rows=len(service.history),
                            api_version=API_VERSION)


@app.get("/model", response_model=S.ModelInfo, tags=["meta"])
def model_info(service: PredictionService = Depends(get_service)) -> S.ModelInfo:
    """Model card and capability disclosure.

    ``capabilities`` deliberately includes unavailable levers with the field
    each would require, so a client can render a disabled control with a
    reason rather than silently omitting it.
    """
    artefact = service.artefact
    return S.ModelInfo(
        model_version=model_version(service),
        model_name=artefact.get("model_name", "unknown"),
        scenario=service.scenario,
        n_features=len(service.feature_columns),
        feature_columns=service.feature_columns,
        threshold=service.threshold,
        calibrated=bool(getattr(service.model, "calibrated_classifiers_", None)),
        trained_at=str(artefact.get("trained_at", "unknown")),
        metrics={k: float(v) for k, v in artefact.get("metrics", {}).items()},
        environment={k: str(v) for k, v in artefact.get("environment", {}).items()},
        capabilities=[S.Capability(name=n, available=c.available,
                                   reason=c.reason, requires=c.requires)
                      for n, c in service.available_levers().items()],
    )


@app.get("/model/importance", tags=["meta"])
def feature_importance(limit: int = Query(15, ge=1, le=100),
                       service: PredictionService = Depends(get_service)) -> list[dict]:
    """Global feature importance. Cached — it does not depend on the order."""
    if _state["importance"] is None:
        _state["importance"] = service.get_feature_importance()
    frame = _state["importance"]
    if frame.empty:
        return []
    return [{"feature": r["feature"], "label": r["label"],
             "importance": float(r["importance"]), "share": float(r["share"])}
            for _, r in frame.head(limit).iterrows()]


# --------------------------------------------------------------------------
# Scoring — the fast path
# --------------------------------------------------------------------------


@app.post("/score", response_model=S.ScoreResponse, tags=["scoring"])
def score(request: S.ScoreRequest,
          service: PredictionService = Depends(get_service),
          store: PredictionStore = Depends(get_store)) -> S.ScoreResponse:
    """Probability, risk band and confidence. No SHAP, so milliseconds.

    Confidence is not the probability: it reports how much the estimate
    should be trusted, and falls when an order sits near the decision
    threshold or its inputs lie outside the training range.
    """
    frame = _frame(request.features, service)
    probability = float(service.predict_probability(frame)[0])
    confidence = service.get_confidence(frame)
    level = service.risk_level(probability)
    version = model_version(service)

    store.log(order_id=request.order_id, probability=probability, risk_level=level,
              threshold=service.threshold, confidence=confidence.score,
              confidence_level=confidence.level, model_version=version,
              scenario=service.scenario)

    return S.ScoreResponse(
        order_id=request.order_id, probability=probability, risk_level=level,
        threshold=service.threshold, flagged=probability >= service.threshold,
        confidence=_confidence(confidence), model_version=version,
        scenario=service.scenario, scored_at=datetime.now(timezone.utc),
        warnings=_warnings(service, confidence),
    )


@app.post("/score/batch", response_model=S.BatchScoreResponse, tags=["scoring"])
def score_batch(request: S.BatchScoreRequest,
                service: PredictionService = Depends(get_service),
                store: PredictionStore = Depends(get_store)) -> S.BatchScoreResponse:
    """Score many orders in one call.

    Probabilities are computed in a single vectorised pass. Confidence stays
    per-row because it depends on each order's own distance from the
    threshold and its familiarity against the training distribution.
    """
    if not request.orders:
        return S.BatchScoreResponse(results=[], count=0)

    frame = pd.concat([_frame(order.features, service) for order in request.orders],
                      ignore_index=True)
    probabilities = service.predict_probability(frame)
    version = model_version(service)
    now = datetime.now(timezone.utc)

    results = []
    for i, order in enumerate(request.orders):
        probability = float(probabilities[i])
        confidence = service.get_confidence(frame.iloc[[i]])
        level = service.risk_level(probability)
        store.log(order_id=order.order_id, probability=probability, risk_level=level,
                  threshold=service.threshold, confidence=confidence.score,
                  confidence_level=confidence.level, model_version=version,
                  scenario=service.scenario)
        results.append(S.ScoreResponse(
            order_id=order.order_id, probability=probability, risk_level=level,
            threshold=service.threshold, flagged=probability >= service.threshold,
            confidence=_confidence(confidence), model_version=version,
            scenario=service.scenario, scored_at=now,
            warnings=_warnings(service, confidence),
        ))
    return S.BatchScoreResponse(results=results, count=len(results))


# --------------------------------------------------------------------------
# Explanation — the slow path
# --------------------------------------------------------------------------


@app.get("/orders/{order_id}/explanation", response_model=S.ExplanationResponse,
         tags=["explanation"])
def explanation(order_id: str,
                service: PredictionService = Depends(get_service)) -> S.ExplanationResponse:
    """SHAP attributions mapped to business language.

    Slow relative to ``/score``. Call it when a user opens an order, not for
    every scored row.
    """
    row = _order_row(service, order_id)
    assessment = service.assess(row, order_id=order_id)
    return S.ExplanationResponse(
        order_id=order_id,
        probability=assessment.probability,
        base_value=assessment.explanation.base_value,
        drivers=_drivers(assessment.brief.drivers),
        calibrated=assessment.explanation.calibrated,
        glossary_coverage=assessment.brief.glossary_coverage,
        caveats=assessment.brief.caveats,
    )


@app.get("/orders/{order_id}/brief", response_model=S.BriefResponse, tags=["explanation"])
def brief(order_id: str,
          currency: str = Query("BRL"),
          refund_share: float = Query(0.42, ge=0, le=1),
          churn_share: float = Query(0.46, ge=0, le=1),
          support_cost: float = Query(12.0, ge=0),
          intervention_cost: float = Query(4.0, ge=0),
          service: PredictionService = Depends(get_service),
          store: PredictionStore = Depends(get_store)) -> S.BriefResponse:
    """The full decision brief: interpret, contextualise, quantify, recommend.

    Cost assumptions are request parameters rather than server configuration
    because every monetary figure here is a function of them, and they are
    echoed back so a client cannot display money without them.
    """
    row = _order_row(service, order_id)
    assumptions = S.CostAssumptions(
        currency=currency, refund_share=refund_share, churn_share=churn_share,
        support_cost=support_cost, intervention_cost=intervention_cost)

    service.costs = _cost_model(assumptions)
    assessment = service.assess(row, order_id=order_id)
    b = assessment.brief

    store.log(order_id=order_id, probability=assessment.probability,
              risk_level=assessment.risk_level, threshold=service.threshold,
              confidence=assessment.confidence.score,
              confidence_level=assessment.confidence.level,
              model_version=model_version(service), scenario=service.scenario,
              top_drivers=[d.label for d in b.drivers],
              recommendation=b.recommendation.name)

    recommended = next((o for o in _options(b.options) if o.recommended),
                       _options(b.options)[0])

    return S.BriefResponse(
        order_id=str(order_id),
        probability=assessment.probability,
        risk_level=assessment.risk_level,
        confidence=_confidence(assessment.confidence),
        headline=b.headline(),
        drivers=_drivers(b.drivers),
        consequences=b.consequences,
        impact=S.BusinessImpact(
            components={k: float(v) for k, v in b.impact.items()},
            exposure=b.exposure,
            expected_loss=assessment.probability * b.exposure,
            currency=assumptions.currency,
        ),
        options=_options(b.options),
        recommendation=recommended,
        net_saving=b.net_saving,
        caveats=b.caveats + assessment.warnings,
        cost_assumptions=assumptions,
        model_version=model_version(service),
    )


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


@app.get("/orders/{order_id}/similar", response_model=S.SimilarResponse, tags=["evidence"])
def similar(order_id: str,
            k: int = Query(10, ge=1, le=50),
            same_route: bool = Query(False),
            service: PredictionService = Depends(get_service)) -> S.SimilarResponse:
    """Comparable historical orders and what actually happened to them.

    Evidence independent of the model. Similarity uses only features knowable
    before delivery: matching on the outcome would make the returned late
    rate circular.
    """
    row = _order_row(service, order_id)
    result = service.find_similar_orders(row[service.feature_columns], k=k,
                                         require_same_route=same_route)

    orders = []
    for _, r in result.orders.iterrows():
        late, delay = r.get("is_late_delivery"), r.get("delivery_delay_days")
        orders.append(S.SimilarOrder(
            order_id=str(r.get("order_id", "")),
            similarity=float(r["similarity"]),
            was_late=None if pd.isna(late) else bool(late),
            delay_days=None if pd.isna(delay) else float(delay),
        ))

    return S.SimilarResponse(
        orders=orders, mean_similarity=result.mean_similarity,
        late_rate=_clean(result.late_rate),
        mean_delay_days=_clean(result.mean_delay_days),
        n_found=result.n_found, sufficient=result.sufficient,
        note=result.note, features_used=result.features_used,
    )


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


@app.post("/what-if", response_model=S.WhatIfResponse, tags=["simulation"])
def what_if(request: S.WhatIfRequest,
            service: PredictionService = Depends(get_service)) -> S.WhatIfResponse:
    """Recompute risk with one input changed, through the deployed pipeline.

    A 422 naming the unavailable lever is deliberate: the client should be
    told the field does not exist in this dataset, not handed a silent zero.
    """
    if request.feature not in service.feature_columns:
        raise HTTPException(
            422,
            f"{request.feature!r} is not a feature of the {service.scenario} model. "
            f"GET /model lists available levers and why the others are unavailable.",
        )

    frame = _frame(request.features, service)
    current_risk = float(service.predict_probability(frame)[0])
    new_risk = service.run_counterfactual(frame, request.feature, request.value)
    current_value = frame[request.feature].iloc[0]
    entry = glossary.GLOSSARY.get(request.feature)

    return S.WhatIfResponse(
        feature=request.feature,
        label=entry.label if entry else request.feature,
        current_value=None if pd.isna(current_value) else float(current_value),
        new_value=request.value,
        current_risk=current_risk,
        new_risk=new_risk,
        risk_change=new_risk - current_risk,
    )


@app.post("/target", response_model=S.TargetResponse, tags=["simulation"])
def target(request: S.TargetRequest,
           service: PredictionService = Depends(get_service)) -> S.TargetResponse:
    """What value of a lever would reach a desired risk.

    The reverse of a prediction: an input derived from a wanted output.
    Returns ``achievable=false`` with the best reachable risk when no value
    of that lever alone gets there, rather than approximating an answer.
    """
    frame = _frame(request.features, service)
    result = service.find_value_for_target(frame, request.feature, request.target_risk)
    entry = glossary.GLOSSARY.get(request.feature)

    return S.TargetResponse(
        feature=result.feature,
        label=entry.label if entry else result.feature,
        target_risk=result.target_risk,
        achievable=result.achievable,
        required_value=result.required_value,
        achieved_risk=result.achieved_risk,
        current_value=result.current_value,
        best_value=result.best_value,
        best_risk=result.best_risk,
        reason=result.reason,
        already_satisfied=result.already_satisfied,
    )


@app.get("/orders/{order_id}/why-not", tags=["simulation"])
def why_not(order_id: str,
            target_level: str = Query("LOW", pattern="^(LOW|MEDIUM)$"),
            service: PredictionService = Depends(get_service)) -> dict:
    """Why the order was not classified at a lower band, and what would change it."""
    row = _order_row(service, order_id)
    return {
        "order_id": order_id,
        "target_level": target_level,
        "current_probability": float(service.predict_probability(row)[0]),
        "explanations": service.explain_why_not(row, target_level),
    }


# --------------------------------------------------------------------------
# Monitoring
# --------------------------------------------------------------------------


@app.post("/outcomes", tags=["monitoring"])
def record_outcome(order_id: str = Query(...), was_late: bool = Query(...),
                   store: PredictionStore = Depends(get_store)) -> dict:
    """Report what actually happened, so calibration can be measured.

    Without this the service can never know whether its probabilities are
    still honest. Wire it to whatever marks an order delivered.
    """
    updated = store.record_outcome(order_id, was_late)
    if not updated:
        raise HTTPException(404, f"No stored prediction for order {order_id!r}.")
    return {"order_id": order_id, "predictions_updated": updated}


@app.get("/monitoring/calibration", response_model=S.CalibrationReport, tags=["monitoring"])
def calibration(tolerance: float = Query(0.10, ge=0.01, le=0.5),
                service: PredictionService = Depends(get_service),
                store: PredictionStore = Depends(get_store)) -> S.CalibrationReport:
    """Predicted risk against observed outcomes.

    The metric that matters in production. Ranking can stay intact while
    calibration drifts, and when it does every monetary figure derived from
    the probability becomes quietly wrong while accuracy still looks fine.
    """
    resolved = store.resolved(model_version=model_version(service))
    if not resolved:
        return S.CalibrationReport(
            n_orders=0, buckets=[], max_gap=0.0, mean_absolute_gap=0.0,
            brier_score=None, drifted=False, tolerance=tolerance,
            checked_at=datetime.now(timezone.utc),
        )

    report = calibration_report(
        np.array([r["probability"] for r in resolved]),
        np.array([r["observed_late"] for r in resolved], dtype=float),
        tolerance=tolerance,
    )
    return S.CalibrationReport(
        n_orders=report["n_orders"],
        buckets=[S.CalibrationBucket(**b) for b in report["buckets"]],
        max_gap=report["max_gap"],
        mean_absolute_gap=report["mean_absolute_gap"],
        brier_score=report["brier_score"],
        drifted=report["drifted"],
        tolerance=report["tolerance"],
        checked_at=datetime.now(timezone.utc),
    )


@app.get("/monitoring/predictions", tags=["monitoring"])
def recent_predictions(limit: int = Query(100, ge=1, le=1000),
                       store: PredictionStore = Depends(get_store)) -> list[dict]:
    """Recent predictions, for audit and debugging."""
    return store.recent(limit=limit)
