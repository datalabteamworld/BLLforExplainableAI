"""Request and response schemas.

These are the contract. The frontend builds against them and the service
fulfils them, so changes here are breaking changes and should be versioned
rather than edited in place.

Two conventions run through the file, both carried over from the decision
surface they serve:

**Probabilities are floats in [0, 1], never percentages.** Formatting is a
presentation decision; the API returns the number.

**Unavailable capabilities are returned, not omitted.** A client that receives
only the available levers cannot distinguish "this dataset has no courier
field" from "not implemented yet", and the disclosure that separates this
system from a demo is lost.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["HIGH", "MEDIUM", "LOW"]
ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


# ==========================================================================
# Shared
# ==========================================================================


class CostAssumptions(BaseModel):
    """Business cost model, supplied per request.

    Deliberately not server configuration. Every monetary figure the API
    returns is a function of these numbers, so they travel with the request
    and the response echoes them back. A client that hardcodes them has made
    an assumption it can no longer see.
    """

    model_config = ConfigDict(extra="forbid")

    currency: str = Field("BRL", max_length=8)
    refund_share: float = Field(0.42, ge=0, le=1,
                                description="Refund exposure as a share of order value")
    churn_share: float = Field(0.46, ge=0, le=1,
                               description="Repeat-purchase revenue at risk, as a share")
    support_cost: float = Field(12.0, ge=0, description="Cost of one support contact")
    intervention_cost: float = Field(4.0, ge=0, description="Cost of acting on one order")


class Driver(BaseModel):
    """One contributing factor, in business language, with its provenance."""

    feature: str
    label: str
    sentence: str
    shap_value: float = Field(
        description="Attribution to the model's internal score (log-odds). Ranks "
                    "drivers against one another; not a percentage-point share of "
                    "the returned probability."
    )
    feature_value: str
    raises_risk: bool
    is_lever: bool = Field(description="Whether the business can act on this factor")


class Confidence(BaseModel):
    """How much the estimate should be trusted. Not the probability of lateness."""

    score: float = Field(ge=0, le=1)
    level: ConfidenceLevel
    threshold_margin: float = Field(ge=0, le=1)
    familiarity: float = Field(ge=0, le=1)
    evidence_density: float = Field(ge=0, le=1)
    reasons: list[str]
    is_estimate: bool = Field(
        True, description="Always true. This is not a probability of correctness."
    )


class Option(BaseModel):
    """A costed course of action."""

    name: str
    description: str
    risk: float | None = Field(None, ge=0, le=1)
    action_cost: float | None = None
    expected_loss: float | None = None
    recommended: bool = False
    feasible: bool = True
    note: str = ""


class Capability(BaseModel):
    """Whether a control is supported, and if not, precisely why."""

    name: str
    available: bool
    reason: str = ""
    requires: str = ""


# ==========================================================================
# Scoring
# ==========================================================================


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float | str | None] = Field(
        description="Feature name to value. Missing features are imputed with "
                    "training medians; unknown names are rejected."
    )
    order_id: str | None = None
    order_value: float | None = Field(None, ge=0)


class ScoreResponse(BaseModel):
    """Fast path: probability, band and confidence. No SHAP."""

    order_id: str | None
    probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    threshold: float = Field(
        ge=0, le=1,
        description="Cost-optimal decision threshold, not 0.5. Flag at or above this."
    )
    flagged: bool
    confidence: Confidence
    model_version: str
    scenario: str
    scored_at: datetime
    warnings: list[str] = []


class BatchScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orders: list[ScoreRequest] = Field(max_length=1000)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    count: int


# ==========================================================================
# Explanation
# ==========================================================================


class ExplanationResponse(BaseModel):
    """Slow path: SHAP attributions mapped to business language."""

    order_id: str | None
    probability: float = Field(ge=0, le=1)
    base_value: float
    drivers: list[Driver]
    calibrated: bool
    glossary_coverage: float = Field(
        ge=0, le=1,
        description="Share of returned drivers with a business mapping. A feature "
                    "without one is labelled as unmapped rather than given generic text."
    )
    caveats: list[str]


class SimilarOrder(BaseModel):
    order_id: str
    similarity: float = Field(ge=0, le=1)
    was_late: bool | None
    delay_days: float | None


class SimilarResponse(BaseModel):
    """Evidence independent of the model: what happened to comparable orders."""

    orders: list[SimilarOrder]
    mean_similarity: float
    late_rate: float | None
    mean_delay_days: float | None
    n_found: int
    sufficient: bool = Field(
        description="False when too few comparable orders exist for the summary "
                    "to be reliable. Clients should surface the note rather than "
                    "the statistics."
    )
    note: str = ""
    features_used: list[str]


# ==========================================================================
# Decision brief
# ==========================================================================


class BusinessImpact(BaseModel):
    components: dict[str, float]
    exposure: float
    expected_loss: float
    currency: str


class BriefResponse(BaseModel):
    """The full decision brief: the four BLL functions in one payload."""

    order_id: str | None
    probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    confidence: Confidence
    headline: str
    drivers: list[Driver]
    consequences: list[str]
    impact: BusinessImpact
    options: list[Option]
    recommendation: Option
    net_saving: float
    caveats: list[str]
    cost_assumptions: CostAssumptions = Field(
        description="Echoed back so a client can never display a monetary figure "
                    "without the assumptions that produced it."
    )
    model_version: str


# ==========================================================================
# Simulation
# ==========================================================================


class WhatIfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float | str | None]
    feature: str = Field(description="The lever to change")
    value: float


class WhatIfResponse(BaseModel):
    feature: str
    label: str
    current_value: float | None
    new_value: float
    current_risk: float = Field(ge=0, le=1)
    new_risk: float = Field(ge=0, le=1)
    risk_change: float
    caveat: str = (
        "Model-estimated risk under a ceteris-paribus assumption. Not a causal "
        "guarantee, and changes evaluated separately do not compose."
    )


class TargetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float | str | None]
    feature: str
    target_risk: float = Field(ge=0, le=1)


class TargetResponse(BaseModel):
    """What input value would achieve a desired risk — the 'When' explanation."""

    feature: str
    label: str
    target_risk: float
    achievable: bool
    required_value: float | None = None
    achieved_risk: float | None = None
    current_value: float | None = None
    best_value: float | None = None
    best_risk: float | None = None
    reason: str = ""
    already_satisfied: bool = Field(
        False,
        description="True when the order already meets the target, so no change "
                    "is required. Clients should say so rather than showing a "
                    "required value."
    )


# ==========================================================================
# Model metadata and monitoring
# ==========================================================================


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    model_name: str
    scenario: str
    n_features: int
    feature_columns: list[str]
    threshold: float
    calibrated: bool
    trained_at: str
    metrics: dict[str, float]
    environment: dict[str, str] = Field(
        description="Library versions at training time. Pin these where the model "
                    "is served: scikit-learn can load a pickle from a different "
                    "version with altered behaviour rather than failing."
    )
    capabilities: list[Capability]


class CalibrationBucket(BaseModel):
    band: str
    n: int
    mean_predicted: float
    observed_rate: float
    gap: float


class CalibrationReport(BaseModel):
    """Predicted risk against observed outcomes.

    The signal that matters for this system. A model whose ranking still works
    but whose probabilities have drifted will produce confident monetary
    recommendations that are quietly wrong, because expected loss is
    probability multiplied by exposure.
    """

    n_orders: int
    buckets: list[CalibrationBucket]
    max_gap: float
    mean_absolute_gap: float
    brier_score: float | None = None
    drifted: bool = Field(description="True when max_gap exceeds the configured tolerance")
    tolerance: float
    checked_at: datetime


class EnvironmentInfo(BaseModel):
    """Which data environment answered this request.

    Returned on every metadata response so a client can never display a
    figure without knowing whether it was measured or simulated.
    """

    key: str
    label: str
    kind: str = Field(description="'empirical' or 'synthetic'")
    currency: str
    provenance: str
    disclosure: str | None = Field(
        None,
        description="Present only for synthetic environments; must be shown "
                    "alongside any figure derived from them.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    history_rows: int
    api_version: str
    environment: EnvironmentInfo | None = None
