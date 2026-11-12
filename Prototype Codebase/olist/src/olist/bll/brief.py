"""The four BLL functions: interpret, contextualise, quantify, recommend.

Each function consumes the previous one's output and adds a layer of business
meaning, ending in a :class:`DecisionBrief` -- the object rendered as Figure 5
of the paper.

Two commitments run through this module, both taken from Section 6 of the
paper, which identifies them as load-bearing rather than optional:

**Traceability.** Every business statement carries the feature and SHAP value
it came from. ``DecisionBrief.audit()`` returns that mapping, so any claim on
the screen can be traced to its originating attribution.

**Honesty about uncertainty.** A costed recommendation reads as more
authoritative than a raw attribution score, including when the model is
wrong. The brief therefore carries the model's precision at its operating
point and the evidence behind each driver, and the recommendation is expressed
as expected value over a distribution rather than as a prediction of what will
happen to this order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CostConfig
from . import glossary
from .explain import LocalExplanation, counterfactual

logger = logging.getLogger(__name__)


# ==========================================================================
# Cost model
# ==========================================================================


@dataclass(frozen=True)
class BusinessCostModel:
    """Monetary consequences of a late delivery, as a share of order value.

    These are **assumptions, not measurements**. The paper's figures use
    exemplary values; so do these defaults. Any monetary claim published from
    this layer must state them and show a sensitivity analysis, because the
    recommendation is a function of these numbers as much as of the model.

    ``refund_share`` is freight refund plus goodwill credit; ``churn_share``
    is repeat-purchase revenue placed at risk by the review-score drop that
    follows a late delivery; ``support_cost`` is the fixed cost of handling
    the resulting customer contact.
    """

    currency: str = "BRL"
    refund_share: float = 0.42
    churn_share: float = 0.46
    support_cost: float = 12.0
    intervention_cost: float = 4.0

    def components(self, order_value: float) -> dict[str, float]:
        return {
            "Refund exposure": self.refund_share * order_value,
            "Repeat-purchase risk": self.churn_share * order_value,
            "Support cost": self.support_cost,
        }

    def exposure(self, order_value: float) -> float:
        """Total value at stake if this order arrives late."""
        return sum(self.components(order_value).values())

    def format(self, amount: float) -> str:
        return f"{self.currency} {amount:,.0f}"


# ==========================================================================
# Output structures
# ==========================================================================


@dataclass
class Driver:
    """One plainly worded contributing factor, with its provenance."""

    feature: str
    label: str
    sentence: str
    shap_value: float
    feature_value: object
    raises_risk: bool
    is_lever: bool

    @property
    def badge(self) -> str:
        return "Adds risk" if self.raises_risk else "Lowers risk"


@dataclass
class Option:
    """A costed course of action."""

    name: str
    description: str
    risk: float
    action_cost: float
    expected_loss: float
    recommended: bool = False
    feasible: bool = True

    @property
    def total_cost(self) -> float:
        return self.expected_loss + self.action_cost


@dataclass
class DecisionBrief:
    """The complete BLL output for one order."""

    order_id: object
    probability: float
    risk_level: str
    order_value: float
    drivers: list[Driver]
    consequences: list[str]
    impact: dict[str, float]
    exposure: float
    options: list[Option]
    recommendation: Option
    net_saving: float
    costs: BusinessCostModel
    caveats: list[str] = field(default_factory=list)
    glossary_coverage: float = 1.0

    @staticmethod
    def _format_value(value: object) -> str:
        """Render a feature value for display.

        Returned as a string deliberately. Feature values are heterogeneous --
        floats, integers, category names, missing -- and a mixed-type column
        cannot be serialised to Arrow, which is what every dataframe viewer
        uses. Formatting here keeps the audit trail renderable everywhere
        rather than pushing the problem to each caller.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "missing"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        if isinstance(value, (float, np.floating)):
            return f"{value:,.2f}"
        return str(value)

    def audit(self) -> pd.DataFrame:
        """Trace every business statement back to its attribution."""
        return pd.DataFrame(
            [
                {
                    "statement": d.sentence,
                    "feature": d.feature,
                    "feature_value": self._format_value(d.feature_value),
                    "shap_value": round(float(d.shap_value), 4),
                    "direction": "raises" if d.raises_risk else "lowers",
                }
                for d in self.drivers
            ]
        )

    def headline(self) -> str:
        if self.recommendation.name == "Ship as-is":
            return "No intervention needed. Ship as planned."
        return f"Do not ship as-is. {self.recommendation.name}."


# ==========================================================================
# 1. Interpret
# ==========================================================================


RISK_BANDS = [(0.50, "High"), (0.25, "Elevated"), (0.10, "Moderate"), (0.0, "Low")]


def risk_level(probability: float) -> str:
    """Band a probability for a reader who thinks in categories, not decimals.

    Bands are set relative to a ~8% base rate, not on an even 0-100 scale.
    A 30% chance of lateness is genuinely elevated when the typical order sits
    near 8%, and labelling it "low" because it falls below a half would
    mislead.
    """
    return next(label for threshold, label in RISK_BANDS if probability >= threshold)


def interpret(explanation: LocalExplanation, n_drivers: int = 4) -> tuple[str, list[Driver]]:
    """Turn attributions into a risk level and plainly worded drivers."""
    top = explanation.top_drivers(n_drivers)

    drivers = []
    for feature, value in top.items():
        raises = value > 0
        label, sentence = glossary.describe(feature, raises)
        drivers.append(
            Driver(
                feature=feature,
                label=label,
                sentence=sentence,
                shap_value=float(value),
                feature_value=explanation.feature_values.get(feature),
                raises_risk=raises,
                is_lever=glossary.is_lever(feature),
            )
        )
    return risk_level(explanation.calibrated_probability), drivers


# ==========================================================================
# 2. Contextualise
# ==========================================================================


def contextualise(probability: float, drivers: list[Driver], order_value: float,
                  costs: BusinessCostModel) -> list[str]:
    """State the operational consequences implied by the prediction."""
    consequences: list[str] = []

    if probability >= 0.25:
        consequences.append("The delivery promise to this customer is likely to be missed.")
        consequences.append(
            "Expect a review-score drop and an inbound support contact if no action is taken."
        )
    elif probability >= 0.10:
        consequences.append("This order carries above-average risk of arriving after the promised date.")
    else:
        consequences.append("This order is on track against its promised date.")

    if any(d.feature == "is_multi_seller" and d.raises_risk for d in drivers):
        consequences.append("Completion waits on the slowest of several sellers.")
    if any("holiday" in d.feature or "carnival" in d.feature for d in drivers):
        consequences.append("Carrier capacity is reduced for part of the delivery window.")
    if any(d.feature == "seller_late_rate_to_date" and d.raises_risk for d in drivers):
        consequences.append("This seller's recent dispatch record is a recurring contributor, not a one-off.")

    consequences.append(
        f"{costs.format(costs.exposure(order_value))} of customer value is exposed on this order."
    )
    return consequences


# ==========================================================================
# 3. Quantify
# ==========================================================================


def quantify(probability: float, order_value: float,
             costs: BusinessCostModel) -> tuple[dict[str, float], float, float]:
    """Put a monetary value on the risk.

    Returns the component breakdown, the total exposure, and the *expected*
    loss -- exposure weighted by probability. The distinction matters: the
    exposure is what is at stake, the expected loss is what it is worth
    spending to avoid it.
    """
    components = costs.components(order_value)
    exposure = sum(components.values())
    return components, exposure, probability * exposure


# ==========================================================================
# 4. Recommend
# ==========================================================================


#: Interventions available in this dataset, as (feature, new value, description).
#: Olist records no courier tier, so the levers are the ones the data actually
#: supports: dispatch speed and promise length. Inventing a courier-upgrade
#: lever would produce a recommendation the model cannot evaluate.
INTERVENTIONS = [
    (
        "Expedite dispatch",
        "carrier_pickup_days",
        1.0,
        "Prioritise pick-and-pack so the parcel reaches the carrier within 24 hours.",
    ),
    (
        "Extend the promise",
        "estimated_delivery_days",
        None,  # resolved at runtime: current value + 5
        "Re-quote the customer a later delivery date before dispatch.",
    ),
]


def recommend(model, row: pd.DataFrame, probability: float, order_value: float,
              costs: BusinessCostModel) -> tuple[list[Option], Option, float]:
    """Build a ranked table of costed alternatives.

    Each option's risk is recomputed by running the perturbed order through
    the deployed pipeline, so the figures shown are the system's own, not an
    approximation. Options whose lever is not present in the model's feature
    set are marked infeasible and shown rather than hidden -- a silently
    dropped option looks like an option that was never available.
    """
    baseline = Option(
        name="Ship as-is",
        description="Take no action and accept the current risk.",
        risk=probability,
        action_cost=0.0,
        expected_loss=probability * costs.exposure(order_value),
    )
    options = [baseline]

    for name, feature, target, description in INTERVENTIONS:
        if feature not in row.columns:
            options.append(
                Option(name=name, description=description, risk=float("nan"),
                       action_cost=costs.intervention_cost, expected_loss=float("nan"),
                       feasible=False)
            )
            continue

        value = target
        if value is None:  # extend the promise by five days from its current length
            current = row[feature].iloc[0]
            value = float(current) + 5 if pd.notna(current) else None
        if value is None:
            continue

        new_risk = counterfactual(model, row, feature, value)
        options.append(
            Option(
                name=name,
                description=description,
                risk=new_risk,
                action_cost=costs.intervention_cost,
                expected_loss=new_risk * costs.exposure(order_value),
            )
        )

    feasible = [o for o in options if o.feasible and not np.isnan(o.total_cost)]
    best = min(feasible, key=lambda o: o.total_cost)
    best.recommended = True
    return options, best, baseline.total_cost - best.total_cost


# ==========================================================================
# Assembly
# ==========================================================================


def build_brief(model, row: pd.DataFrame, explanation: LocalExplanation,
                order_value: float, costs: BusinessCostModel | None = None,
                model_precision: float | None = None,
                order_id: object = None) -> DecisionBrief:
    """Run all four functions and assemble the decision brief."""
    costs = costs or BusinessCostModel()
    probability = explanation.calibrated_probability

    level, drivers = interpret(explanation)
    consequences = contextualise(probability, drivers, order_value, costs)
    components, exposure, _ = quantify(probability, order_value, costs)
    options, best, saving = recommend(model, row, probability, order_value, costs)

    caveats = [
        "Monetary figures rest on assumed cost shares, not measured Olist finances.",
        "Contributions explain the model's risk ranking; the displayed probability "
        "additionally passes through calibration.",
    ]
    if model_precision is not None:
        caveats.append(
            f"At this operating point the model is right about {model_precision:.0%} "
            f"of the orders it flags. Treat this as a prioritisation signal, not a certainty."
        )
    if not explanation.calibrated:
        caveats.append(
            "This model is UNCALIBRATED: the probability shown is not a reliable "
            "frequency and the monetary figures inherit that error."
        )

    return DecisionBrief(
        order_id=order_id if order_id is not None else explanation.order_index,
        probability=probability,
        risk_level=level,
        order_value=order_value,
        drivers=drivers,
        consequences=consequences,
        impact=components,
        exposure=exposure,
        options=options,
        recommendation=best,
        net_saving=saving,
        costs=costs,
        caveats=caveats,
        glossary_coverage=glossary.coverage([d.feature for d in drivers]),
    )


def render_text(brief: DecisionBrief) -> str:
    """Plain-text decision brief -- the console form of Figure 5."""
    c = brief.costs
    lines = [
        f"DELIVERY RISK ASSESSMENT — order {brief.order_id}",
        "=" * 64,
        f"Late-delivery probability : {brief.probability:.1%}  ({brief.risk_level} risk)",
        f"Order value               : {c.format(brief.order_value)}",
        "",
        "KEY CONTRIBUTING FACTORS",
    ]
    for d in brief.drivers:
        lines.append(f"  [{d.badge:11s}] {d.label}")
        lines.append(f"                  {d.sentence}")

    lines += ["", "WHAT THIS MEANS"]
    lines += [f"  - {s}" for s in brief.consequences]

    lines += ["", "BUSINESS IMPACT IF LATE"]
    for name, amount in brief.impact.items():
        lines.append(f"  {name:24s} {c.format(amount):>16s}")
    lines.append(f"  {'Total exposure':24s} {c.format(brief.exposure):>16s}")

    lines += ["", "OPTIONS"]
    lines.append(f"  {'Option':22s} {'Risk':>8s} {'Action':>12s} {'Expected loss':>16s}")
    for o in brief.options:
        if not o.feasible:
            lines.append(f"  {o.name:22s} {'n/a':>8s}  (lever not available in this model)")
            continue
        mark = " <- recommended" if o.recommended else ""
        lines.append(
            f"  {o.name:22s} {o.risk:>7.1%} {c.format(o.action_cost):>12s} "
            f"{c.format(o.expected_loss):>16s}{mark}"
        )

    lines += ["", f"RECOMMENDATION: {brief.headline()}"]
    lines.append(f"  Net saving vs shipping as-is: {c.format(brief.net_saving)}")
    lines += ["", "CAVEATS"]
    lines += [f"  ! {s}" for s in brief.caveats]
    return "\n".join(lines)
