"""Prediction service: the single interface between the UI and the model.

Section 24 of the specification asks for an abstraction so the application can
evolve without the interface being coupled to the ML implementation. Every
screen calls this class; none imports scikit-learn, SHAP or the pipeline
directly.

The service also enforces the specification's most important rule — never
present invented results. Where a capability cannot be supported by the Olist
data, the corresponding method returns an explicitly disabled result carrying
the reason, rather than a plausible-looking number.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import TARGET
from . import glossary
from .brief import BusinessCostModel, DecisionBrief, build_brief
from .confidence import ConfidenceEstimate, TrainingProfile, estimate_confidence
from .explain import LocalExplanation, counterfactual, explain_order
from .similarity import SimilarOrders, compare_to_nearest, find_similar_orders

logger = logging.getLogger(__name__)

RISK_BANDS = [(0.50, "HIGH"), (0.20, "MEDIUM"), (0.0, "LOW")]


@dataclass
class Capability:
    """Whether a feature is available, and if not, precisely why.

    Returned rather than raising, so the interface can render a disabled state
    with an explanation instead of either crashing or silently inventing data.
    """

    available: bool
    reason: str = ""
    requires: str = ""


@dataclass
class TargetSearch:
    """Result of asking what input value would achieve a desired risk.

    The *When* explanation of Table 1 (Speckmann et al., 2025): simulated
    inputs derived from a desired output, rather than an output derived from
    given inputs.
    """

    feature: str
    target_risk: float
    achievable: bool
    required_value: float | None = None
    achieved_risk: float | None = None
    current_value: float | None = None
    best_value: float | None = None
    best_risk: float | None = None
    reason: str = ""
    already_satisfied: bool = False
    """True when the order already meets the target and no change is required."""


@dataclass
class Intervention:
    """A lever the business can pull, and what the model says it would do."""

    name: str
    feature: str
    description: str
    new_value: float
    current_value: float
    new_risk: float
    risk_change: float
    feasible: bool = True
    note: str = ""


@dataclass
class Assessment:
    """Everything the interface needs for one order."""

    order_id: object
    probability: float
    risk_level: str
    confidence: ConfidenceEstimate
    explanation: LocalExplanation
    brief: DecisionBrief
    similar: SimilarOrders
    interventions: list[Intervention] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PredictionService:
    """Loads a trained artefact and answers questions about single orders."""

    def __init__(self, artefact: dict, history: pd.DataFrame | None = None,
                 background: pd.DataFrame | None = None,
                 costs: BusinessCostModel | None = None,
                 environment=None):
        # Scoring one environment's orders with another's model would be
        # silent: the feature names match, so every call succeeds and every
        # number is meaningless. Checked here, at the only place a model and
        # a dataset are brought together.
        if environment is not None:
            from .. import environments as _env

            _env.check_artefact(artefact, environment)
        self.environment = environment
        self.artefact = artefact
        self.model = artefact["pipeline"]
        self.feature_columns = artefact["feature_columns"]
        self.threshold = float(artefact["threshold"])
        self.scenario = artefact.get("scenario", "unknown")
        self.history = history if history is not None else pd.DataFrame()
        self.background = background
        self.costs = costs or BusinessCostModel()

        source = background if background is not None and len(background) else self.history
        self.profile = (
            TrainingProfile.from_frame(source[self._available(source)])
            if source is not None and len(source) else None
        )

    # -- construction -----------------------------------------------------

    @classmethod
    def for_environment(cls, environment, scenario: str = "post_dispatch",
                        **kwargs) -> "PredictionService":
        """Load the model and data belonging to one environment.

        The preferred constructor: it cannot assemble a mismatched pair,
        because both paths come from the same environment object.
        """
        return cls.from_files(
            environment.model_path(scenario),
            environment.processed_path,
            environment.processed_path.parent / "background.parquet",
            environment=environment,
            **kwargs,
        )

    @classmethod
    def from_files(cls, model_path: Path, history_path: Path | None = None,
                   background_path: Path | None = None, **kwargs) -> "PredictionService":
        artefact = joblib.load(model_path)
        history = pd.read_parquet(history_path) if history_path and history_path.exists() else None
        background = pd.read_parquet(background_path) if background_path and background_path.exists() else None
        return cls(artefact, history=history, background=background, **kwargs)

    def _available(self, frame: pd.DataFrame) -> list[str]:
        return [c for c in self.feature_columns if c in frame.columns]

    # -- core prediction --------------------------------------------------

    def predict_probability(self, orders: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of late delivery for each row."""
        return self.model.predict_proba(orders[self.feature_columns])[:, 1]

    def predict(self, orders: pd.DataFrame) -> np.ndarray:
        """Binary decision at the cost-optimal threshold, not at 0.5."""
        return (self.predict_probability(orders) >= self.threshold).astype(int)

    @staticmethod
    def risk_level(probability: float) -> str:
        """Band a probability. Bands are set against an ~8% base rate.

        A 30% chance of lateness is genuinely elevated when the typical order
        sits near 8%; calling it "low" because it falls below a half would
        mislead a reader who assumes a 0-100 scale.
        """
        return next(label for cutoff, label in RISK_BANDS if probability >= cutoff)

    # -- confidence -------------------------------------------------------

    def get_confidence(self, row: pd.DataFrame) -> ConfidenceEstimate:
        """Estimate how much this prediction should be trusted.

        Deliberately not the predicted probability. See ``confidence.py`` for
        the reasoning and the three components used.
        """
        probability = float(self.predict_probability(row)[0])
        if self.profile is None:
            return ConfidenceEstimate(
                score=0.5, level="LOW", threshold_margin=0.0, familiarity=0.0,
                evidence_density=0.0,
                reasons=["No training profile is available, so confidence cannot be assessed."],
            )
        reference = self.history[self._available(self.history)] if len(self.history) else None
        return estimate_confidence(
            probability, row[self.feature_columns].iloc[0],
            self.threshold, self.profile, reference,
        )

    # -- explanation ------------------------------------------------------

    def explain_prediction(self, row: pd.DataFrame) -> LocalExplanation:
        background = self.background[self.feature_columns] if self.background is not None else None
        return explain_order(self.model, row[self.feature_columns], background=background)

    @functools.lru_cache(maxsize=1)
    def get_feature_importance(self) -> pd.DataFrame:
        """Global importance by mean absolute SHAP contribution.

        Computed over a sample of historical orders and cached, since it does
        not depend on the order being viewed.
        """
        from .explain import explain_orders

        if not len(self.history):
            return pd.DataFrame(columns=["feature", "importance", "share"])

        sample = self.history[self.feature_columns].sample(
            min(200, len(self.history)), random_state=0
        )
        background = self.background[self.feature_columns] if self.background is not None else None
        explanations = explain_orders(self.model, sample, background=background)

        totals = pd.concat([e.contributions.abs() for e in explanations], axis=1).mean(axis=1)
        out = totals.sort_values(ascending=False).rename("importance").reset_index()
        out.columns = ["feature", "importance"]
        out["share"] = out["importance"] / out["importance"].sum()
        out["label"] = out["feature"].map(lambda f: glossary.describe(f, True)[0])
        return out

    # -- evidence ---------------------------------------------------------

    def find_similar_orders(self, row: pd.DataFrame, k: int = 10,
                            require_same_route: bool = False,
                            exclude_order_id: object = None) -> SimilarOrders:
        if not len(self.history):
            return SimilarOrders(
                orders=pd.DataFrame(), mean_similarity=0.0, late_rate=float("nan"),
                mean_delay_days=float("nan"), n_found=0, n_requested=k,
                features_used=[], sufficient=False,
                note="No historical orders are loaded.",
            )
        return find_similar_orders(
            row.iloc[0], self.history, k=k, require_same_route=require_same_route,
            exclude_order_id=exclude_order_id,
        )

    def compare_to_similar(self, row: pd.DataFrame, similar: SimilarOrders) -> pd.DataFrame:
        return compare_to_nearest(row.iloc[0], similar)

    # -- what-if and counterfactuals --------------------------------------

    def available_levers(self) -> dict[str, Capability]:
        """Which operational levers this data supports.

        The Olist release records no courier, warehouse, shipping tier or
        weather. Offering those controls would mean fabricating both the input
        and the resulting risk, so they are reported as unavailable with the
        field that would be required.
        """
        levers = {}
        for feature, entry in glossary.GLOSSARY.items():
            if entry.lever:
                levers[feature] = Capability(available=feature in self.feature_columns,
                                             reason="" if feature in self.feature_columns
                                             else f"{feature} is not a feature of the {self.scenario} model.")
        dataset = (self.environment.label if self.environment is not None
                   else "Olist")
        for name, field_needed in [
            ("Courier / shipping tier", "courier or service level"),
            ("Warehouse / fulfilment centre", "warehouse identifier"),
            ("Weather", "weather observations"),
            ("Seller rating", "seller rating"),
        ]:
            levers[name] = Capability(
                available=False,
                reason=f"The {dataset} dataset records no {field_needed}.",
                requires=field_needed,
            )
        return levers

    def find_value_for_target(self, row: pd.DataFrame, feature: str,
                              target_risk: float, bounds: tuple[float, float] | None = None,
                              tolerance: float = 0.002, max_iter: int = 40) -> "TargetSearch":
        """What value of ``feature`` would bring risk down to ``target_risk``?

        This is the *When* explanation of Table 1 — simulated inputs derived
        from a desired output — and it answers Figure 7's closing question,
        "what would need to change for this order to become low risk?".

        Implemented as a bisection over the lever, evaluating the real
        pipeline at each step. Bisection assumes the risk is monotone in the
        lever over the search range; that is checked at the endpoints and
        reported rather than assumed, because a non-monotone response would
        make the returned value arbitrary.
        """
        if feature not in self.feature_columns:
            return TargetSearch(feature=feature, target_risk=target_risk,
                                achievable=False, reason=f"{feature} is not a model feature.")

        current_value = float(row[feature].iloc[0])
        current_risk = float(self.predict_probability(row)[0])

        # If the order already meets the target, there is nothing to search
        # for. Running the bisection anyway returns whatever value the first
        # probe happens to land on, which reads as a required change when in
        # fact none is needed.
        if current_risk <= target_risk + tolerance:
            return TargetSearch(
                feature=feature, target_risk=target_risk, achievable=True,
                already_satisfied=True, current_value=current_value,
                required_value=current_value, achieved_risk=current_risk,
                reason=(f"No change needed: the order is already at "
                        f"{current_risk:.0%}, within the {target_risk:.0%} target."),
            )
        if bounds is None:
            reference = self.history[feature] if feature in self.history else None
            lo = 0.0
            hi = float(reference.quantile(0.99)) if reference is not None else current_value * 3
            bounds = (lo, max(hi, current_value))

        lo, hi = bounds
        risk_lo, risk_hi = (self.run_counterfactual(row, feature, lo),
                            self.run_counterfactual(row, feature, hi))

        # The direction that reduces risk is whichever endpoint is lower.
        best_end, best_risk = (lo, risk_lo) if risk_lo <= risk_hi else (hi, risk_hi)
        if best_risk > target_risk + tolerance:
            return TargetSearch(
                feature=feature, target_risk=target_risk, achievable=False,
                best_value=best_end, best_risk=best_risk,
                reason=(
                    f"Even at the most favourable value the model reaches "
                    f"{best_risk:.0%}, above the {target_risk:.0%} target. This lever "
                    f"alone cannot get the order there."
                ),
            )

        # Bisect between the endpoint that misses the target and the one that meets it.
        near, far = (hi, lo) if risk_lo <= risk_hi else (lo, hi)
        for _ in range(max_iter):
            mid = (near + far) / 2
            risk_mid = self.run_counterfactual(row, feature, mid)
            if abs(risk_mid - target_risk) <= tolerance:
                break
            if risk_mid > target_risk:
                near = mid
            else:
                far = mid
        else:
            mid, risk_mid = far, self.run_counterfactual(row, feature, far)

        return TargetSearch(
            feature=feature, target_risk=target_risk, achievable=True,
            required_value=float(mid), achieved_risk=float(risk_mid),
            current_value=current_value, best_value=best_end, best_risk=best_risk,
        )

    def explain_why_not(self, row: pd.DataFrame, target_level: str = "LOW") -> list[str]:
        """Why the order was *not* classified at ``target_level``.

        The *Why Not* explanation of Table 1. Rather than restating the
        drivers, this names the specific factors that would each have to move,
        and by how much, which is what distinguishes "why not" from "why".
        """
        ceiling = {"LOW": 0.20, "MEDIUM": 0.50}.get(target_level.upper())
        current = float(self.predict_probability(row)[0])
        if ceiling is None or current < ceiling:
            return [f"This order already sits within the {target_level} band."]

        lines = [
            f"Predicted risk is {current:.0%}. To be classed {target_level} it would "
            f"need to fall below {ceiling:.0%} — a drop of "
            f"{(current - ceiling) * 100:.0f} percentage points."
        ]
        for search in (self.find_value_for_target(row, f, ceiling)
                       for f in self.feature_columns
                       if glossary.is_lever(f)):
            entry = glossary.GLOSSARY.get(search.feature)
            label = entry.label if entry else search.feature
            if search.already_satisfied:
                continue
            if search.achievable:
                lines.append(
                    f"**{label}** would have to reach {search.required_value:.1f}"
                    + (f" (currently {search.current_value:.1f})." if search.current_value is not None else ".")
                )
            elif search.best_risk is not None:
                lines.append(f"**{label}** alone cannot achieve it — {search.reason}")
        return lines

    def run_counterfactual(self, row: pd.DataFrame, feature: str, value: float) -> float:
        """Recompute risk with one feature changed, through the real pipeline."""
        return counterfactual(self.model, row[self.feature_columns], feature, value)

    def suggest_interventions(self, row: pd.DataFrame) -> list[Intervention]:
        """Model-estimated risk change for each supported lever.

        Each lever is evaluated **independently**: the figures do not compose,
        because the model is not additive and the levers are correlated in the
        underlying process. This is a model-estimated association under a
        ceteris-paribus assumption, not a causal claim about what would happen
        if the business acted.
        """
        current = float(self.predict_probability(row)[0])
        targets = [
            ("Expedite dispatch", "carrier_pickup_days", 1.0,
             "Prioritise pick-and-pack so the parcel reaches the carrier within 24 hours."),
            ("Extend the promise by 5 days", "estimated_delivery_days", None,
             "Re-quote the customer a later delivery date before dispatch."),
            ("Tighten the dispatch deadline", "shipping_limit_slack_days", None,
             "Bring the seller's contractual dispatch deadline forward."),
        ]

        out = []
        for name, feature, target, description in targets:
            if feature not in self.feature_columns:
                out.append(Intervention(
                    name=name, feature=feature, description=description,
                    new_value=np.nan, current_value=np.nan, new_risk=np.nan,
                    risk_change=np.nan, feasible=False,
                    note=f"Not available in the {self.scenario} model.",
                ))
                continue

            current_value = float(row[feature].iloc[0]) if pd.notna(row[feature].iloc[0]) else np.nan
            value = target if target is not None else (
                current_value + 5 if feature == "estimated_delivery_days" else current_value + 3
            )
            if pd.isna(value):
                continue

            new_risk = self.run_counterfactual(row, feature, value)
            out.append(Intervention(
                name=name, feature=feature, description=description,
                new_value=value, current_value=current_value,
                new_risk=new_risk, risk_change=new_risk - current,
            ))

        return sorted(out, key=lambda i: (np.isnan(i.risk_change), i.risk_change))

    # -- business impact --------------------------------------------------

    def calculate_business_impact(self, row: pd.DataFrame, explanation: LocalExplanation,
                                  order_value: float, order_id=None) -> DecisionBrief:
        precision = self.artefact.get("metrics", {}).get("precision_at_chosen")
        return build_brief(
            self.model, row[self.feature_columns], explanation,
            order_value=order_value, costs=self.costs,
            model_precision=precision, order_id=order_id,
        )

    # -- orchestration ----------------------------------------------------

    def assess(self, row: pd.DataFrame, order_value: float | None = None,
               order_id=None, k_similar: int = 10) -> Assessment:
        """Full assessment for one order."""
        features = row[self.feature_columns]
        probability = float(self.predict_probability(row)[0])
        confidence = self.get_confidence(row)
        explanation = self.explain_prediction(row)

        if order_value is None:
            order_value = float(row["total_price"].iloc[0]) if "total_price" in row else 0.0

        brief = self.calculate_business_impact(row, explanation, order_value, order_id)
        # Exclude the order from its own neighbour list: its outcome is not
        # evidence about itself.
        similar = self.find_similar_orders(features, k=k_similar,
                                           exclude_order_id=order_id)
        interventions = self.suggest_interventions(row)

        warnings = []
        if confidence.is_low:
            warnings.append(
                "Low confidence: review the contributing factors before taking "
                "operational action on this order."
            )
        if not similar.sufficient and similar.note:
            warnings.append(similar.note)
        if self.scenario == "post_dispatch":
            warnings.append(
                "This is a post-dispatch model: it uses carrier handover timing, "
                "so it can only be run once the parcel has left the seller."
            )

        return Assessment(
            order_id=order_id, probability=probability,
            risk_level=self.risk_level(probability), confidence=confidence,
            explanation=explanation, brief=brief, similar=similar,
            interventions=interventions, warnings=warnings,
        )

    # -- metadata ---------------------------------------------------------

    def model_info(self) -> dict:
        """Model card. Only fields actually recorded in the artefact."""
        metrics = self.artefact.get("metrics", {})
        info = {
            "Model": self.artefact.get("model_name", "unknown"),
            "Scenario": self.scenario,
            "Features": len(self.feature_columns),
            "Decision threshold": f"{self.threshold:.4f}",
            "Trained at": str(self.artefact.get("trained_at", "unknown"))[:19],
            "Calibrated": "yes" if getattr(self.model, "calibrated_classifiers_", None) else "no",
        }
        for key, label in [
            ("roc_auc", "ROC AUC"), ("average_precision", "Average precision"),
            ("brier", "Brier score"), ("recall_at_chosen", "Recall at threshold"),
            ("precision_at_chosen", "Precision at threshold"),
        ]:
            if key in metrics:
                info[label] = f"{metrics[key]:.4f}"
        return info
