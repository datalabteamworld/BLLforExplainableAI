"""Business Language Layer: from model explanation to decision brief.

Implements the four functions of the BLL framework (Oyeleke et al., 2026) --
interpret, contextualise, quantify, recommend -- on top of an existing model
and its SHAP explainer. The layer reads explanations; it does not retrain or
replace the model.
"""

from .brief import (
    BusinessCostModel,
    DecisionBrief,
    Driver,
    Option,
    build_brief,
    contextualise,
    interpret,
    quantify,
    recommend,
    render_text,
    risk_level,
)
from . import glossary
from .confidence import ConfidenceEstimate, TrainingProfile, estimate_confidence
from .report import build_report
from .service import (
    Assessment,
    Capability,
    Intervention,
    PredictionService,
    TargetSearch,
)
from .similarity import SimilarOrders, compare_to_nearest, find_similar_orders
from .explain import LocalExplanation, counterfactual, explain_order, explain_orders

__all__ = [
    "BusinessCostModel", "DecisionBrief", "Driver", "Option", "LocalExplanation",
    "build_brief", "interpret", "contextualise", "quantify", "recommend",
    "glossary", "render_text", "risk_level",
    "build_report", "PredictionService", "Assessment", "Capability", "Intervention", "TargetSearch",
    "ConfidenceEstimate", "TrainingProfile", "estimate_confidence",
    "SimilarOrders", "find_similar_orders", "compare_to_nearest", "explain_order", "explain_orders", "counterfactual",
]
