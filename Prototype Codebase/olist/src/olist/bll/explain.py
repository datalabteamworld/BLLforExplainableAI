"""Local SHAP explanations for the deployed artefact.

The deployed model is a ``CalibratedClassifierCV`` wrapping an
``imblearn.Pipeline`` that itself contains imputation, scaling, target
encoding and a classifier. Handing that object to ``shap.Explainer`` does not
work, and forcing it produces attributions for the wrong thing.

**The faithfulness problem.** SHAP explains the base classifier's decision
function. The number shown to the user is the *calibrated* probability. These
are different quantities related by a monotone map. The paper is explicit that
"faithful preservation of the original evidence" is load-bearing rather than
optional, so this module does not paper over the distinction: it explains the
base classifier, reports the calibrated probability separately, and records
both. Attributions are therefore statements about *relative contribution to
the risk ranking*, not additive decompositions of the displayed percentage.

Claiming otherwise would be the exact failure the paper warns about -- a
fluent output that is more persuasive than its evidence supports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LocalExplanation:
    """SHAP attributions for a single order, in original feature space."""

    order_index: object
    calibrated_probability: float
    base_value: float
    contributions: pd.Series          #: signed, indexed by original feature name
    feature_values: pd.Series
    calibrated: bool

    def top_drivers(self, n: int = 4, direction: str = "both") -> pd.Series:
        """Largest contributions by absolute magnitude."""
        series = self.contributions
        if direction == "positive":
            series = series[series > 0]
        elif direction == "negative":
            series = series[series < 0]
        return series.reindex(series.abs().sort_values(ascending=False).index).head(n)


def unwrap(model):
    """Return ``(inner_pipeline, is_calibrated)`` from a deployed artefact.

    Handles ``CalibratedClassifierCV`` over a ``FrozenEstimator`` over a
    pipeline, which is what ``pipeline.fit_calibrated`` produces on
    scikit-learn >= 1.6.
    """
    from sklearn.calibration import CalibratedClassifierCV

    if not isinstance(model, CalibratedClassifierCV):
        return model, False

    inner = model.calibrated_classifiers_[0].estimator
    inner = getattr(inner, "estimator", inner)  # unwrap FrozenEstimator
    return inner, True


def _aggregate_to_original(values: np.ndarray, encoded_names: list[str],
                           original: list[str]) -> pd.Series:
    """Sum encoded-column attributions back onto their source feature.

    One-hot encoding turns ``customer_state`` into dozens of columns. A
    business reader needs one statement about customer state, not thirty
    about individual levels, so contributions are summed over the columns
    derived from each original feature. This is valid because SHAP values
    are additive.
    """
    totals = pd.Series(0.0, index=original, dtype="float64")
    for name, value in zip(encoded_names, values):
        # ColumnTransformer emits "<feature>_<level>" for one-hot columns and
        # the bare name for numeric ones. Longest match wins so that
        # "seller_state" is not absorbed by "state".
        matches = [f for f in original if name == f or name.startswith(f + "_")]
        if matches:
            totals[max(matches, key=len)] += value
    return totals


def explain_orders(model, X: pd.DataFrame, background: pd.DataFrame | None = None,
                   max_background: int = 200) -> list[LocalExplanation]:
    """Compute local SHAP attributions for each row of ``X``.

    ``background`` supplies the reference distribution against which
    contributions are measured; a sample of the training data is the right
    choice. Without it the explanation is relative to the rows being
    explained, which makes attributions incomparable between batches.
    """
    import shap

    inner, calibrated = unwrap(model)
    prep = inner.named_steps["prep"]
    clf = inner.named_steps["clf"]

    encoded = prep.transform(X)
    encoded_names = list(prep.get_feature_names_out())
    original = list(X.columns)

    if background is None or len(background) == 0:
        bg = encoded
    else:
        sample = background.sample(min(len(background), max_background), random_state=0)
        bg = prep.transform(sample)

    explainer = shap.Explainer(clf, bg, feature_names=encoded_names)
    shap_values = explainer(encoded)

    values = np.asarray(shap_values.values)
    if values.ndim == 3:  # some explainers return one column per class
        values = values[:, :, -1]
    base = np.asarray(shap_values.base_values)
    base = base[:, -1] if base.ndim == 2 else base

    probabilities = model.predict_proba(X)[:, 1]

    return [
        LocalExplanation(
            order_index=X.index[i],
            calibrated_probability=float(probabilities[i]),
            base_value=float(np.ravel(base)[i]),
            contributions=_aggregate_to_original(values[i], encoded_names, original),
            feature_values=X.iloc[i],
            calibrated=calibrated,
        )
        for i in range(len(X))
    ]


def explain_order(model, row: pd.DataFrame, background: pd.DataFrame | None = None) -> LocalExplanation:
    """Convenience wrapper for a single-row frame."""
    if len(row) != 1:
        raise ValueError(f"Expected exactly one row, got {len(row)}")
    return explain_orders(model, row, background)[0]


def counterfactual(model, row: pd.DataFrame, feature: str, value: float) -> float:
    """Recompute calibrated risk with one feature set to ``value``.

    This is the *what if* rung of the framework. It runs the full deployed
    pipeline rather than approximating from SHAP values, so the resulting
    probability is the one the system would actually produce -- which is what
    makes it safe to display next to a costed recommendation.

    Note the limit: this holds every other feature fixed. Dispatching faster
    in reality may correlate with other changes the model cannot see, so the
    result is the model's answer, not a causal claim.
    """
    if feature not in row.columns:
        raise KeyError(f"{feature!r} is not a feature of this model")
    perturbed = row.copy()
    perturbed[feature] = value
    return float(model.predict_proba(perturbed)[0, 1])
