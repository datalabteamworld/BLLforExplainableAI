"""Prediction confidence.

The naive implementation is ``confidence = predicted_probability``. That is
wrong, and the spec explicitly asks why. A calibrated model outputting 8% on a
typical order is *confident* that the order is safe; the naive formula would
report 8% confidence. Probability answers "how likely is lateness"; confidence
answers "how much should this estimate be trusted".

This module separates the two and builds confidence from three measurable
signals, each of which can be computed from the deployed artefact and the
training data:

1. **Distance from the decision threshold.** A prediction at the operating
   point could flip either way; one far from it would not change under small
   perturbations. This is the only component many systems use.

2. **Familiarity of the input.** An order whose features sit outside the range
   the model was trained on is an extrapolation. Measured as the share of
   features falling within the training interquartile range, so it does not
   assume normality.

3. **Local evidence density.** How many similar historical orders exist. A
   region of feature space with few training examples supports a weaker claim
   than a dense one.

None of these is a probability of correctness, and this module does not claim
otherwise. They are combined into a bounded score and reported as an estimate,
with the contributing reasons exposed so a user can see *why* a prediction is
uncertain rather than being handed a bare number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Confidence bands. Chosen so that "low" flags predictions a business user
#: should not act on without review, rather than to make the numbers look good.
CONFIDENCE_BANDS = [(0.70, "HIGH"), (0.45, "MEDIUM"), (0.0, "LOW")]


@dataclass
class ConfidenceEstimate:
    """A confidence score with its components and human-readable reasons."""

    score: float
    level: str
    threshold_margin: float
    familiarity: float
    evidence_density: float
    reasons: list[str] = field(default_factory=list)
    is_estimate: bool = True

    @property
    def is_low(self) -> bool:
        return self.level == "LOW"


@dataclass
class TrainingProfile:
    """Summary of the training distribution, used to judge unfamiliarity.

    Stored alongside the model so confidence can be computed at serving time
    without shipping the training set.
    """

    q1: pd.Series
    q3: pd.Series
    median: pd.Series
    numeric_columns: list[str]
    n_train: int

    @classmethod
    def from_frame(cls, X: pd.DataFrame) -> "TrainingProfile":
        numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        block = X[numeric]
        return cls(
            q1=block.quantile(0.25),
            q3=block.quantile(0.75),
            median=block.median(),
            numeric_columns=numeric,
            n_train=len(X),
        )


def _threshold_margin(probability: float, threshold: float) -> float:
    """How far the prediction sits from the decision boundary, scaled to 0-1.

    Normalised by the larger side of the threshold so that a low operating
    point (0.20, as here) does not mechanically produce low confidence for
    every prediction below it.
    """
    reach = max(threshold, 1.0 - threshold)
    return float(min(abs(probability - threshold) / reach, 1.0))


def _familiarity(row: pd.Series, profile: TrainingProfile) -> tuple[float, list[str]]:
    """Share of features inside the training interquartile range.

    Uses the IQR rather than a standard-deviation band because these features
    are heavily skewed -- freight, weight and distance all have long right
    tails, and a mean/sd rule would call ordinary large orders anomalous.

    A tolerance of 1.5 x IQR beyond each quartile is the conventional Tukey
    fence, so "unfamiliar" here means what it usually means in outlier
    detection.
    """
    inside, unusual = 0, []
    for col in profile.numeric_columns:
        value = row.get(col)
        if value is None or pd.isna(value):
            continue
        q1, q3 = profile.q1[col], profile.q3[col]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        if lo <= value <= hi:
            inside += 1
        else:
            unusual.append(col)

    total = sum(1 for c in profile.numeric_columns if not pd.isna(row.get(c, np.nan)))
    score = inside / total if total else 0.0
    return score, unusual


def _evidence_density(row: pd.Series, reference: pd.DataFrame,
                      profile: TrainingProfile, k: int = 50) -> float:
    """How densely populated this region of feature space is.

    Distance is computed on IQR-scaled features so that no single large-valued
    column dominates. The score is the inverse of mean distance to the k
    nearest historical orders, squashed to 0-1.
    """
    cols = [c for c in profile.numeric_columns if c in reference.columns]
    if not cols or reference.empty:
        return 0.0

    scale = (profile.q3[cols] - profile.q1[cols]).replace(0, np.nan)
    ref = ((reference[cols] - profile.median[cols]) / scale).fillna(0.0).to_numpy()
    point = ((row[cols] - profile.median[cols]) / scale).fillna(0.0).to_numpy(dtype="float64")

    distances = np.sqrt(((ref - point) ** 2).sum(axis=1))
    nearest = np.sort(distances)[: min(k, len(distances))]
    mean_distance = float(nearest.mean())

    # A mean scaled distance of ~1 across features is unremarkable; larger
    # means the point sits away from the bulk of the data.
    return float(1.0 / (1.0 + mean_distance / max(len(cols) ** 0.5, 1.0)))


def estimate_confidence(
    probability: float,
    row: pd.Series,
    threshold: float,
    profile: TrainingProfile,
    reference: pd.DataFrame | None = None,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> ConfidenceEstimate:
    """Combine the three signals into a bounded confidence estimate.

    Weights favour threshold margin because it is the component most directly
    tied to whether the *decision* would change, which is what a business user
    is acting on. They are exposed as a parameter rather than hard-coded so
    the balance can be revisited against user feedback.
    """
    margin = _threshold_margin(probability, threshold)
    familiarity, unusual = _familiarity(row, profile)
    density = _evidence_density(row, reference, profile) if reference is not None else 0.5

    w_margin, w_familiar, w_density = weights
    score = w_margin * margin + w_familiar * familiarity + w_density * density
    score = float(np.clip(score, 0.0, 1.0))
    level = next(label for cutoff, label in CONFIDENCE_BANDS if score >= cutoff)

    reasons = []
    if margin < 0.25:
        reasons.append(
            f"The predicted probability ({probability:.0%}) sits close to the "
            f"decision threshold ({threshold:.0%}), so the recommendation could "
            f"flip on a small change in the inputs."
        )
    if unusual:
        # Reported whenever any feature is out of range, not only when the
        # familiarity score drops below a cutoff. A handful of extreme values
        # among many ordinary ones barely moves the score, but it is exactly
        # what a user needs to see before acting.
        shown = ", ".join(unusual[:4])
        reasons.append(
            f"Some inputs fall outside the range the model was trained on: {shown}"
            + (f" and {len(unusual) - 4} others." if len(unusual) > 4 else ".")
        )
    if density < 0.4:
        reasons.append(
            "Few historical orders resemble this one, so there is limited "
            "evidence supporting the estimate."
        )
    if pd.isna(row).any():
        missing = [c for c in row.index if pd.isna(row[c])]
        reasons.append(
            f"{len(missing)} feature(s) are missing and were filled with training "
            f"medians, which weakens the estimate."
        )
    if not reasons:
        reasons.append(
            "The prediction is far from the decision threshold, the inputs are "
            "typical of the training data, and similar historical orders exist."
        )

    return ConfidenceEstimate(
        score=score, level=level, threshold_margin=margin,
        familiarity=familiarity, evidence_density=density, reasons=reasons,
    )
