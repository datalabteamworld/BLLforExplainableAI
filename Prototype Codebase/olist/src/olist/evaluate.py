"""Evaluation, threshold selection and business framing.

The default 0.5 decision threshold encodes an assumption almost never true in
operations: that a false positive and a false negative cost the same. Here
they plainly do not. Missing a late delivery costs a refund, a support
contact, and the churn implied by the review-score drop; flagging an on-time
order costs one proactive message or a shipping upgrade.

Selecting the threshold that minimises expected cost is what turns a ranking
model into a decision system, and it is why comparing models by AUC alone was
misleading in the original notebook: XGBoost won on AUC (0.862) while
recalling only 29.6% of late deliveries -- the worst of the four -- on a
problem whose entire justification was catching them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import TARGET, TIME_KEY, CostConfig

logger = logging.getLogger(__name__)


@dataclass
class ThresholdChoice:
    """The selected operating point and what it costs."""

    threshold: float
    expected_cost: float
    cost_per_order: float
    baseline_cost: float
    savings_vs_no_model: float
    precision: float
    recall: float
    n_flagged: int
    flag_rate: float


def expected_cost_curve(
    y_true: np.ndarray, y_proba: np.ndarray, cost: CostConfig
) -> pd.DataFrame:
    """Total expected cost across candidate thresholds.

    Cost at a threshold is ``cost_fn * FN + cost_fp * FP``. Thresholds are
    taken from the precision-recall curve, which uses every distinct
    predicted probability, so the true minimum is always a candidate.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one more precision/recall than threshold.
    precision, recall = precision[:-1], recall[:-1]

    n_positive = int(y_true.sum())
    tp = recall * n_positive
    fn = n_positive - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted_positive = np.where(precision > 0, tp / precision, 0.0)
    fp = np.maximum(predicted_positive - tp, 0.0)

    return pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "n_flagged": predicted_positive,
        "false_negatives": fn,
        "false_positives": fp,
        "expected_cost": cost.cost_fn * fn + cost.cost_fp * fp,
    })


def choose_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, cost: CostConfig
) -> tuple[ThresholdChoice, pd.DataFrame]:
    """Pick the threshold minimising expected cost, and report the saving.

    The baseline is doing nothing: every late delivery is missed, costing
    ``cost_fn`` each. A model that cannot beat that baseline is not worth
    deploying regardless of its AUC.
    """
    curve = expected_cost_curve(y_true, y_proba, cost)
    best = curve.loc[curve["expected_cost"].idxmin()]

    baseline = cost.cost_fn * float(y_true.sum())
    n = len(y_true)

    choice = ThresholdChoice(
        threshold=float(best["threshold"]),
        expected_cost=float(best["expected_cost"]),
        cost_per_order=float(best["expected_cost"]) / n,
        baseline_cost=baseline,
        savings_vs_no_model=baseline - float(best["expected_cost"]),
        precision=float(best["precision"]),
        recall=float(best["recall"]),
        n_flagged=int(round(best["n_flagged"])),
        flag_rate=float(best["n_flagged"]) / n,
    )
    return choice, curve


def recall_at_budget(y_true: np.ndarray, y_proba: np.ndarray, budget: float) -> float:
    """Recall achieved when only the top ``budget`` fraction can be actioned.

    A logistics team can chase a fixed number of orders per week, not an
    arbitrary number set by a probability cutoff. This is usually the metric
    an operations stakeholder actually wants.
    """
    k = max(1, int(round(budget * len(y_proba))))
    top_k = np.argsort(y_proba)[::-1][:k]
    return float(y_true[top_k].sum() / max(y_true.sum(), 1))


def holdout_threshold(
    model,
    train: pd.DataFrame,
    cols: list[str],
    cost: CostConfig,
    holdout_frac: float = 0.2,
    target: str = TARGET,
    time_key: str = TIME_KEY,
) -> float:
    """Cost-optimal threshold chosen on the latest slice of the training period.

    The threshold is a fitted parameter like any other. Choosing it on the
    test set -- picking the cutoff that minimises cost on the very labels it
    is then scored against -- makes the reported recall and precision an
    upper bound rather than an estimate. Here it is chosen on the same
    date-ordered holdout ``pipeline.fit_calibrated`` calibrates on: later
    than anything the classifier was fitted on, earlier than anything in test.
    """
    ordered = train.sort_values(time_key)
    holdout = ordered.iloc[int(len(ordered) * (1 - holdout_frac)):]
    proba = model.predict_proba(holdout[cols])[:, 1]
    choice, _ = choose_threshold(holdout[target].to_numpy(), proba, cost)
    return choice.threshold


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost: CostConfig,
    label: str = "model",
    budgets: tuple[float, ...] = (0.05, 0.10, 0.20),
    threshold: float | None = None,
) -> dict:
    """Full metric set: ranking quality, calibration and the operating point.

    ``threshold`` should be chosen without the evaluation labels -- see
    :func:`holdout_threshold`. The ``*_at_chosen`` metrics are reported at
    it. The cost-optimal threshold on the evaluation set itself is reported
    alongside as ``oracle_threshold`` / ``*_at_oracle``: an upper bound on
    what any threshold could achieve, not an estimate of deployed
    performance. If ``threshold`` is omitted the oracle is used, and
    ``threshold_source`` says so.
    """
    y_true = np.asarray(y_true)
    oracle, curve = choose_threshold(y_true, y_proba, cost)
    source = "holdout" if threshold is not None else "oracle (evaluation set)"
    if threshold is None:
        threshold = oracle.threshold
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    y_oracle = (y_proba >= oracle.threshold).astype(int)

    n = len(y_true)
    expected_cost = cost.cost_fn * fn + cost.cost_fp * fp
    baseline = cost.cost_fn * float(y_true.sum())

    metrics = {
        "model": label,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "positive_rate": float(y_true.mean()),
        # Metrics at the default cutoff, for comparison with the original run.
        "recall_at_0.5": recall_score(y_true, y_proba >= 0.5, zero_division=0),
        "precision_at_0.5": precision_score(y_true, y_proba >= 0.5, zero_division=0),
        "f1_at_0.5": f1_score(y_true, y_proba >= 0.5, zero_division=0),
        # Metrics at the operating threshold.
        "threshold_source": source,
        "chosen_threshold": float(threshold),
        "recall_at_chosen": recall_score(y_true, y_pred, zero_division=0),
        "precision_at_chosen": precision_score(y_true, y_pred, zero_division=0),
        "f1_at_chosen": f1_score(y_true, y_pred, zero_division=0),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "flag_rate": float(y_pred.mean()),
        "cost_per_order": float(expected_cost) / n,
        "savings_vs_no_model": baseline - float(expected_cost),
        # Upper bound: the threshold chosen with the evaluation labels.
        "oracle_threshold": oracle.threshold,
        "recall_at_oracle": recall_score(y_true, y_oracle, zero_division=0),
        "precision_at_oracle": precision_score(y_true, y_oracle, zero_division=0),
        "cost_per_order_at_oracle": oracle.cost_per_order,
    }
    for b in budgets:
        metrics[f"recall_at_{int(b * 100)}pct_budget"] = recall_at_budget(y_true, y_proba, b)

    operating = asdict(oracle)
    operating.update(
        threshold=float(threshold), expected_cost=float(expected_cost),
        cost_per_order=float(expected_cost) / n, baseline_cost=baseline,
        savings_vs_no_model=baseline - float(expected_cost),
        precision=metrics["precision_at_chosen"], recall=metrics["recall_at_chosen"],
        n_flagged=int(y_pred.sum()), flag_rate=float(y_pred.mean()),
    )
    return {"metrics": metrics, "threshold": operating, "cost_curve": curve}


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """Tidy side-by-side of several ``evaluate`` outputs."""
    return pd.DataFrame([r["metrics"] for r in results]).set_index("model").round(4)
