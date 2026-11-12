"""Preprocessing, model construction and data splitting.

Everything the model needs -- imputation, encoding, scaling, resampling -- is
inside a single ``imblearn.Pipeline``. That is not a stylistic preference: it
is what makes the cross-validation honest.

In the original notebook, SMOTE was applied to the whole training set and the
resampled matrix was then handed to ``cross_val_score``. Synthetic minority
points in a validation fold had been interpolated from real points sitting in
the training fold, so the folds were not independent. The reported CV AUC of
0.99 against a test AUC of 0.86 is that leak, not overfitting in the ordinary
sense. Inside a pipeline, SMOTE is refitted within each fold and never touches
validation data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBClassifier

from .config import (
    CONFIG,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    POST_DISPATCH_FEATURES,
    RANDOM_STATE,
    TARGET,
    TIME_KEY,
    UNBOUNDED_TIME_COUNTERS,
)

logger = logging.getLogger(__name__)

#: Categorical columns with few levels -- one-hot encoded.
LOW_CARDINALITY = ["customer_state", "seller_state", "payment_type"]

#: Categorical columns with many levels. Target encoding is used rather than
#: label encoding: label encoding imposes an arbitrary ordinal ranking that is
#: meaningless to a linear or distance-based model, and sklearn's
#: ``TargetEncoder`` fits its encodings under internal cross-fitting, so it
#: does not leak the target into the training rows.
HIGH_CARDINALITY = ["customer_city", "seller_city", "product_category"]

#: Columns dropped for reasons other than leakage: free text, raw timestamps
#: already decomposed into features, and post-hoc bookkeeping.
NON_FEATURE_COLUMNS = [
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
    "product_category_name",
    "product_category_name_english",
    "seller_city_raw",
    "has_review_comment",
]


# ==========================================================================
# Splitting
# ==========================================================================


def temporal_split(
    df: pd.DataFrame,
    train_end: str | None = None,
    time_key: str = TIME_KEY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by purchase date: train on the past, test on the future.

    A random split lets the model train on December 2017 and be tested on
    November 2017. It then scores well by memorising which months had high
    late rates -- which is why ``purchase_quarter`` and ``purchase_month``
    topped the original feature importances. Deployment never offers that
    information, so the random-split score is not an estimate of anything
    useful.
    """
    train_end = pd.Timestamp(train_end or CONFIG.split.train_end)
    is_train = df[time_key] <= train_end

    train, test = df.loc[is_train].copy(), df.loc[~is_train].copy()
    if len(test) < CONFIG.split.min_test_orders:
        logger.warning(
            "Test set has only %d orders (< %d). Move train_end earlier.",
            len(test), CONFIG.split.min_test_orders,
        )

    logger.info(
        "Temporal split at %s | train %d (%s to %s, late %.2f%%) | test %d (%s to %s, late %.2f%%)",
        train_end.date(), len(train),
        train[time_key].min().date(), train[time_key].max().date(),
        100 * train[TARGET].mean(), len(test),
        test[time_key].min().date(), test[time_key].max().date(),
        100 * test[TARGET].mean(),
    )
    return train, test


def feature_columns(df: pd.DataFrame, scenario: str = "pre_dispatch",
                    use_history: bool = True) -> list[str]:
    """Columns usable as features under a given prediction scenario.

    ``pre_dispatch`` -- known the moment the order is placed. This is the
    scenario the business case requires, because an intervention is only
    possible before the parcel moves.

    ``post_dispatch`` -- additionally allows carrier-handover timings. More
    accurate, but usable only once the seller has handed the parcel over, so
    it supports a narrower set of interventions.

    The original notebook mixed the two: it dropped ``actual_delivery_days``
    as unknown at order time, then kept ``carrier_pickup_days``, which is
    equally unknown, as its third most important feature.
    """
    if scenario not in {"pre_dispatch", "post_dispatch"}:
        raise ValueError(f"Unknown scenario {scenario!r}")

    excluded = (set(LEAKAGE_COLUMNS) | set(ID_COLUMNS) | set(NON_FEATURE_COLUMNS)
                | set(UNBOUNDED_TIME_COUNTERS))
    if not use_history:
        excluded |= {c for c in df.columns if c.endswith(("_to_date", "_history_confidence"))}
    if scenario == "pre_dispatch":
        excluded |= set(POST_DISPATCH_FEATURES)
        # approval happens within minutes of purchase, so it is available in
        # both scenarios; shipping-limit slack is contractual and known up front.

    cols = [c for c in df.columns if c not in excluded]
    dropped_leak = sorted(set(df.columns) & set(LEAKAGE_COLUMNS))
    logger.info("Scenario %s: %d features (leakage columns excluded: %s)",
                scenario, len(cols), ", ".join(dropped_leak))
    return cols


# ==========================================================================
# Preprocessing
# ==========================================================================


def _target_encoder() -> TargetEncoder:
    """Build a reproducible TargetEncoder across scikit-learn versions.

    ``random_state``/``shuffle`` were deprecated in 1.9 in favour of passing
    a cross-validation generator as ``cv``, and are removed in 1.11. Passing
    an explicit ``StratifiedKFold`` works on every version from 1.4 onward
    and keeps the internal cross-fitting deterministic.
    """
    if Version(sklearn_version) >= Version("1.9"):
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        return TargetEncoder(target_type="binary", cv=inner_cv)
    # Before 1.9 the ``cv`` parameter accepts only an int; reproducibility
    # comes from ``random_state`` instead. Note the constructor does not
    # validate, so this must be a version check rather than try/except.
    return TargetEncoder(target_type="binary", random_state=RANDOM_STATE)


def build_preprocessor(numeric: list[str], low_card: list[str], high_card: list[str]) -> ColumnTransformer:
    """Column-wise preprocessing.

    Imputers and scalers are fitted inside the pipeline, so they see only the
    training fold. Fitting them on the full dataset before splitting -- as in
    the original winsorisation step -- lets test-set statistics influence the
    transform.
    """
    numeric_tf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    low_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                                 sparse_output=False)),
    ])
    high_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("target", _target_encoder()),
        ("scale", StandardScaler()),
    ])

    return ColumnTransformer(
        [
            ("num", numeric_tf, numeric),
            ("low", low_tf, low_card),
            ("high", high_tf, high_card),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_column_types(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Partition feature columns into numeric, low- and high-cardinality."""
    low = [c for c in LOW_CARDINALITY if c in cols]
    high = [c for c in HIGH_CARDINALITY if c in cols]
    numeric = [
        c for c in cols
        if c not in low + high and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric, low, high


# ==========================================================================
# Models
# ==========================================================================


def gpu_available() -> bool:
    """True when XGBoost can actually use a CUDA device.

    On a Colab T4 this cuts XGBoost fitting time substantially, which matters
    most during search where the model is refitted hundreds of times.

    Detection is deliberately indirect. XGBoost does not raise when no GPU is
    present: it logs a fallback message from its C++ layer -- which bypasses
    Python's ``warnings`` machinery entirely -- and silently trains on CPU. So
    neither try/except nor a warnings filter can detect it. Instead, check
    that the library was built with CUDA *and* that a device is visible.
    """
    import shutil
    import subprocess

    try:
        import xgboost as xgb

        if not xgb.build_info().get("USE_CUDA", False):
            return False
        if shutil.which("nvidia-smi") is None:
            return False
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=15,
        )
        return probe.returncode == 0 and bool(probe.stdout.strip())
    except Exception:
        return False


def model_zoo(scale_pos_weight: float = 1.0, n_jobs: int = -1) -> dict[str, object]:
    """Candidate estimators.

    Note that ``class_weight`` is *not* set here. The pipeline already
    rebalances with SMOTE; stacking class weights on top of it double-counts
    the correction and pushes the decision boundary further than intended.
    Imbalance is handled once, by the resampler, and the operating point is
    then chosen explicitly by threshold tuning.
    """
    device = "cuda" if gpu_available() else "cpu"
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=n_jobs, random_state=RANDOM_STATE
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            n_jobs=n_jobs, random_state=RANDOM_STATE,
            tree_method="hist", device=device,
        ),
    }


#: Search spaces for ``RandomizedSearchCV``. Prefixed for the pipeline step.
SEARCH_SPACES = {
    "logistic_regression": {
        "clf__C": np.logspace(-3, 2, 20),
        "clf__penalty": ["l2"],
        "smote__k_neighbors": [3, 5, 7],
    },
    "random_forest": {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [6, 8, 12, 16, None],
        "clf__min_samples_leaf": [1, 3, 5, 10, 20],
        "clf__max_features": ["sqrt", "log2", 0.3],
        "smote__k_neighbors": [3, 5, 7],
    },
    "xgboost": {
        "clf__n_estimators": [200, 400, 600, 800],
        "clf__max_depth": [3, 4, 6, 8, 10],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5, 10],
        "clf__reg_lambda": [0.5, 1.0, 3.0, 10.0],
        "smote__k_neighbors": [3, 5, 7],
    },
}


def build_pipeline(
    df: pd.DataFrame,
    cols: list[str],
    model_name: str = "xgboost",
    use_smote: bool | None = None,
    y: pd.Series | None = None,
    n_jobs: int = -1,
) -> ImbPipeline:
    """Assemble preprocessing + optional SMOTE + classifier into one estimator.

    Returning a single fitted object also removes the deployment mismatch in
    the original Streamlit app, which had to reconstruct an 88-column frame
    from ``scaler.feature_names_in_`` to feed a 34-feature model and could
    not reproduce its ``LabelEncoder`` mappings at all.
    """
    # Gradient boosting handles imbalance through ``scale_pos_weight`` without
    # synthesising rows. SMOTE on ~70k rows is a large share of each fit, and
    # it buys XGBoost nothing that the weight does not, so it is off by
    # default there. Linear and distance-based models still benefit.
    if use_smote is None:
        use_smote = model_name != "xgboost"

    spw = 1.0
    if not use_smote and y is not None and y.sum():
        spw = float((len(y) - y.sum()) / y.sum())

    numeric, low, high = split_column_types(df, cols)
    steps = [("prep", build_preprocessor(numeric, low, high))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)))
    steps.append(("clf", model_zoo(scale_pos_weight=spw, n_jobs=n_jobs)[model_name]))
    return ImbPipeline(steps)


# ==========================================================================
# Calibration
# ==========================================================================


def fit_calibrated(
    pipe: ImbPipeline,
    train: pd.DataFrame,
    cols: list[str],
    target: str = TARGET,
    holdout_frac: float = 0.2,
    method: str = "sigmoid",
):
    """Fit, then calibrate on a held-out *later* slice of the training period.

    SMOTE trains the classifier on an artificially balanced world, so its
    outputs are not probabilities of anything real. Logistic regression showed
    this plainly: a Brier score of 0.266 against XGBoost's 0.051, and a
    cost-optimal threshold of 0.77 on a problem with a 6% base rate. Rankings
    were fine; the numbers attached to them were not.

    That matters wherever a probability is displayed or fed into an expected-
    value calculation rather than used purely to sort a queue.

    The calibration slice is taken by date, not at random, so the mapping is
    learned on data later than what the classifier saw -- the same direction
    as deployment. Calibrating on random rows would let the classifier's own
    training period set its correction.

    ``method="sigmoid"`` (Platt scaling) is the default rather than isotonic.
    Isotonic fits a step function: on this data it collapsed 3,812 distinct
    probabilities into 48 levels, and the resulting ties destroy ranking
    information -- average precision fell even though calibration improved.
    Sigmoid is strictly monotone, so AUC and average precision are preserved
    exactly while the Brier score improves just as much.
    """
    from sklearn.calibration import CalibratedClassifierCV

    try:  # scikit-learn >= 1.6 replaced cv="prefit" with FrozenEstimator
        from sklearn.frozen import FrozenEstimator
    except ImportError:
        FrozenEstimator = None

    ordered = train.sort_values(TIME_KEY)
    cut = int(len(ordered) * (1 - holdout_frac))
    fit_part, calib_part = ordered.iloc[:cut], ordered.iloc[cut:]

    if calib_part[target].sum() < 50:
        logger.warning(
            "Only %d positives in the calibration slice; skipping calibration.",
            int(calib_part[target].sum()),
        )
        pipe.fit(train[cols], train[target])
        return pipe

    pipe.fit(fit_part[cols], fit_part[target])
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method=method)
    else:
        calibrated = CalibratedClassifierCV(pipe, method=method, cv="prefit")
    calibrated.fit(calib_part[cols], calib_part[target])

    logger.info(
        "Calibrated on %d orders (%s to %s), %d positives",
        len(calib_part), calib_part[TIME_KEY].min().date(),
        calib_part[TIME_KEY].max().date(), int(calib_part[target].sum()),
    )
    return calibrated
