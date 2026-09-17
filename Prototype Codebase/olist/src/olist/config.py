"""Central configuration for the Olist late-delivery risk project.

All paths, column groups and modelling constants live here so that no module
hard-codes a filesystem location or a magic number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------

TARGET = "is_late_delivery"
GROUP_KEY = "order_id"
TIME_KEY = "order_purchase_timestamp"

#: Fields that are only knowable *after* the carrier has collected the parcel.
#: Excluded from the pre-dispatch model; available to the post-dispatch model.
POST_DISPATCH_FEATURES = ["carrier_pickup_days", "carrier_pickup_hours"]

#: Fields derived from, or revealed at the same time as, the target. Never
#: features under any scenario.
LEAKAGE_COLUMNS = [
    "actual_delivery_days",
    "delivery_delay_days",
    "order_delivered_customer_date",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
    TARGET,
]

#: Counters that only increase with time. Under a temporal split their test
#: values fall outside the training range entirely -- 89% of test rows exceed
#: the training maximum for ``seller_tenure_days``. Linear models extrapolate
#: them without bound and trees cannot extrapolate at all, so both degrade.
#: The bounded equivalents (``*_late_rate_to_date``, ``*_history_confidence``)
#: carry the same information on a stable scale and are kept.
UNBOUNDED_TIME_COUNTERS = [
    "seller_orders_to_date",
    "route_orders_to_date",
    "category_orders_to_date",
    "customer_state_orders_to_date",
    "seller_tenure_days",
    "market_late_rate_to_date",
]

#: Identifier columns: no predictive signal, high cardinality, drop before fit.
ID_COLUMNS = [
    "order_id",
    "customer_id",
    "customer_unique_id",
    "seller_id_primary",
    "product_id_primary",
]


@dataclass(frozen=True)
class SplitConfig:
    """Temporal hold-out configuration.

    A random split leaks seasonality: the model memorises which months had
    high late rates rather than learning generalisable risk. Orders run
    2016-09 to 2018-10, so we train on the earlier period and test forward.
    """

    train_end: str = "2018-04-30"
    #: Orders after ``train_end`` form the test set.
    min_test_orders: int = 5_000


@dataclass(frozen=True)
class CostConfig:
    """Business cost model used to select the decision threshold.

    ``cost_fn`` is the expected cost of a late delivery that was *not*
    flagged: refund exposure, support contact, and the churn implied by the
    review-score drop. ``cost_fp`` is the cost of intervening on an order
    that would have arrived on time: an expedited-shipping upgrade or a
    proactive customer contact.

    Defaults are placeholders. Override them from ``config.yaml`` with
    Olist-specific figures before quoting any monetary result.
    """

    cost_fn: float = 25.0
    cost_fp: float = 4.0

    @property
    def ratio(self) -> float:
        return self.cost_fn / self.cost_fp


@dataclass(frozen=True)
class TuningConfig:
    """Hyperparameter search settings.

    ``temporal_cv`` selects ``TimeSeriesSplit`` over ``StratifiedKFold``.
    This is not a detail. Random folds shuffle within the training period, so
    a fold may train on March and validate on February -- which rewards
    memorising seasonality. The test set is strictly future data, where that
    memorisation is worthless.

    Measured on this dataset: with random folds, XGBoost scored 0.460 CV
    average precision against 0.259 on test, while a heavily regularised
    logistic regression scored 0.399 against 0.332. The search reliably
    selected the model that exploited seasonality best, and that model
    generalised worst.
    """

    n_iter: int = 40
    cv_folds: int = 5
    scoring: str = "average_precision"
    n_jobs: int = -1
    temporal_cv: bool = True


@dataclass(frozen=True)
class Config:
    split: SplitConfig = field(default_factory=SplitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    random_state: int = RANDOM_STATE


CONFIG = Config()
