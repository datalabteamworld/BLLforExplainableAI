"""Training entry point.

Run end to end::

    python -m olist.train --scenario pre_dispatch --tune

Produces, under ``models/`` and ``reports/``: a single pickled pipeline that
takes a raw feature frame and returns calibrated probabilities, the chosen
decision threshold, a metrics table, and a permutation-importance ranking.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
)

from . import cleaning, features, io, pipeline
from .config import (
    CONFIG,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    TIME_KEY,
)
from .evaluate import comparison_table, evaluate

logger = logging.getLogger(__name__)


def prepare_data(use_cache: bool = True) -> pd.DataFrame:
    """Load, clean and engineer features, caching the result to parquet."""
    cache = PROCESSED_DIR / "orders_features.parquet"
    if use_cache:
        cached = io.read_dataset(cache)
        if cached is not None:
            logger.info("Loading cached feature table from %s", cache)
            return cached

    raw = io.load_all()
    cleaned, reports = cleaning.clean_all(raw)
    (REPORTS_DIR / "cleaning_report.txt").write_text(
        "\n".join(str(r) for r in reports), encoding="utf-8"
    )

    df = features.build_modelling_table(cleaned)
    written = io.write_dataset(df, cache)
    logger.info("Cached feature table to %s", written)
    return df


def fit_model(
    train: pd.DataFrame,
    cols: list[str],
    model_name: str,
    tune: bool,
    fast: bool = False,
    calibrate: bool = True,
) -> tuple[object, dict]:
    """Fit one model, optionally with a randomised hyperparameter search.

    The search cross-validates the *whole pipeline*, so SMOTE, imputation,
    scaling and target encoding are all refitted inside each fold. Scoring
    uses average precision rather than AUC-ROC: with a ~8% positive rate,
    AUC-ROC is dominated by the majority class and barely moves between a
    useful model and a useless one.
    """
    X, y = train[cols], train[TARGET]
    # The estimator runs single-threaded during search: the search itself is
    # parallel, and nesting ``n_jobs=-1`` inside ``n_jobs=-1`` oversubscribes
    # the CPU and inflates memory until workers are killed.
    inner_jobs = 1 if tune else -1
    pipe = pipeline.build_pipeline(train, cols, model_name, y=y, n_jobs=inner_jobs)

    if not tune:
        if not calibrate:
            pipe.fit(X, y)
            return pipe, {}
        return pipeline.fit_calibrated(pipe, train, cols), {}

    # Rows arrive sorted by purchase date, so TimeSeriesSplit produces folds
    # that always validate on data later than they trained on -- the same
    # direction as the temporal test split.
    if CONFIG.tuning.temporal_cv:
        cv = TimeSeriesSplit(n_splits=CONFIG.tuning.cv_folds)
        X, y = X.copy(), y.copy()
        order = train[TIME_KEY].argsort()
        X, y = X.iloc[order], y.iloc[order]
    else:
        cv = StratifiedKFold(
            n_splits=CONFIG.tuning.cv_folds, shuffle=True, random_state=RANDOM_STATE
        )
    space = dict(pipeline.SEARCH_SPACES[model_name])
    if "smote" not in dict(pipe.steps):
        space.pop("smote__k_neighbors", None)

    if fast:
        # Successive halving: run many configurations on a small subsample,
        # keep the best, and give survivors progressively more data. Reaches a
        # comparable optimum for a fraction of the full-fit budget.
        search = HalvingRandomSearchCV(
            pipe, param_distributions=space,
            n_candidates=CONFIG.tuning.n_iter, factor=3,
            resource="n_samples", min_resources=max(2000, len(X) // 20),
            scoring=CONFIG.tuning.scoring, cv=cv,
            n_jobs=CONFIG.tuning.n_jobs, random_state=RANDOM_STATE,
            refit=True, verbose=0,
        )
    else:
        search = RandomizedSearchCV(
            pipe, param_distributions=space,
            n_iter=CONFIG.tuning.n_iter,
            scoring=CONFIG.tuning.scoring, cv=cv,
            n_jobs=CONFIG.tuning.n_jobs, random_state=RANDOM_STATE,
            refit=True, verbose=0,
        )
    start = time.time()
    search.fit(X, y)
    logger.info(
        "%s: best CV %s = %.4f (%.0fs) | params %s",
        model_name, CONFIG.tuning.scoring, search.best_score_,
        time.time() - start, search.best_params_,
    )
    best = pipeline.build_pipeline(train, cols, model_name, y=y).set_params(
        **search.best_params_
    )
    if calibrate:
        final = pipeline.fit_calibrated(best, train, cols)
    else:
        best.fit(X, y)
        final = best
    return final, {
        "best_cv_score": float(search.best_score_),
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
    }


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run(scenario: str = "pre_dispatch", tune: bool = False,
        models: list[str] | None = None, fast: bool = False,
        calibrate: bool = True, use_history: bool = True) -> pd.DataFrame:
    df = prepare_data()
    train, test = pipeline.temporal_split(df)
    cols = pipeline.feature_columns(df, scenario, use_history=use_history)

    results, fitted, search_info = [], {}, {}
    for name in models or list(pipeline.model_zoo()):
        logger.info("--- %s (%s) ---", name, scenario)
        model, info = fit_model(train, cols, name, tune, fast, calibrate)
        proba = model.predict_proba(test[cols])[:, 1]
        results.append(evaluate(test[TARGET].to_numpy(), proba, CONFIG.cost, label=name))
        fitted[name] = model
        search_info[name] = info

    table = comparison_table(results)

    # Selection is by expected business cost, not by AUC. A model that ranks
    # well but cannot be operated at a profitable threshold is not the winner.
    best_name = table["cost_per_order"].idxmin()
    best_result = next(r for r in results if r["metrics"]["model"] == best_name)
    logger.info("Selected %s: cost/order %.3f, recall %.3f at threshold %.3f",
                best_name, best_result["metrics"]["cost_per_order"],
                best_result["metrics"]["recall_at_chosen"],
                best_result["metrics"]["chosen_threshold"])

    _persist(fitted[best_name], best_name, best_result, table, cols, scenario, search_info)
    _permutation_importance(fitted[best_name], test, cols, scenario)
    return table


def _persist(model, name, result, table, cols, scenario, search_info) -> None:
    artefact = {
        "pipeline": model,
        "model_name": name,
        "scenario": scenario,
        "feature_columns": cols,
        "threshold": result["threshold"]["threshold"],
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
    }
    joblib.dump(artefact, MODELS_DIR / f"model_{scenario}.joblib")

    table.to_csv(REPORTS_DIR / f"metrics_{scenario}.csv")
    result["cost_curve"].to_csv(REPORTS_DIR / f"cost_curve_{scenario}.csv", index=False)
    (REPORTS_DIR / f"run_{scenario}.json").write_text(
        json.dumps({"selected": name, "threshold": result["threshold"],
                    "metrics": {k: _jsonable(v) for k, v in result["metrics"].items()},
                    "search": search_info}, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved model_%s.joblib and reports", scenario)


def _permutation_importance(model, test, cols, scenario, n_repeats: int = 5) -> None:
    """Permutation importance on the test set.

    Impurity-based importances are biased toward high-cardinality continuous
    features -- which is why ``customer_lng`` and ``product_length_cm``
    outranked real signals in the original run. Permutation importance
    measures the actual drop in held-out performance when a column is
    shuffled.
    """
    result = permutation_importance(
        model, test[cols], test[TARGET],
        n_repeats=n_repeats, random_state=RANDOM_STATE,
        scoring="average_precision", n_jobs=-1,
    )
    (
        pd.DataFrame({"feature": cols,
                      "importance_mean": result.importances_mean,
                      "importance_std": result.importances_std})
        .sort_values("importance_mean", ascending=False)
        .to_csv(REPORTS_DIR / f"permutation_importance_{scenario}.csv", index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Olist late-delivery risk model.")
    parser.add_argument("--scenario", default="pre_dispatch",
                        choices=["pre_dispatch", "post_dispatch"])
    parser.add_argument("--tune", action="store_true", help="run randomised hyperparameter search")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--random-cv", action="store_true",
                        help="use random CV folds instead of TimeSeriesSplit "
                             "(rewards memorising seasonality; not recommended)")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="skip probability calibration (for A/B comparison)")
    parser.add_argument("--no-history", action="store_true",
                        help="drop seller/route/category history features")
    parser.add_argument("--fast", action="store_true",
                        help="successive-halving search: much faster, near-equal result")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    if args.random_cv:
        object.__setattr__(CONFIG.tuning, 'temporal_cv', False)
    table = run(args.scenario, args.tune, args.models, args.fast,
                calibrate=not args.no_calibrate, use_history=not args.no_history)
    print("\n" + table.to_string())


if __name__ == "__main__":
    main()
