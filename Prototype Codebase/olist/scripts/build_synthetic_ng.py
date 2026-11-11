"""Build the synthetic Nigerian demonstration environment, end to end.

    python scripts/build_synthetic_ng.py              # generate, clean, train
    python scripts/build_synthetic_ng.py --verify     # check reproducibility

Runs the same cleaning, feature engineering, splitting and training code the
empirical Olist environment uses. The only differences are the raw directory,
the holiday calendar and the geographic bounding box -- all supplied by the
environment object rather than branched on here.

Nothing this script writes goes near an Olist path.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olist import cleaning, environments, evaluate, features, pipeline  # noqa: E402
from olist.config import TARGET  # noqa: E402
from olist.synthetic_ng.generate import GeneratorConfig, generate, write_raw  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_synthetic_ng")


def build_tables(cfg: GeneratorConfig, env) -> pd.DataFrame:
    """Generate, clean and feature-engineer. Returns the modelling table."""
    tables = generate(cfg)
    write_raw(tables, env.raw_dir)

    cleaned, reports = cleaning.clean_all(tables)
    # Geolocation needs this environment's bounding box; every other cleaner
    # is geography-agnostic.
    cleaned["geolocation"], geo_report = cleaning.clean_geolocation(
        tables["geolocation"], bounds=env.geo_bounds, region=env.geo_region
    )
    for report in [*reports[:3], geo_report]:
        logger.info("  %s", report)

    df = features.build_modelling_table(cleaned, calendar=env.calendar)
    env.processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(env.processed_path, index=False)
    logger.info("Modelling table: %d rows x %d cols -> %s",
                len(df), df.shape[1], env.processed_path)
    return df


def train(df: pd.DataFrame, env, scenario: str = "post_dispatch") -> dict:
    """Fit, calibrate and evaluate, using the shared pipeline code."""
    train_df, test_df = pipeline.temporal_split(df, train_end="2024-06-30")
    cols = pipeline.feature_columns(df, scenario)

    base = pipeline.build_pipeline(train_df, cols, "logistic_regression",
                                   y=train_df[TARGET])
    model = pipeline.fit_calibrated(base, train_df, cols)

    proba = model.predict_proba(test_df[cols])[:, 1]
    cost = __import__("olist.config", fromlist=["CONFIG"]).CONFIG.cost
    result = evaluate.evaluate(
        test_df[TARGET].to_numpy(), proba, cost, scenario,
        threshold=evaluate.holdout_threshold(model, train_df, cols, cost))

    artefact = {
        "pipeline": model,
        "model_name": "logistic_regression",
        "scenario": scenario,
        "feature_columns": cols,
        "threshold": float(result["threshold"]["threshold"]),
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "metrics": {k: float(v) for k, v in result["metrics"].items()
                    if isinstance(v, (int, float, np.floating))},
        environments.ENVIRONMENT_KEY: env.key,
        "environment_label": env.label,
        "environment_kind": env.kind,
        "generator_seed": GeneratorConfig().seed,
    }

    env.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, env.model_path(scenario))

    # A background sample for SHAP, drawn from the training period only.
    background = train_df[cols].sample(min(200, len(train_df)), random_state=42)
    background.to_parquet(env.processed_path.parent / "background.parquet", index=False)

    return artefact


def fingerprint(df: pd.DataFrame) -> str:
    num = df.select_dtypes("number")
    num = num.reindex(sorted(num.columns), axis=1)
    return hashlib.sha256(
        np.ascontiguousarray(num.fillna(-999.5).to_numpy(dtype="float64"))
    ).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true",
                    help="regenerate twice and confirm the output is identical")
    args = ap.parse_args()

    env = environments.get("synthetic_ng")
    cfg = GeneratorConfig(n_orders=args.orders, seed=args.seed)

    logger.info("Building the %s environment", env.label)
    logger.info("  seed %d, %d orders", cfg.seed, cfg.n_orders)

    df = build_tables(cfg, env)

    if args.verify:
        # Must take the identical path, including the Nigerian geo bounds.
        # Comparing against a differently-cleaned build would report a
        # difference that is an artefact of the check, not of the generator.
        second = build_tables(cfg, env)
        same = fingerprint(df) == fingerprint(second)
        logger.info("Reproducibility check: %s", "IDENTICAL" if same else "DIFFERS")
        if not same:
            raise SystemExit("Generation is not reproducible at a fixed seed.")
        return

    artefact = train(df, env)

    logger.info("")
    logger.info("Trained %s model for %s", artefact["scenario"], env.label)
    logger.info("  threshold %.4f, %d features", artefact["threshold"],
                len(artefact["feature_columns"]))
    for metric in ("roc_auc", "average_precision", "brier", "recall_at_chosen"):
        if metric in artefact["metrics"]:
            logger.info("  %-20s %.4f", metric, artefact["metrics"][metric])
    logger.info("")
    logger.info("These are properties of a simulated environment, not")
    logger.info("measurements of Nigerian delivery performance.")


if __name__ == "__main__":
    main()
