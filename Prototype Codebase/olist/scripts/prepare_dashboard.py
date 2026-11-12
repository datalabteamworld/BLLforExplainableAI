"""Train the paper's headline model and prepare the dashboard to match it.

    python scripts/prepare_dashboard.py

Writes, under the empirical Olist environment:

    models/model_post_dispatch.joblib      the calibrated model, with its
                                           measured metrics recorded
    data/processed/background.parquet      SHAP reference sample

Why this exists. The dashboard loads whatever model artefact it can find, and
the repository ships a small demo bundle in ``app/`` built from test fixtures,
whose metrics dictionary holds a hand-written AUC. A screenshot taken against
that bundle would display numbers the paper's results table does not contain.
This script fits the exact configuration §4 describes -- logistic regression,
post-dispatch, sigmoid calibration on a date-ordered holdout -- and stores the
metrics the model actually achieved, so the interface and the manuscript agree.

It also moves the demo bundle aside, so it cannot be picked up by the fallback
search.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olist import environments, evaluate as ev, pipeline  # noqa: E402
from olist.config import CONFIG, PROCESSED_DIR, RANDOM_STATE, TARGET  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("prepare")

MODEL, SCENARIO = "logistic_regression", "post_dispatch"


def main() -> None:
    path = PROCESSED_DIR / "orders_features.parquet"
    if not path.exists():
        raise SystemExit(
            f"{path} not found.\n"
            "Run: python scripts/collect_paper_evidence.py --rebuild"
        )

    env = environments.get("olist")
    df = pd.read_parquet(path)
    train, test = pipeline.temporal_split(df)
    cols = pipeline.feature_columns(df, SCENARIO)

    logger.info("Fitting %s / %s on %d orders, %d features",
                MODEL, SCENARIO, len(train), len(cols))
    base = pipeline.build_pipeline(train, cols, MODEL, y=train[TARGET])
    model = pipeline.fit_calibrated(base, train, cols)

    proba = model.predict_proba(test[cols])[:, 1]
    result = ev.evaluate(test[TARGET].to_numpy(), proba, CONFIG.cost, SCENARIO,
                         threshold=ev.holdout_threshold(model, train, cols, CONFIG.cost))
    metrics = {k: float(v) for k, v in result["metrics"].items()
               if isinstance(v, (int, float, np.floating))}

    artefact = {
        "pipeline": model,
        "model_name": MODEL,
        "scenario": SCENARIO,
        "feature_columns": cols,
        "threshold": float(result["threshold"]["threshold"]),
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "metrics": metrics,
        "n_train": len(train),
        "n_test": len(test),
        environments.ENVIRONMENT_KEY: env.key,
        "environment_label": env.label,
        "environment_kind": env.kind,
    }

    env.models_dir.mkdir(parents=True, exist_ok=True)
    out = env.model_path(SCENARIO)
    joblib.dump(artefact, out)
    logger.info("Wrote %s", out)

    # SHAP reference sample, drawn from the training period only.
    background = train[cols].sample(min(200, len(train)), random_state=RANDOM_STATE)
    bg_path = PROCESSED_DIR / "background.parquet"
    background.to_parquet(bg_path, index=False)
    logger.info("Wrote %s (%d rows)", bg_path, len(background))

    # Move the fixture-built demo bundle out of the fallback search path.
    for name in ("model_post_dispatch.joblib", "background.parquet",
                 "sample_orders.parquet"):
        demo = ROOT / "app" / name
        if demo.exists():
            shutil.move(str(demo), str(demo.with_suffix(demo.suffix + ".demo")))
            logger.info("Set aside %s", demo.name)

    logger.info("")
    logger.info("Measured on the held-out test set (%d orders):", len(test))
    for k in ("roc_auc", "average_precision", "brier", "recall_at_chosen",
              "precision_at_chosen"):
        if k in metrics:
            logger.info("  %-22s %.4f", k, metrics[k])
    logger.info("  %-22s %.4f", "threshold", artefact["threshold"])
    logger.info("")
    logger.info("These should match Table 3 of the paper. If they do not, stop")
    logger.info("and say so rather than screenshotting.")


if __name__ == "__main__":
    main()
