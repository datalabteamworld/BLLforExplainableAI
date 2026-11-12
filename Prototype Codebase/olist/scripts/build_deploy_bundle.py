"""Build a self-contained bundle for Streamlit Community Cloud.

Community Cloud deploys from a GitHub repository and has no access to the raw
Olist CSVs, which are 120 MB and not committed. It also caps memory at roughly
1 GB, so the full 96,171-order feature table is too large to ship.

This script writes a small bundle into ``deploy/``:

    model_post_dispatch.joblib   the trained artefact
    sample_orders.parquet        a stratified sample of scored orders
    background.parquet           the SHAP background sample

The sample is **stratified by risk decile**, not taken from the head of the
table. A head slice would be chronologically ordered and would under-represent
high-risk orders, so the deployed demo would show a narrower range of
behaviour than the model actually produces.

Run from the project root::

    python scripts/build_deploy_bundle.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olist.config import MODELS_DIR, PROCESSED_DIR  # noqa: E402


def build(n_orders: int = 2000, n_background: int = 150) -> Path:
    deploy = ROOT / "deploy"
    deploy.mkdir(exist_ok=True)

    # ---- model -----------------------------------------------------------
    model_path = MODELS_DIR / "model_post_dispatch.joblib"
    if not model_path.exists():
        raise SystemExit(
            f"No trained model at {model_path}.\n"
            f"Run: python -m olist.train --scenario post_dispatch"
        )
    artefact = joblib.load(model_path)
    shutil.copy(model_path, deploy / "model_post_dispatch.joblib")

    cols = artefact["feature_columns"]
    model = artefact["pipeline"]

    # ---- orders ----------------------------------------------------------
    features_path = PROCESSED_DIR / "orders_features.parquet"
    if not features_path.exists():
        raise SystemExit(
            f"No feature table at {features_path}.\n"
            f"Run: python -m olist.validate --build"
        )
    df = pd.read_parquet(features_path)

    # Score everything once so the sample can be stratified by risk.
    risk = model.predict_proba(df[cols])[:, 1]

    # Stratify by decile: the deployed demo should show the model's full
    # range, including the rare high-risk orders that make the decision
    # brief interesting. A head slice would be mostly 2016 low-risk orders.
    deciles = pd.qcut(risk, 10, labels=False, duplicates="drop")
    per_bin = max(1, n_orders // len(np.unique(deciles)))

    rng = np.random.default_rng(42)
    picked: list[int] = []
    for d in np.unique(deciles):
        idx = np.flatnonzero(deciles == d)
        take = min(per_bin, len(idx))
        picked.extend(rng.choice(idx, take, replace=False))

    sample = df.iloc[sorted(picked)].reset_index(drop=True)

    # Keep only what the cloud app reads: the model's features plus the
    # context columns the brief displays. Dropping the rest roughly halves
    # the file.
    context = [c for c in (
        "order_id", "customer_state", "customer_city", "seller_state",
        "product_category", "total_price", "order_purchase_timestamp",
        "order_estimated_delivery_date", "is_late_delivery",
        "delivery_delay_days",
    ) if c in sample.columns]

    keep = list(dict.fromkeys(context + cols))
    sample[keep].to_parquet(deploy / "sample_orders.parquet", index=False)

    # ---- SHAP background -------------------------------------------------
    # Drawn from the training period only, so the explainer's reference
    # distribution never includes orders the model was tested on.
    train_cut = pd.Timestamp("2018-04-30")
    train = df[df["order_purchase_timestamp"] <= train_cut]
    background = train[cols].sample(
        min(n_background, len(train)), random_state=42
    ).reset_index(drop=True)
    background.to_parquet(deploy / "background.parquet", index=False)

    # ---- report ----------------------------------------------------------
    total = sum(f.stat().st_size for f in deploy.glob("*")) / 1e6
    print(f"Bundle written to {deploy}")
    for f in sorted(deploy.glob("*")):
        print(f"  {f.name:34s} {f.stat().st_size / 1e6:6.2f} MB")
    print(f"  {'total':34s} {total:6.2f} MB")
    print()
    print(f"Sampled {len(sample):,} orders from {len(df):,}, stratified by risk decile.")
    print(f"Risk range in sample: {risk[sorted(picked)].min():.1%} to "
          f"{risk[sorted(picked)].max():.1%}")
    print(f"SHAP background: {len(background)} rows from the training period.")

    if total > 50:
        print("\nWARNING: bundle exceeds 50 MB. GitHub warns above 50 MB per file "
              "and Community Cloud will be slow to start.")
    return deploy


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", type=int, default=2000,
                    help="orders to include in the sample (default 2000)")
    ap.add_argument("--background", type=int, default=150,
                    help="rows in the SHAP background (default 150)")
    args = ap.parse_args()
    build(args.orders, args.background)
