"""Preflight validation of the raw data directory.

Run this **before** training::

    python -m olist.validate

It checks that all nine files are present, that each has the columns the
pipeline expects, that the join keys are unique where the merge assumes they
are, and that row counts are in the expected range. It then prints the
cleaning audit so you can see exactly what will be changed before anything
is modelled.

The point is to fail in ten seconds with a clear message rather than thirty
minutes into a training run with a ``KeyError``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import cleaning, features, io
from .config import PROCESSED_DIR, RAW_DIR, RAW_FILES

logger = logging.getLogger(__name__)

#: Columns the pipeline reads from each table. Extra columns are fine;
#: missing ones are fatal.
REQUIRED_COLUMNS = {
    "orders": [
        "order_id", "customer_id", "order_status",
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "customers": [
        "customer_id", "customer_unique_id", "customer_zip_code_prefix",
        "customer_city", "customer_state",
    ],
    "order_items": [
        "order_id", "order_item_id", "product_id", "seller_id",
        "shipping_limit_date", "price", "freight_value",
    ],
    "order_payments": [
        "order_id", "payment_sequential", "payment_type",
        "payment_installments", "payment_value",
    ],
    "order_reviews": [
        "review_id", "order_id", "review_score",
        "review_comment_message", "review_creation_date",
        "review_answer_timestamp",
    ],
    "products": [
        "product_id", "product_category_name", "product_name_lenght",
        "product_description_lenght", "product_photos_qty",
        "product_weight_g", "product_length_cm", "product_height_cm",
        "product_width_cm",
    ],
    "sellers": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    "geolocation": [
        "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
    ],
    "category_translation": ["product_category_name", "product_category_name_english"],
}

#: Keys that must be unique, because the merge chain joins on them without
#: aggregating first. A duplicate here silently multiplies rows.
UNIQUE_KEYS = {
    "orders": "order_id",
    "customers": "customer_id",
    "products": "product_id",
    "sellers": "seller_id",
    "category_translation": "product_category_name",
}

#: Approximate row counts in the public release. A large deviation usually
#: means a partial download or a file opened and re-saved by Excel.
EXPECTED_ROWS = {
    "orders": 99_441,
    "customers": 99_441,
    "order_items": 112_650,
    "order_payments": 103_886,
    "order_reviews": 99_224,
    "products": 32_951,
    "sellers": 3_095,
    "geolocation": 1_000_163,
    "category_translation": 71,
}

TOLERANCE = 0.02


class ValidationError(Exception):
    """Raised when the raw data cannot support the pipeline."""


def check_files_present(raw_dir: Path) -> list[str]:
    """Report which of the nine expected files are missing."""
    missing = [name for name in RAW_FILES if not io.raw_path(name, raw_dir).exists()]
    for name in RAW_FILES:
        status = "MISSING" if name in missing else "found"
        print(f"  {RAW_FILES[name]:<45} {status}")
    return missing


def check_table(name: str, df: pd.DataFrame) -> list[str]:
    """Schema, uniqueness and row-count checks for one table."""
    problems = []

    missing_cols = [c for c in REQUIRED_COLUMNS.get(name, []) if c not in df.columns]
    if missing_cols:
        problems.append(f"{name}: missing columns {missing_cols}")

    key = UNIQUE_KEYS.get(name)
    if key and key in df.columns:
        n_dupes = int(df[key].duplicated().sum())
        if n_dupes:
            problems.append(
                f"{name}: {n_dupes:,} duplicate {key} values. The merge chain "
                f"joins on this key without aggregating and will fan out."
            )

    expected = EXPECTED_ROWS.get(name)
    if expected and abs(len(df) - expected) / expected > TOLERANCE:
        problems.append(
            f"{name}: {len(df):,} rows, expected about {expected:,}. "
            f"Check for a truncated download or a file re-saved by Excel."
        )
    return problems


def check_referential_integrity(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Warn where foreign keys do not resolve.

    These are warnings, not errors: the merge uses left joins, so unresolved
    keys become NaN rather than lost rows. But a high rate means something is
    wrong with the download.
    """
    warnings = []

    def _orphan_rate(child: str, parent: str, key: str) -> float | None:
        if child not in tables or parent not in tables:
            return None
        if key not in tables[child].columns or key not in tables[parent].columns:
            return None
        known = set(tables[parent][key])
        return float((~tables[child][key].isin(known)).mean())

    for child, parent, key in [
        ("orders", "customers", "customer_id"),
        ("order_items", "orders", "order_id"),
        ("order_items", "products", "product_id"),
        ("order_items", "sellers", "seller_id"),
        ("order_payments", "orders", "order_id"),
        ("order_reviews", "orders", "order_id"),
    ]:
        rate = _orphan_rate(child, parent, key)
        if rate and rate > 0.01:
            warnings.append(
                f"{child}.{key}: {rate:.1%} of values do not appear in {parent}"
            )
    return warnings


def validate(raw_dir: Path | None = None, strict: bool = True) -> dict[str, pd.DataFrame]:
    """Run every check and return the loaded raw tables."""
    raw_dir = raw_dir or RAW_DIR

    print(f"\nRaw data directory: {raw_dir}\n")
    missing = check_files_present(raw_dir)
    if missing:
        raise ValidationError(
            "\nMissing files: "
            + ", ".join(RAW_FILES[m] for m in missing)
            + "\n\nDownload the Brazilian E-Commerce Public Dataset by Olist from"
            "\nhttps://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
            f"\nand unzip all nine CSVs into {raw_dir}."
        )

    print("\nLoading and checking schemas...")
    tables = io.load_all(raw_dir)

    problems: list[str] = []
    for name, df in tables.items():
        problems.extend(check_table(name, df))

    warnings = check_referential_integrity(tables)

    if problems:
        print("\nProblems found:")
        for p in problems:
            print(f"  - {p}")
        if strict:
            raise ValidationError(f"{len(problems)} problem(s) must be fixed first.")
    else:
        print("\nAll schema and uniqueness checks passed.")

    if warnings:
        print("\nWarnings (left joins will produce NaN for these):")
        for w in warnings:
            print(f"  - {w}")

    return tables


def build_dataset(
    raw_dir: Path | None = None,
    output: Path | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Validate, clean, merge and write the modelling table.

    This is the clean-and-merge step on its own, without any modelling, so
    the intermediate output can be inspected before training.
    """
    tables = validate(raw_dir, strict=strict)

    print("\nCleaning...\n")
    cleaned, reports = cleaning.clean_all(tables)
    for report in reports:
        print(report)

    print("\nMerging to order grain...")
    model_df = features.build_modelling_table(cleaned)

    output = output or (PROCESSED_DIR / "orders_features.parquet")
    output = io.write_dataset(model_df, output)

    print(
        f"\nWrote {len(model_df):,} orders x {model_df.shape[1]} columns to {output}"
        f"\nLate-delivery rate: {model_df['is_late_delivery'].mean():.2%}"
        f"\nDate range: {model_df['order_purchase_timestamp'].min().date()}"
        f" to {model_df['order_purchase_timestamp'].max().date()}"
        f"\n\nNext: python -m olist.train --scenario pre_dispatch --tune"
    )
    return model_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and build the Olist dataset.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--build", action="store_true",
                        help="also clean, merge and write the modelling table")
    parser.add_argument("--no-strict", action="store_true",
                        help="report problems without stopping")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    try:
        if args.build:
            build_dataset(args.raw_dir, strict=not args.no_strict)
        else:
            validate(args.raw_dir, strict=not args.no_strict)
            print("\nRun with --build to clean and merge.")
    except ValidationError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
