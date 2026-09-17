"""Raw-file loading.

Every reader declares its dtypes explicitly rather than letting pandas infer
them. Inference on this dataset silently turns zip-code prefixes into ints
(dropping leading zeros) and reads the category-translation file with a UTF-8
BOM welded onto the first column name, which makes the downstream merge
produce all-NaN without raising.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, RAW_FILES

logger = logging.getLogger(__name__)

#: Timestamp columns per table, parsed on load.
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}

#: Zip prefixes are categorical identifiers, not quantities. Read as string to
#: preserve leading zeros, which matter for the São Paulo metro area (01xxx).
DTYPES = {
    "customers": {"customer_zip_code_prefix": "string"},
    "sellers": {"seller_zip_code_prefix": "string"},
    "geolocation": {"geolocation_zip_code_prefix": "string"},
}


def raw_path(name: str, raw_dir: Path | None = None) -> Path:
    """Resolve a logical table name to its file on disk."""
    if name not in RAW_FILES:
        raise KeyError(f"Unknown table {name!r}. Expected one of {sorted(RAW_FILES)}")
    return (raw_dir or RAW_DIR) / RAW_FILES[name]


def load_table(name: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Load one raw table with explicit dtypes and parsed dates.

    ``encoding="utf-8-sig"`` strips the byte-order mark present in
    ``product_category_name_translation.csv``. Applied to every file because
    it is a no-op when no BOM is present.
    """
    path = raw_path(name, raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw file: {path}\n"
            f"Download the Brazilian E-Commerce Public Dataset by Olist from "
            f"Kaggle and place the CSVs in {path.parent}. See data/README.md."
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=DTYPES.get(name),
        parse_dates=DATE_COLUMNS.get(name),
    )
    logger.info("Loaded %-22s %7d rows x %2d cols", name, len(df), df.shape[1])
    return df


def load_all(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all nine raw tables into a name-keyed dict."""
    return {name: load_table(name, raw_dir) for name in RAW_FILES}


def available_tables(raw_dir: Path | None = None) -> list[str]:
    """Names of raw tables actually present on disk. Useful for partial runs."""
    return [n for n in RAW_FILES if raw_path(n, raw_dir).exists()]


# --------------------------------------------------------------------------
# Processed-dataset cache
# --------------------------------------------------------------------------

def write_dataset(df: pd.DataFrame, path: Path) -> Path:
    """Write the processed table, falling back to CSV without pyarrow.

    Parquet preserves dtypes and is far smaller, but it needs an optional
    engine. Crashing at the end of a long merge because that engine is
    absent wastes the whole run, so fall back rather than fail.
    """
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        logger.warning(
            "pyarrow not installed; wrote CSV to %s instead of parquet. "
            "Install pyarrow to preserve dtypes: pip install pyarrow",
            fallback,
        )
        return fallback


def read_dataset(path: Path) -> pd.DataFrame | None:
    """Read the processed table from parquet or its CSV fallback."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except ImportError:
            pass
    fallback = path.with_suffix(".csv")
    if fallback.exists():
        return pd.read_csv(fallback, parse_dates=DATE_COLUMNS["orders"])
    return None
