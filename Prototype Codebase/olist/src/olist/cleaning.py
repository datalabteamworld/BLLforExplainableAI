"""Per-table cleaning.

Each function takes a raw table and returns a cleaned copy plus a record of
what it changed. Cleaning is deliberately *non-destructive*: rows are flagged
or set to NaN rather than dropped, so that downstream code decides what to do
with them and the row count stays reconcilable against the raw file.

Every rule here corresponds to a defect verified in the raw Kaggle release;
see ``tests/test_cleaning.py`` for the assertions that pin them.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================================
# Reporting
# ==========================================================================


@dataclass
class CleaningReport:
    """Accumulates what each cleaning step changed, for audit and tests."""

    table: str
    rows_in: int
    rows_out: int = 0
    actions: dict[str, int] = field(default_factory=dict)

    def record(self, action: str, n: int) -> None:
        if n:
            self.actions[action] = self.actions.get(action, 0) + int(n)

    def __str__(self) -> str:
        head = f"{self.table}: {self.rows_in:,} -> {self.rows_out:,} rows"
        if not self.actions:
            return head + "  (no changes)"
        body = "\n".join(f"    {k:.<52} {v:,}" for k, v in self.actions.items())
        return f"{head}\n{body}"


# ==========================================================================
# Shared text helpers
# ==========================================================================

#: Separators that indicate a "city, state" or "city/state" compound string.
_CITY_SPLIT = re.compile(r"\s*[/,\\]\s*|\s+-\s+")

#: Two-letter Brazilian state codes, used to detect a state masquerading as a
#: city and to strip a trailing state suffix.
_STATE_CODES = {
    "ac", "al", "am", "ap", "ba", "ce", "df", "es", "go", "ma", "mg", "ms",
    "mt", "pa", "pb", "pe", "pi", "pr", "rj", "rn", "ro", "rr", "rs", "sc",
    "se", "sp", "to",
}

_STATE_NAMES = {
    "sao paulo", "rio de janeiro", "minas gerais", "espirito santo",
    "rio grande do sul", "rio grande do norte", "santa catarina", "parana",
    "bahia", "goias", "mato grosso", "mato grosso do sul", "distrito federal",
    "pernambuco", "ceara", "para", "paraiba", "piaui", "alagoas", "sergipe",
    "amazonas", "acre", "rondonia", "roraima", "amapa", "tocantins",
    "maranhao", "brasil",
}

#: Hand-verified corrections for strings the generic rules cannot recover.
_CITY_ALIASES = {
    "sao paulop": "sao paulo",
    "sbc": "sao bernardo do campo",
    "s jose do rio preto": "sao jose do rio preto",
    "sp": np.nan,          # a state code, not a city
    "rio de janeiro rio de janeiro": "rio de janeiro",
}


def strip_accents(text: str) -> str:
    """Fold accented characters to ASCII: ``são`` -> ``sao``.

    Also normalises the acute accent used as an apostrophe in
    ``santa barbara d´oeste``.
    """
    text = text.replace("´", "'").replace("`", "'")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise_city(value: object) -> object:
    """Reduce a free-text Brazilian city string to a canonical form.

    Handles, in order: accent folding, case and whitespace, compound
    ``city/state`` strings, trailing state suffixes, numeric junk, and a
    small alias table.

    >>> normalise_city("São Paulo / SAO PAULO")
    'sao paulo'
    >>> normalise_city("rio de janeiro \\\\rio de janeiro")
    'rio de janeiro'
    >>> normalise_city("04482255") is np.nan
    True
    """
    if not isinstance(value, str):
        return np.nan

    text = strip_accents(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return np.nan

    # A bare number is a zip code recorded in the wrong field.
    if re.fullmatch(r"[\d\W_]+", text):
        return np.nan

    # Split compounds such as "santo andre/sao paulo" and keep the first
    # segment. Only two-letter state codes are filtered outright: several
    # state *names* ("sao paulo", "rio de janeiro") are also legitimate city
    # names, so dropping them by name would discard the city itself.
    parts = [p.strip() for p in _CITY_SPLIT.split(text) if p.strip()]
    parts = [p for p in parts if p not in _STATE_CODES]
    if not parts:
        return np.nan
    text = parts[0]

    # Strip a trailing state suffix left over from "sao paulo sp" or
    # "brasilia df", plus any residual separator characters.
    text = text.strip(" -/\\,")
    tokens = text.split()
    while len(tokens) > 1 and tokens[-1] in _STATE_CODES:
        tokens.pop()
    text = " ".join(tokens)

    # A repeated "city city" string collapses to one occurrence.
    half = len(tokens) // 2
    if half and tokens[:half] == tokens[half:]:
        text = " ".join(tokens[:half])
        tokens = tokens[:half]

    # A bare state name with nothing else is not a usable city value.
    if text in _STATE_CODES:
        return np.nan

    text = _CITY_ALIASES.get(text, text)
    if not isinstance(text, str) or not text.strip():
        return np.nan
    return text


# ==========================================================================
# Table cleaners
# ==========================================================================


def clean_payments(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean ``olist_order_payments_dataset``.

    Defects addressed:
      * ``payment_type == "not_defined"`` (3 rows, all zero-value)
      * ``payment_value == 0`` (9 rows)
      * ``payment_installments == 0`` (2 rows) -- a single payment is 1
    """
    rep = CleaningReport("order_payments", len(df))
    out = df.copy()

    mask = out["payment_type"].eq("not_defined")
    rep.record("payment_type 'not_defined' -> NaN", mask.sum())
    out.loc[mask, "payment_type"] = np.nan

    zero_val = out["payment_value"].eq(0)
    rep.record("zero payment_value flagged", zero_val.sum())
    out["is_zero_payment"] = zero_val.astype("int8")

    bad_inst = out["payment_installments"].le(0)
    rep.record("payment_installments <= 0 -> 1", bad_inst.sum())
    out.loc[bad_inst, "payment_installments"] = 1

    neg = out["payment_value"].lt(0)
    rep.record("negative payment_value -> NaN", neg.sum())
    out.loc[neg, "payment_value"] = np.nan

    rep.rows_out = len(out)
    return out, rep


def clean_products(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean ``olist_products_dataset``.

    Defects addressed:
      * source column names misspell "length" as "lenght"
      * 610 products with no category or listing metadata
      * 4 products with ``product_weight_g == 0`` (physically impossible)
      * 2 products with no weight or dimensions at all
    """
    rep = CleaningReport("products", len(df))
    out = df.rename(
        columns={
            "product_name_lenght": "product_name_length",
            "product_description_lenght": "product_description_length",
        }
    ).copy()
    rep.record("renamed misspelled 'lenght' columns", 2)

    dims = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    for col in dims:
        zero = out[col].eq(0)
        rep.record(f"{col} == 0 -> NaN", zero.sum())
        out.loc[zero, col] = np.nan

    missing_cat = out["product_category_name"].isna()
    rep.record("missing category -> 'unlisted'", missing_cat.sum())
    out["product_category_name"] = out["product_category_name"].fillna("unlisted")
    out["has_listing_metadata"] = (~missing_cat).astype("int8")

    out["product_volume_cm3"] = (
        out["product_length_cm"] * out["product_height_cm"] * out["product_width_cm"]
    )
    #: Density separates a heavy compact parcel from a light bulky one; both
    #: are handled differently by carriers and both affect delivery risk.
    out["product_density_g_cm3"] = out["product_weight_g"] / out["product_volume_cm3"]

    rep.rows_out = len(out)
    return out, rep


def clean_sellers(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean ``olist_sellers_dataset``.

    The only meaningful defect is free-text ``seller_city``: 611 distinct
    strings that collapse substantially once accents, compound
    ``city/state`` forms, trailing state codes and typos are normalised.
    """
    rep = CleaningReport("sellers", len(df))
    out = df.copy()

    before = out["seller_city"].nunique()
    out["seller_city_raw"] = out["seller_city"]
    out["seller_city"] = out["seller_city"].map(normalise_city)
    after = out["seller_city"].nunique()

    rep.record("distinct seller_city values merged", before - after)
    rep.record("seller_city set to NaN (unrecoverable)", out["seller_city"].isna().sum())
    rep.record("seller_city strings changed", (out["seller_city"] != out["seller_city_raw"]).sum())

    out["seller_state"] = out["seller_state"].str.upper().str.strip()
    rep.rows_out = len(out)
    return out, rep


def clean_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean ``olist_customers_dataset`` -- same city normalisation as sellers."""
    rep = CleaningReport("customers", len(df))
    out = df.copy()

    before = out["customer_city"].nunique()
    out["customer_city"] = out["customer_city"].map(normalise_city)
    rep.record("distinct customer_city values merged", before - out["customer_city"].nunique())

    out["customer_state"] = out["customer_state"].str.upper().str.strip()
    rep.rows_out = len(out)
    return out, rep


#: Categories present in ``products`` but absent from the official
#: translation file. Verified against the raw release.
MISSING_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "kitchen_portables_and_food_preparators",
    "unlisted": "unlisted",
}


def clean_translation(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean ``product_category_name_translation``.

    The BOM is stripped at load time by ``io.load_table``. The substantive
    defect is coverage: two categories used by ``products`` have no row here,
    which silently produces NaN English names after the merge.
    """
    rep = CleaningReport("category_translation", len(df))
    out = df.copy()

    if out.columns[0].startswith("\ufeff"):  # defensive; io strips this
        out.columns = [c.lstrip("\ufeff") for c in out.columns]
        rep.record("stripped UTF-8 BOM from header", 1)

    known = set(out["product_category_name"])
    additions = [
        {"product_category_name": pt, "product_category_name_english": en}
        for pt, en in MISSING_TRANSLATIONS.items()
        if pt not in known
    ]
    if additions:
        out = pd.concat([out, pd.DataFrame(additions)], ignore_index=True)
        rep.record("added missing category translations", len(additions))

    out = out.drop_duplicates(subset=["product_category_name"])
    rep.rows_out = len(out)
    return out, rep


def clean_geolocation(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Aggregate ``olist_geolocation_dataset`` to one row per zip prefix.

    The raw table has ~1M rows because each zip prefix appears once per
    recorded address. Coordinates outside Brazil's bounding box are dropped
    before averaging so that a handful of bad fixes cannot drag a zip
    centroid into the Atlantic.
    """
    rep = CleaningReport("geolocation", len(df))
    out = df.copy()

    # Continental Brazil plus offshore territories.
    in_bounds = (
        out["geolocation_lat"].between(-34.0, 5.3)
        & out["geolocation_lng"].between(-74.0, -34.0)
    )
    rep.record("coordinates outside Brazil dropped", (~in_bounds).sum())
    out = out[in_bounds]

    agg = (
        out.groupby("geolocation_zip_code_prefix")
        .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"))
        .reset_index()
    )
    rep.record("rows collapsed to zip-prefix centroids", len(out) - len(agg))
    rep.rows_out = len(agg)
    return agg, rep


def clean_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean ``olist_order_reviews_dataset``: one review per order, latest wins."""
    rep = CleaningReport("order_reviews", len(df))
    out = df.copy()

    invalid = ~out["review_score"].between(1, 5)
    rep.record("review_score outside 1-5 -> NaN", invalid.sum())
    out.loc[invalid, "review_score"] = np.nan

    n_before = len(out)
    out = out.sort_values("review_answer_timestamp", ascending=False).drop_duplicates(
        subset=["order_id"], keep="first"
    )
    rep.record("duplicate reviews per order removed", n_before - len(out))

    out["has_review_comment"] = out["review_comment_message"].notna().astype("int8")
    rep.rows_out = len(out)
    return out, rep


def clean_orders(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean ``olist_orders_dataset`` -- timestamp sanity only."""
    rep = CleaningReport("orders", len(df))
    out = df.copy()

    impossible = (
        out["order_delivered_customer_date"].notna()
        & (out["order_delivered_customer_date"] < out["order_purchase_timestamp"])
    )
    rep.record("delivered before purchase -> NaN", impossible.sum())
    out.loc[impossible, "order_delivered_customer_date"] = pd.NaT

    early_carrier = (
        out["order_delivered_carrier_date"].notna()
        & (out["order_delivered_carrier_date"] < out["order_purchase_timestamp"])
    )
    rep.record("carrier pickup before purchase -> NaN", early_carrier.sum())
    out.loc[early_carrier, "order_delivered_carrier_date"] = pd.NaT

    rep.rows_out = len(out)
    return out, rep


def clean_order_items(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean ``olist_order_items_dataset`` -- non-negative money only."""
    rep = CleaningReport("order_items", len(df))
    out = df.copy()

    for col in ("price", "freight_value"):
        neg = out[col].lt(0)
        rep.record(f"negative {col} -> NaN", neg.sum())
        out.loc[neg, col] = np.nan

    rep.rows_out = len(out)
    return out, rep


CLEANERS = {
    "orders": clean_orders,
    "order_items": clean_order_items,
    "order_payments": clean_payments,
    "order_reviews": clean_reviews,
    "products": clean_products,
    "sellers": clean_sellers,
    "customers": clean_customers,
    "geolocation": clean_geolocation,
    "category_translation": clean_translation,
}


def clean_all(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[CleaningReport]]:
    """Apply the matching cleaner to every table present in ``tables``."""
    cleaned, reports = {}, []
    for name, df in tables.items():
        cleaner = CLEANERS.get(name)
        if cleaner is None:
            cleaned[name] = df
            continue
        cleaned[name], report = cleaner(df)
        reports.append(report)
        logger.info("%s", report)
    return cleaned, reports
