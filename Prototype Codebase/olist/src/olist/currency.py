"""Currency presentation for the Olist case study.

The Olist dataset records real Brazilian transactions in Brazilian reais. This
module converts those values to Nigerian naira for presentation, at a sourced
historical rate, and keeps the original figure traceable.

What this module does **not** do is rename BRL to NGN. A converted value is a
Brazilian transaction expressed on a naira scale; it is not a Nigerian
transaction, and nothing here should be read as evidence about Nigerian prices,
Nigerian customers or the Nigerian market.

Rates
-----
No direct BRL/NGN series is published. The cross-rate is derived from two
national rates taken from the *same* source and the *same* series, so the two
legs are consistent with one another:

    International Monetary Fund, International Financial Statistics
    "Official exchange rate (LCU per US$, period average)"

    Year   NGN per USD    BRL per USD    -> NGN per BRL
    2016        253.49           3.49           72.63
    2017        305.79           3.19           95.86
    2018        306.08           3.65           83.86

Because the Olist window spans September 2016 to August 2018 and every order
carries a purchase timestamp, conversion is applied **per order at its own
purchase year** rather than at a single average. That is the more faithful
treatment: the naira moved by more than 20% across the window, so one blended
rate would misstate early and late orders in opposite directions.

A note on which Nigerian rate
-----------------------------
Nigeria operated multiple exchange rates through this period. The IMF official
series above averages roughly N306/USD for 2018. The Investors & Exporters
(NAFEX) window traded nearer N360/USD, which implies roughly N99/BRL rather
than N84/BRL -- about 19% higher.

Neither is wrong; they answer different questions. The official rate is what a
national-accounts comparison uses and is citable from a single authoritative
source. The NAFEX rate is closer to what an importer or consumer actually
faced. ``BASIS`` selects between them, and the NAFEX leg stays unpopulated
until a citable annual average is supplied, because an exchange rate repeated
from memory is a fabricated number however plausible it looks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Sourced rates
# --------------------------------------------------------------------------

SOURCE = (
    "International Monetary Fund, International Financial Statistics: "
    "official exchange rate (LCU per US$, period average)."
)

#: NGN per USD, annual period average, IMF IFS.
NGN_PER_USD_OFFICIAL = {2016: 253.49, 2017: 305.79, 2018: 306.08}

#: BRL per USD, annual period average, IMF IFS.
BRL_PER_USD = {2016: 3.49, 2017: 3.19, 2018: 3.65}

#: NGN per USD on the Investors & Exporters (NAFEX) window. Left empty on
#: purpose: populate only from a citable annual average, never an estimate.
NGN_PER_USD_NAFEX: dict[int, float] = {}

BASIS = "imf_official"


def cross_rates(basis: str = BASIS) -> dict[int, float]:
    """NGN per BRL, by year, derived from the two national rates."""
    if basis == "imf_official":
        ngn = NGN_PER_USD_OFFICIAL
    elif basis == "nafex":
        if not NGN_PER_USD_NAFEX:
            raise ValueError(
                "No NAFEX rates are recorded. Supply a citable annual average "
                "before using this basis; do not estimate one."
            )
        ngn = NGN_PER_USD_NAFEX
    else:
        raise ValueError(f"Unknown basis {basis!r}")
    return {y: round(ngn[y] / BRL_PER_USD[y], 4)
            for y in sorted(set(ngn) & set(BRL_PER_USD))}


NGN_PER_BRL = cross_rates()


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Converted:
    """A converted series, with everything needed to reproduce it."""

    ngn: np.ndarray
    brl: np.ndarray
    rate_per_row: np.ndarray
    effective_rate: float          #: value-weighted, i.e. total NGN / total BRL
    basis: str
    source: str
    years_used: dict[int, float]
    n_outside_rate_table: int


def to_ngn(values_brl, years, basis: str = BASIS,
           override_rate: float | None = None) -> Converted:
    """Convert BRL amounts to NGN, each at its own purchase year's rate.

    ``override_rate`` applies a single flat rate instead, for a house rate an
    institution has chosen. It still has to be sourced before publication.
    """
    brl = np.asarray(values_brl, dtype="float64")
    yr = np.asarray(pd.Series(years).astype("Int64").to_numpy(dtype="float64"))

    if override_rate is not None:
        rates = np.full_like(brl, float(override_rate))
        table, missing = {}, 0
    else:
        table = cross_rates(basis)
        default = table[max(table)]          # orders outside the table
        rates = np.array([table.get(int(y), default) if np.isfinite(y) else default
                          for y in yr])
        missing = int(sum(1 for y in yr
                          if not np.isfinite(y) or int(y) not in table))

    ngn = brl * rates
    total = brl.sum()
    return Converted(
        ngn=ngn,
        brl=brl,
        rate_per_row=rates,
        effective_rate=float(ngn.sum() / total) if total else float("nan"),
        basis="override" if override_rate is not None else basis,
        source="Supplied by the authors" if override_rate is not None else SOURCE,
        years_used=table,
        n_outside_rate_table=missing,
    )


def fmt(amount: float, currency: str = "NGN") -> str:
    symbol = {"NGN": "₦", "BRL": "R$"}.get(currency, currency + " ")
    return f"{symbol}{amount:,.0f}"


# --------------------------------------------------------------------------
# Manuscript wording
# --------------------------------------------------------------------------

DISCLOSURE = (
    "Original Olist monetary values are denominated in Brazilian reais (BRL). "
    "For contextual presentation they are reported here in Nigerian naira "
    "(NGN), converted per order at the official annual average exchange rate "
    "for the order's purchase year, derived from IMF International Financial "
    "Statistics official rates for both currencies "
    "({table}). The conversion provides a common naira representation of "
    "monetary scale and does not imply that the underlying Olist transactions "
    "occurred in Nigeria or represent Nigerian market prices. Cost parameters "
    "(refund share, churn share, support cost, intervention cost) are "
    "stipulated assumptions and are stated directly in naira."
)


def disclosure(basis: str = BASIS) -> str:
    table = "; ".join(f"{y}: ₦{r:,.2f}/BRL" for y, r in cross_rates(basis).items())
    return DISCLOSURE.format(table=table)
