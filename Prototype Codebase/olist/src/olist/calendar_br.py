"""Brazilian holiday and commercial-event calendar.

Two things this module does that a naive holiday join does not:

1. It builds a *daily* calendar spanning the dataset, not a sparse list of
   event dates. That lets features be computed over an interval rather than
   at a single instant.
2. It exposes :func:`count_events_in_window`, which counts non-working days
   between purchase and estimated delivery. Delivery lateness is caused by
   holidays falling inside the *transit window*, not by a holiday on the
   purchase date -- an order placed the day before Carnival is at risk
   because of the five days that follow it, and a purchase-date flag cannot
   express that.
"""

from __future__ import annotations

import functools
from datetime import date

import numpy as np
import pandas as pd
from workalendar.america import Brazil


@functools.lru_cache(maxsize=None)
def _calendar() -> Brazil:
    return Brazil()


def national_holidays(year: int) -> dict[date, str]:
    """Fixed and movable national holidays for one year."""
    return {d: name for d, name in _calendar().holidays(year)}


def black_friday(year: int) -> pd.Timestamp:
    """Fourth Friday of November."""
    nov1 = pd.Timestamp(year, 11, 1)
    first_friday = nov1 + pd.Timedelta(days=(4 - nov1.dayofweek) % 7)
    return first_friday + pd.Timedelta(weeks=3)


def cyber_monday(year: int) -> pd.Timestamp:
    return black_friday(year) + pd.Timedelta(days=3)


def carnival_days(year: int) -> list[pd.Timestamp]:
    """The five-day Carnival period ending on Ash Wednesday.

    Ash Wednesday is Easter Sunday minus 46 days; Carnival runs from the
    preceding Saturday through Ash Wednesday itself.
    """
    easter = next(d for d, name in _calendar().holidays(year) if name == "Easter Sunday")
    ash_wednesday = pd.Timestamp(easter) - pd.Timedelta(days=46)
    start = ash_wednesday - pd.Timedelta(days=4)
    return [start + pd.Timedelta(days=i) for i in range(5)]


def build_daily_calendar(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """One row per calendar day between ``start`` and ``end`` inclusive.

    Returns columns: ``date``, ``holiday_name``, ``is_holiday``,
    ``is_black_friday``, ``is_cyber_monday``, ``is_carnival``,
    ``is_weekend``, ``is_working_day``.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    days = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame({"date": days})

    years = range(start.year, end.year + 1)

    holiday_map: dict[pd.Timestamp, str] = {}
    for year in years:
        for d, name in national_holidays(year).items():
            holiday_map[pd.Timestamp(d)] = name

    bf = {black_friday(y) for y in years}
    cm = {cyber_monday(y) for y in years}
    carnival = {d for y in years for d in carnival_days(y)}

    cal["holiday_name"] = cal["date"].map(holiday_map).fillna("")
    cal["is_holiday"] = (cal["holiday_name"] != "").astype("int8")
    cal["is_black_friday"] = cal["date"].isin(bf).astype("int8")
    cal["is_cyber_monday"] = cal["date"].isin(cm).astype("int8")
    cal["is_carnival"] = cal["date"].isin(carnival).astype("int8")
    cal["is_weekend"] = cal["date"].dt.dayofweek.isin([5, 6]).astype("int8")

    # Carnival is not a statutory national holiday but the logistics network
    # effectively stops, so it counts as non-working for our purposes.
    cal["is_working_day"] = (
        (cal["is_holiday"] == 0) & (cal["is_weekend"] == 0) & (cal["is_carnival"] == 0)
    ).astype("int8")
    return cal


def days_to_nearest_black_friday(dates: pd.Series, years: range | list[int]) -> pd.Series:
    """Signed day offset to the closest Black Friday. Negative = before.

    Uses pandas timedelta arithmetic rather than integer views: pandas 2.x
    may hold datetimes at microsecond resolution while ``Timestamp.value``
    is always nanoseconds, and mixing the two silently yields garbage.
    """
    normalised = dates.dt.normalize()
    diffs = np.column_stack(
        [(normalised - black_friday(y)).dt.days.to_numpy(dtype="float64") for y in years]
    )
    nearest_idx = np.nanargmin(np.abs(np.nan_to_num(diffs, nan=np.inf)), axis=1)
    nearest = diffs[np.arange(len(diffs)), nearest_idx]
    return pd.Series(nearest, index=dates.index).where(dates.notna())


def count_events_in_window(
    starts: pd.Series,
    ends: pd.Series,
    calendar: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Count days where ``column`` is 1, between each ``start`` and ``end``.

    Implemented as a cumulative-sum lookup rather than per-row iteration:
    with ~100k orders a naive loop over date ranges takes minutes, this
    takes milliseconds.

    Both bounds are treated as inclusive of ``start`` and exclusive of
    ``end``. Rows with a null bound, or where ``end < start``, return NaN.
    """
    cal = calendar.sort_values("date").reset_index(drop=True)
    day_index = pd.DatetimeIndex(cal["date"])
    # Exclusive prefix sum: ``prefix[i]`` is the total over the first i days,
    # so the half-open interval [lo, hi) is simply prefix[hi] - prefix[lo].
    # Without the leading zero the start day is dropped from every count.
    prefix = np.concatenate([[0], cal[column].to_numpy().cumsum()])

    def _positions(series: pd.Series) -> np.ndarray:
        return day_index.searchsorted(series.dt.normalize(), side="left")

    valid = starts.notna() & ends.notna() & (ends >= starts)
    result = pd.Series(np.nan, index=starts.index, dtype="float64")
    if not valid.any():
        return result

    lo = np.clip(_positions(starts[valid]), 0, len(prefix) - 1)
    hi = np.clip(_positions(ends[valid]), 0, len(prefix) - 1)
    result.loc[valid] = (prefix[hi] - prefix[lo]).astype("float64")
    return result
