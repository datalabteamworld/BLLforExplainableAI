"""Nigerian holiday and commercial-event calendar.

A drop-in counterpart to :mod:`olist.calendar_br`, exposing the same four
public functions so :func:`olist.features.add_calendar_features` needs no
conditional.

Three things differ from the Brazilian calendar, and none of them is cosmetic:

**The two Eids move.** Eid al-Fitr and Eid al-Adha follow the lunar calendar,
so they drift roughly eleven days earlier each year. Nigeria observes both as
two-day public holidays, and they matter for logistics: the northern corridors
(Kano, Kaduna, Maiduguri) effectively pause. A fixed-date holiday table cannot
express this, which is why ``workalendar`` is used rather than a hard-coded
list.

**The commercial peak is December, not November.** Nigerian retail concentrates
around Christmas and New Year rather than Black Friday, and the "ember months"
(October to December) carry a widely reported rise in road-freight incidents.

**There is no Carnival.** The Brazilian module's five-day Carnival window has
no Nigerian equivalent, so the slot is used for the Eid periods instead.

A caveat worth carrying into any result that depends on this module:
``workalendar`` warns that its Islamic date computation is approximate. Eid
dates here may be a day out from the dates actually gazetted by the Nigerian
government, which announces them on sighting of the moon. For a synthetic
demonstration environment that is acceptable; for a production system serving
Nigerian operations it would not be, and the gazetted dates should be loaded
from a table instead.
"""

from __future__ import annotations

import functools
import warnings
from datetime import date

import numpy as np
import pandas as pd


@functools.lru_cache(maxsize=None)
def _calendar():
    """The workalendar Nigeria calendar, imported lazily.

    ``workalendar`` emits a UserWarning about Islamic date approximation on
    first use. It is suppressed here because the caveat is documented in this
    module's docstring, and a warning repeated on every feature build trains
    the reader to ignore warnings generally.
    """
    from workalendar.africa import Nigeria

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Nigeria()


def national_holidays(year: int) -> dict[date, str]:
    """Fixed and movable national public holidays for one year."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return {d: name for d, name in _calendar().holidays(year)}


def eid_periods(year: int) -> list[pd.Timestamp]:
    """Both Eids, each extended to the two days Nigeria observes.

    The gazetted holiday is two days for each Eid. The day after is included
    because that is when the dispatch backlog actually clears, which is the
    thing the transit-window count is trying to measure.
    """
    days: list[pd.Timestamp] = []
    for d, name in national_holidays(year).items():
        if "eid" in name.lower():
            start = pd.Timestamp(d)
            days.extend([start, start + pd.Timedelta(days=1)])
    return sorted(set(days))


def peak_shopping_day(year: int) -> pd.Timestamp:
    """The commercial peak: 23 December.

    Nigerian e-commerce volume concentrates in the days before Christmas
    rather than on a single promotional Friday. The 23rd is used as the
    reference point because it is the last practical delivery day before the
    Christmas shutdown, so it is where the promise pressure peaks.
    """
    return pd.Timestamp(year, 12, 23)


def ember_months(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """October to December, the period of elevated road-freight risk."""
    return pd.Timestamp(year, 10, 1), pd.Timestamp(year, 12, 31)


def build_daily_calendar(start, end) -> pd.DataFrame:
    """One row per day with holiday, weekend and peak flags.

    Column names match :func:`olist.calendar_br.build_daily_calendar` exactly
    -- ``is_holiday``, ``is_black_friday``, ``is_cyber_monday``,
    ``is_carnival``, ``is_weekend``, ``is_working_day`` -- so the feature
    builder is identical across environments.

    The names are retained rather than renamed because they are consumed by
    shared code. ``is_black_friday`` carries the December peak,
    ``is_carnival`` carries the Eid periods, and ``is_cyber_monday`` carries
    the day after the peak. Mapping is documented here so a reader is not
    misled by the Brazilian vocabulary.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    years = range(start.year, end.year + 1)

    cal = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})

    holidays = {pd.Timestamp(d) for y in years for d in national_holidays(y)}
    peak = {peak_shopping_day(y) for y in years}
    after_peak = {d + pd.Timedelta(days=1) for d in peak}
    eid = {d for y in years for d in eid_periods(y)}

    names = {pd.Timestamp(d): n for y in years for d, n in national_holidays(y).items()}
    cal["holiday_name"] = cal["date"].map(names)

    cal["is_holiday"] = cal["date"].isin(holidays).astype("int8")
    cal["is_black_friday"] = cal["date"].isin(peak).astype("int8")
    cal["is_cyber_monday"] = cal["date"].isin(after_peak).astype("int8")
    cal["is_carnival"] = cal["date"].isin(eid).astype("int8")
    cal["is_weekend"] = cal["date"].dt.dayofweek.isin([5, 6]).astype("int8")

    cal["is_working_day"] = (
        (cal["is_holiday"] == 0) & (cal["is_weekend"] == 0) & (cal["is_carnival"] == 0)
    ).astype("int8")
    return cal


def days_to_nearest_black_friday(dates: pd.Series, years) -> pd.Series:
    """Signed days to the nearest December peak.

    Named for the Brazilian equivalent because shared feature code calls it
    by that name. Negative means before the peak, positive after.
    """
    normalised = pd.to_datetime(dates).dt.normalize()
    offsets = np.column_stack(
        [(normalised - peak_shopping_day(y)).dt.days.to_numpy(dtype="float64")
         for y in years]
    )
    nearest = offsets[np.arange(len(offsets)), np.abs(offsets).argmin(axis=1)]
    return pd.Series(nearest, index=dates.index)


def count_events_in_window(
    start: pd.Series, end: pd.Series, calendar: pd.DataFrame, flag: str
) -> pd.Series:
    """Count flagged days between two dates, inclusive of the start.

    Implemented as a cumulative-sum lookup rather than a per-row loop: at
    ~100k orders a loop takes minutes and this takes milliseconds.
    """
    cal = calendar.sort_values("date").reset_index(drop=True)
    cumulative = cal[flag].cumsum()
    lookup = pd.Series(cumulative.to_numpy(), index=cal["date"])

    s = pd.to_datetime(start).dt.normalize()
    e = pd.to_datetime(end).dt.normalize()

    before_start = s.map(lookup).astype("float64") - s.map(lookup.shift(1, fill_value=0)).astype("float64")
    at_end = e.map(lookup).astype("float64")
    at_start = s.map(lookup).astype("float64")

    counts = at_end - at_start + before_start
    return counts.where(e >= s, 0.0)
