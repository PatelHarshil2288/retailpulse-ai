"""Shared daily and Monday-start weekly time-series helpers."""

import pandas as pd


TIME_GRANULARITIES = ("daily", "weekly")
DEFAULT_FORECAST_HORIZONS = {"daily": 28, "weekly": 4}
MINIMUM_FORECAST_HISTORY = {"daily": 60, "weekly": 26}
SEASONAL_PERIODS = {"daily": 7, "weekly": 52}


def validate_granularity(granularity):
    """Return a supported time granularity or raise an explicit error."""
    if granularity not in TIME_GRANULARITIES:
        raise ValueError(
            "Time granularity must be either 'daily' or 'weekly'."
        )
    return granularity


def period_starts(values, granularity):
    """Convert date-like values to daily or Monday-start weekly periods."""
    validate_granularity(granularity)
    dates = pd.to_datetime(
        pd.Series(values),
        errors="coerce",
        format="mixed",
        utc=True,
    ).dt.tz_convert(None)
    if granularity == "daily":
        return dates.dt.normalize()
    return dates.dt.to_period("W-SUN").dt.start_time


def period_frequency(granularity):
    """Return the Pandas frequency alias for a supported period."""
    validate_granularity(granularity)
    return "D" if granularity == "daily" else "W-MON"


def granularity_label(granularity):
    """Return a user-facing label for a supported time granularity."""
    validate_granularity(granularity)
    return "Daily" if granularity == "daily" else "Weekly"


def period_display(value, granularity):
    """Return a clear daily date or Monday-start weekly label."""
    validate_granularity(granularity)
    date = pd.Timestamp(value)
    if granularity == "weekly":
        return f"Week of {date:%d %b %Y}"
    return date.strftime("%d %b %Y")
