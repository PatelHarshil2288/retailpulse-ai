import pandas as pd
import pytest

from src.time_series import (
    granularity_label,
    period_frequency,
    period_starts,
    validate_granularity,
)


def test_daily_period_starts_normalize_to_midnight():
    dates = pd.Series(
        pd.to_datetime(["2024-01-01 09:30", "2024-01-02 18:15"])
    )

    periods = period_starts(dates, "daily")

    assert periods.tolist() == list(
        pd.to_datetime(["2024-01-01", "2024-01-02"])
    )
    assert period_frequency("daily") == "D"
    assert granularity_label("daily") == "Daily"


def test_weekly_period_starts_use_monday_to_sunday():
    dates = pd.Series(pd.to_datetime(["2024-01-07", "2024-01-08"]))

    periods = period_starts(dates, "weekly")

    assert periods.tolist() == list(
        pd.to_datetime(["2024-01-01", "2024-01-08"])
    )
    assert period_frequency("weekly") == "W-MON"
    assert granularity_label("weekly") == "Weekly"


def test_time_series_helpers_reject_monthly_granularity():
    with pytest.raises(ValueError, match="daily.*weekly"):
        validate_granularity("monthly")

    with pytest.raises(ValueError, match="daily.*weekly"):
        period_starts(pd.Series(pd.to_datetime(["2024-01-01"])), "monthly")
