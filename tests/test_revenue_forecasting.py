import numpy as np
import pandas as pd
import pytest

from src.revenue_forecasting import (
    create_revenue_forecast,
    evaluate_forecasting_models,
    validate_period_revenue,
)


@pytest.fixture
def daily_revenue():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    values = 100 + np.arange(len(dates)) + np.resize(
        np.array([0.0, 5.0, 10.0, 3.0, 8.0, 2.0, 12.0]),
        len(dates),
    )
    return pd.Series(values, index=dates)


@pytest.fixture
def weekly_revenue():
    dates = pd.date_range("2022-01-03", periods=80, freq="W-MON")
    values = 500 + np.arange(len(dates)) * 2 + np.resize(
        np.arange(52, dtype=float),
        len(dates),
    )
    return pd.Series(values, index=dates)


@pytest.mark.parametrize(
    ("granularity", "periods"),
    [("daily", 120), ("weekly", 80)],
)
def test_validate_period_revenue_preserves_consecutive_observations(
    daily_revenue,
    weekly_revenue,
    granularity,
    periods,
):
    values = daily_revenue if granularity == "daily" else weekly_revenue

    result = validate_period_revenue(values, granularity)

    assert len(result) == periods
    assert result.index.is_monotonic_increasing
    assert result.index.equals(
        pd.date_range(
            result.index[0],
            periods=periods,
            freq="D" if granularity == "daily" else "W-MON",
        )
    )


@pytest.mark.parametrize(
    ("granularity", "minimum", "frequency"),
    [("daily", 60, "D"), ("weekly", 26, "W-MON")],
)
def test_validation_rejects_insufficient_history(
    granularity,
    minimum,
    frequency,
):
    values = pd.Series(
        [100.0] * (minimum - 1),
        index=pd.date_range(
            "2024-01-01",
            periods=minimum - 1,
            freq=frequency,
        ),
    )

    with pytest.raises(ValueError, match="At least"):
        validate_period_revenue(values, granularity)


def test_validation_rejects_missing_period(daily_revenue):
    values = daily_revenue.drop(pd.Timestamp("2024-02-10"))

    with pytest.raises(ValueError, match="consecutive"):
        validate_period_revenue(values)


def test_validation_rejects_non_finite_revenue(daily_revenue):
    values = daily_revenue.copy()
    values.iloc[0] = np.nan

    with pytest.raises(ValueError, match="finite and non-missing"):
        validate_period_revenue(values)


def test_validation_rejects_monthly_granularity(daily_revenue):
    with pytest.raises(ValueError, match="daily.*weekly"):
        validate_period_revenue(daily_revenue, "monthly")


@pytest.mark.parametrize(
    ("granularity", "fixture_name", "horizon"),
    [("daily", "daily_revenue", 28), ("weekly", "weekly_revenue", 4)],
)
def test_forecast_returns_consecutive_periods_and_evaluation(
    request,
    granularity,
    fixture_name,
    horizon,
):
    values = request.getfixturevalue(fixture_name)

    forecast, evaluation = create_revenue_forecast(
        values,
        time_granularity=granularity,
    )

    frequency = "D" if granularity == "daily" else "W-MON"
    expected_periods = pd.date_range(
        values.index[-1],
        periods=horizon + 1,
        freq=frequency,
    )[1:]
    assert forecast["Period"].equals(
        pd.Series(expected_periods, name="Period")
    )
    assert len(forecast) == horizon
    assert forecast["Forecast Revenue"].notna().all()
    assert (forecast["Forecast Revenue"] >= 0).all()
    assert forecast["Model"].nunique() == 1
    assert forecast["Model"].iloc[0] == evaluation.iloc[0]["Model"]
    assert evaluation["MAE"].notna().all()
    assert evaluation["MAE"].is_monotonic_increasing


def test_forecast_rejects_horizon_above_selected_cadence(daily_revenue):
    with pytest.raises(ValueError, match="28 days"):
        create_revenue_forecast(daily_revenue, horizon=29)
