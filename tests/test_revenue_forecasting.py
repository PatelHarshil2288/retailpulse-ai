
import numpy as np
import pandas as pd
import pytest

from src.revenue_forecasting import (
    create_revenue_forecast,
    evaluate_forecasting_models,
    validate_monthly_revenue,
)


@pytest.fixture
def monthly_revenue():
    dates = pd.date_range("2010-12-01", periods=13, freq="MS")
    values = [
        821452.73,
        689811.61,
        522545.56,
        716215.26,
        536968.49,
        769296.61,
        760547.01,
        718076.12,
        746779.32,
        1056435.19,
        1151263.73,
        1503866.78,
        637790.33,
    ]
    return pd.Series(values, index=dates)


def test_validate_monthly_revenue_preserves_all_observations(monthly_revenue):
    result = validate_monthly_revenue(monthly_revenue)

    assert len(result) == 13
    assert result.index.is_monotonic_increasing
    assert result.index.freqstr == "MS"


def test_validation_rejects_too_few_months():
    values = pd.Series(
        [100.0] * 5,
        index=pd.date_range("2024-01-01", periods=5, freq="MS"),
    )

    with pytest.raises(ValueError, match="At least"):
        validate_monthly_revenue(values)


def test_validation_rejects_missing_month(monthly_revenue):
    values = monthly_revenue.drop(pd.Timestamp("2011-04-01"))

    with pytest.raises(ValueError, match="consecutive months"):
        validate_monthly_revenue(values)


def test_validation_rejects_non_finite_revenue(monthly_revenue):
    values = monthly_revenue.copy()
    values.iloc[0] = np.nan

    with pytest.raises(ValueError, match="finite and non-missing"):
        validate_monthly_revenue(values)


def test_evaluation_returns_both_models(monthly_revenue):
    result = evaluate_forecasting_models(monthly_revenue)

    assert set(result["Model"]) == {
        "Last Month Baseline",
        "Linear Trend",
    }
    assert result["MAE"].notna().all()
    assert (result["MAE"] >= 0).all()
    assert result["MAE"].is_monotonic_increasing


def test_forecast_returns_three_consecutive_future_months(monthly_revenue):
    forecast, evaluation = create_revenue_forecast(
        monthly_revenue,
        horizon=3,
    )

    expected_months = pd.date_range(
        "2012-01-01",
        periods=3,
        freq="MS",
    )

    assert forecast["Month"].equals(pd.Series(expected_months, name="Month"))
    assert len(forecast) == 3
    assert forecast["Forecast Revenue"].notna().all()
    assert (forecast["Forecast Revenue"] >= 0).all()
    assert forecast["Model"].nunique() == 1
    assert forecast["Model"].iloc[0] == evaluation.iloc[0]["Model"]


def test_forecast_rejects_invalid_horizon(monthly_revenue):
    with pytest.raises(ValueError, match="positive integer"):
        create_revenue_forecast(monthly_revenue, horizon=0)
