import numpy as np
import pandas as pd
import pytest

from src.universal_forecasting import (
    build_period_series,
    evaluate_and_forecast,
)


def _daily_data(periods=120):
    dates = pd.date_range("2022-01-01", periods=periods, freq="D")
    seasonal = np.resize(
        np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]),
        periods,
    )
    return pd.DataFrame(
        {
            "EventDate": dates,
            "Amount": seasonal + np.arange(periods, dtype=float),
            "Group": ["A"] * periods,
        }
    )


def _weekly_data(periods=80):
    dates = pd.date_range("2022-01-03", periods=periods, freq="W-MON")
    return pd.DataFrame(
        {
            "EventDate": dates,
            "Amount": 100 + np.arange(periods, dtype=float),
        }
    )


def test_build_period_series_aggregates_date_and_numeric_pairs():
    data = pd.DataFrame(
        {
            "When": list(pd.date_range("2024-01-01", periods=60, freq="D"))
            + [pd.Timestamp("2024-01-01")],
            "Metric": list(range(60)) + [4],
        }
    )

    series = build_period_series(data, "When", "Metric", "sum")

    assert len(series) == 60
    assert series.iloc[0] == 4
    assert series.index[0] == pd.Timestamp("2024-01-01")
    assert series.index.is_monotonic_increasing


def test_build_period_series_uses_monday_start_for_weekly_periods():
    series = build_period_series(
        _weekly_data(),
        "EventDate",
        "Amount",
        "mean",
        "weekly",
    )

    assert series.index[0] == pd.Timestamp("2022-01-03")
    assert (series.index.weekday == 0).all()


def test_evaluate_forecast_uses_daily_holdout_and_future_periods():
    series = build_period_series(
        _daily_data(),
        "EventDate",
        "Amount",
        "mean",
    )

    report = evaluate_and_forecast(series)

    assert report["test_periods"] == 28
    assert [row["Model"] for row in report["metrics"]][0] == report["best_model"]
    assert {row["Model"] for row in report["metrics"]} == {
        "Last Period Baseline",
        "Linear Trend",
        "Seasonal Naive",
    }
    assert report["forecast"]["Period"].tolist() == list(
        pd.date_range(series.index[-1], periods=29, freq="D")[1:]
    )
    assert (
        report["forecast"]["Lower range"]
        <= report["forecast"]["Forecast"]
    ).all()
    assert (
        report["forecast"]["Forecast"]
        <= report["forecast"]["Upper range"]
    ).all()


def test_evaluate_forecast_uses_four_week_horizon():
    series = build_period_series(
        _weekly_data(),
        "EventDate",
        "Amount",
        "sum",
        "weekly",
    )

    report = evaluate_and_forecast(series, 4, "weekly")

    assert report["test_periods"] == 4
    assert report["forecast"]["Period"].tolist() == list(
        pd.date_range(series.index[-1], periods=5, freq="W-MON")[1:]
    )


def test_build_period_series_rejects_gaps_instead_of_filling_zero():
    data = _daily_data().drop(index=4)

    with pytest.raises(ValueError, match="contains gaps"):
        build_period_series(data, "EventDate", "Amount", "sum")


def test_forecast_requires_sufficient_history_and_valid_horizon():
    short_data = _daily_data(periods=59)
    with pytest.raises(ValueError, match="At least 60"):
        build_period_series(short_data, "EventDate", "Amount", "mean")

    series = build_period_series(
        _daily_data(),
        "EventDate",
        "Amount",
        "mean",
    )
    with pytest.raises(ValueError, match="28 days"):
        evaluate_and_forecast(series, 29)
