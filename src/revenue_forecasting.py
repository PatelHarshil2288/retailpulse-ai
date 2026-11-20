"""Chronological daily and weekly revenue forecasting utilities."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.time_series import (
    DEFAULT_FORECAST_HORIZONS,
    MINIMUM_FORECAST_HISTORY,
    SEASONAL_PERIODS,
    period_frequency,
    period_starts,
    validate_granularity,
)


BACKTEST_PERIODS = 28


def validate_period_revenue(period_revenue, time_granularity="daily"):
    """Validate and normalize a consecutive daily or weekly revenue series."""
    validate_granularity(time_granularity)
    if not isinstance(period_revenue, pd.Series):
        raise TypeError("period_revenue must be a pandas Series.")
    if not isinstance(period_revenue.index, pd.DatetimeIndex):
        raise TypeError("The Series index must be a DatetimeIndex.")

    values = pd.to_numeric(period_revenue, errors="raise")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Revenue values must be finite and non-missing.")

    values = values.sort_index().astype(float)
    values.index = pd.DatetimeIndex(
        period_starts(values.index, time_granularity)
    )
    if values.index.has_duplicates:
        raise ValueError("Each time period must have exactly one revenue value.")

    expected_periods = pd.date_range(
        start=values.index.min(),
        periods=len(values),
        freq=period_frequency(time_granularity),
    )
    if not values.index.equals(expected_periods):
        raise ValueError(
            "Revenue history must have consecutive daily or weekly periods."
        )

    minimum_history = MINIMUM_FORECAST_HISTORY[time_granularity]
    if len(values) < minimum_history:
        raise ValueError(
            f"At least {minimum_history} consecutive "
            f"{time_granularity} observations are required."
        )
    return values


def _predict_linear_trend(training_values, steps):
    """Predict future values using a non-negative linear trend."""
    x_train = np.arange(len(training_values)).reshape(-1, 1)
    y_train = np.asarray(training_values, dtype=float)
    model = LinearRegression()
    model.fit(x_train, y_train)
    x_future = np.arange(
        len(training_values),
        len(training_values) + steps,
    ).reshape(-1, 1)
    return np.maximum(model.predict(x_future), 0.0)


def _predict_model(model_name, training_values, steps, seasonal_period):
    if model_name == "Last Period Baseline":
        return np.repeat(float(training_values.iloc[-1]), steps)
    if model_name == "Seasonal Naive":
        seasonal_values = training_values.iloc[-seasonal_period:].to_numpy()
        return np.resize(seasonal_values, steps).astype(float)
    if model_name == "Linear Trend":
        return _predict_linear_trend(training_values, steps)
    raise ValueError(f"Unsupported revenue forecasting model: {model_name}")


def evaluate_forecasting_models(period_revenue, time_granularity="daily"):
    """Compare available baselines on a chronological holdout."""
    values = validate_period_revenue(period_revenue, time_granularity)
    minimum_history = MINIMUM_FORECAST_HISTORY[time_granularity]
    test_periods = min(
        DEFAULT_FORECAST_HORIZONS[time_granularity],
        len(values) - minimum_history + 1,
        BACKTEST_PERIODS,
    )
    test_start = len(values) - test_periods
    seasonal_period = SEASONAL_PERIODS[time_granularity]
    models = ["Last Period Baseline", "Linear Trend"]
    if test_start >= seasonal_period:
        models.append("Seasonal Naive")

    errors = {name: [] for name in models}
    for position in range(test_start, len(values)):
        training = values.iloc[:position]
        actual = float(values.iloc[position])
        for model_name in models:
            prediction = float(
                _predict_model(
                    model_name,
                    training,
                    1,
                    seasonal_period,
                )[0]
            )
            errors[model_name].append(actual - prediction)

    results = pd.DataFrame(
        [
            {
                "Model": name,
                "MAE": float(np.mean(np.abs(model_errors))),
                "RMSE": float(np.sqrt(np.mean(np.square(model_errors)))),
            }
            for name, model_errors in errors.items()
        ]
    ).sort_values("RMSE", ignore_index=True)
    return results


def create_revenue_forecast(
    period_revenue,
    horizon=None,
    time_granularity="daily",
):
    """Select the lowest-error baseline and forecast future periods."""
    validate_granularity(time_granularity)
    if horizon is None:
        horizon = DEFAULT_FORECAST_HORIZONS[time_granularity]
    max_horizon = DEFAULT_FORECAST_HORIZONS[time_granularity]
    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or not 1 <= horizon <= max_horizon
    ):
        unit = "days" if time_granularity == "daily" else "weeks"
        raise ValueError(
            f"horizon must be an integer from 1 to {max_horizon} {unit}."
        )

    values = validate_period_revenue(period_revenue, time_granularity)
    evaluation = evaluate_forecasting_models(values, time_granularity)
    selected_model = evaluation.iloc[0]["Model"]
    predictions = _predict_model(
        selected_model,
        values,
        horizon,
        SEASONAL_PERIODS[time_granularity],
    )
    future_periods = pd.date_range(
        values.index[-1],
        periods=horizon + 1,
        freq=period_frequency(time_granularity),
    )[1:]
    forecast = pd.DataFrame(
        {
            "Period": future_periods,
            "Forecast Revenue": predictions,
            "Model": selected_model,
        }
    )
    return forecast, evaluation
