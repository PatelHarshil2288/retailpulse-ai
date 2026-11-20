"""Chronological baseline forecasting for arbitrary uploaded datasets."""

import numpy as np
import pandas as pd

from src.time_series import (
    DEFAULT_FORECAST_HORIZONS,
    MINIMUM_FORECAST_HISTORY,
    SEASONAL_PERIODS,
    granularity_label,
    period_frequency,
    period_starts,
    validate_granularity,
)


def build_period_series(
    data,
    date_column,
    measure_column,
    aggregation,
    time_granularity="daily",
):
    """Aggregate valid date/measure pairs into a continuous daily/weekly series."""
    validate_granularity(time_granularity)
    if date_column not in data.columns:
        raise ValueError(f"Unknown date column: {date_column}")
    if measure_column not in data.columns:
        raise ValueError(f"Unknown measure column: {measure_column}")
    if aggregation not in {"sum", "mean"}:
        raise ValueError("Forecast aggregation must be Sum or Mean.")

    dates = pd.to_datetime(
        data[date_column],
        errors="coerce",
        format="mixed",
        utc=True,
    )
    values = pd.to_numeric(data[measure_column], errors="coerce")
    values = values.where(np.isfinite(values), np.nan)
    pairs = pd.DataFrame(
        {
            "date": dates.dt.tz_convert(None),
            "value": values,
        }
    ).dropna()
    if pairs.empty:
        raise ValueError(
            "No valid date and numeric value pairs are available."
        )

    pairs["period"] = period_starts(
        pairs["date"],
        time_granularity,
    )
    series = pairs.groupby("period")["value"].agg(aggregation).sort_index()
    expected_periods = pd.date_range(
        series.index.min(),
        series.index.max(),
        freq=period_frequency(time_granularity),
    )
    label = granularity_label(time_granularity).lower()
    if not series.index.equals(expected_periods):
        raise ValueError(
            f"{label.title()} history contains gaps. Resolve missing periods "
            "before forecasting; gaps are not treated as zero."
        )
    minimum_history = MINIMUM_FORECAST_HISTORY[time_granularity]
    if len(series) < minimum_history:
        raise ValueError(
            f"At least {minimum_history} consecutive {label} observations "
            "are required for forecasting."
        )
    return series.astype(float)


def _linear_forecast(training_values, steps):
    x_train = np.arange(len(training_values), dtype=float)
    coefficients = np.polyfit(x_train, np.asarray(training_values), deg=1)
    x_future = np.arange(
        len(training_values),
        len(training_values) + steps,
        dtype=float,
    )
    return np.polyval(coefficients, x_future)


def _model_prediction(
    model_name,
    training_values,
    steps,
    seasonal_period,
):
    if model_name == "Last Period Baseline":
        return np.repeat(float(training_values.iloc[-1]), steps)
    if model_name == "Seasonal Naive":
        if len(training_values) < seasonal_period:
            raise ValueError(
                "Seasonal naive forecasting requires a full seasonal cycle."
            )
        seasonal_values = training_values.iloc[-seasonal_period:].to_numpy()
        return np.resize(seasonal_values, steps).astype(float)
    if model_name == "Linear Trend":
        return _linear_forecast(training_values, steps)
    raise ValueError(f"Unsupported forecasting model: {model_name}")


def _validate_period_series(periods, time_granularity):
    if not isinstance(periods, pd.Series):
        raise TypeError("periods must be a pandas Series.")
    if not isinstance(periods.index, pd.DatetimeIndex):
        raise TypeError("The period index must be a DatetimeIndex.")
    if periods.empty:
        raise ValueError("At least one time period is required.")

    values = pd.to_numeric(periods, errors="raise").sort_index().astype(float)
    values.index = pd.DatetimeIndex(
        period_starts(values.index, time_granularity)
    )
    if values.index.has_duplicates:
        raise ValueError("Each time period must have exactly one value.")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Period values must be finite and non-missing.")

    expected_periods = pd.date_range(
        values.index.min(),
        periods=len(values),
        freq=period_frequency(time_granularity),
    )
    if not values.index.equals(expected_periods):
        label = granularity_label(time_granularity).lower()
        raise ValueError(
            f"{label.title()} history must have consecutive periods."
        )

    minimum_history = MINIMUM_FORECAST_HISTORY[time_granularity]
    if len(values) < minimum_history:
        label = granularity_label(time_granularity).lower()
        raise ValueError(
            f"At least {minimum_history} consecutive {label} observations "
            "are required."
        )
    return values


def evaluate_and_forecast(
    periods,
    horizon=None,
    time_granularity="daily",
):
    """Compare chronological baselines and forecast daily or weekly values."""
    validate_granularity(time_granularity)
    label = granularity_label(time_granularity).lower()
    max_horizon = DEFAULT_FORECAST_HORIZONS[time_granularity]
    if horizon is None:
        horizon = max_horizon
    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or not 1 <= horizon <= max_horizon
    ):
        unit = "days" if time_granularity == "daily" else "weeks"
        raise ValueError(
            f"Forecast horizon must be an integer from 1 to "
            f"{max_horizon} {unit}."
        )

    values = _validate_period_series(periods, time_granularity)
    minimum_history = MINIMUM_FORECAST_HISTORY[time_granularity]
    test_size = min(
        max_horizon,
        len(values) - minimum_history + 1,
    )
    test_start = len(values) - test_size
    if test_start < 6:
        raise ValueError(
            "Insufficient training history for chronological backtesting."
        )

    seasonal_period = SEASONAL_PERIODS[time_granularity]
    model_names = ["Last Period Baseline", "Linear Trend"]
    if test_start >= seasonal_period:
        model_names.append("Seasonal Naive")

    errors = {name: [] for name in model_names}
    for position in range(test_start, len(values)):
        training = values.iloc[:position]
        actual = float(values.iloc[position])
        for model_name in model_names:
            prediction = float(
                _model_prediction(
                    model_name,
                    training,
                    1,
                    seasonal_period,
                )[0]
            )
            errors[model_name].append(
                {
                    "actual": actual,
                    "prediction": prediction,
                    "error": actual - prediction,
                }
            )

    metrics = []
    for model_name in model_names:
        model_errors = errors[model_name]
        residuals = np.asarray([item["error"] for item in model_errors])
        metrics.append(
            {
                "Model": model_name,
                "MAE": float(np.mean(np.abs(residuals))),
                "RMSE": float(np.sqrt(np.mean(np.square(residuals)))),
            }
        )
    metrics.sort(key=lambda item: (item["RMSE"], item["MAE"]))
    best_model = metrics[0]["Model"]

    predictions = _model_prediction(
        best_model,
        values,
        horizon,
        seasonal_period,
    )
    future_periods = pd.date_range(
        values.index[-1],
        periods=horizon + 1,
        freq=period_frequency(time_granularity),
    )[1:]
    validation_errors = np.asarray(
        [item["error"] for item in errors[best_model]],
        dtype=float,
    )
    interval_width = float(np.quantile(np.abs(validation_errors), 0.9))
    forecast = pd.DataFrame(
        {
            "Period": future_periods,
            "Forecast": predictions,
            "Lower range": predictions - interval_width,
            "Upper range": predictions + interval_width,
        }
    )
    return {
        "history": values,
        "forecast": forecast,
        "metrics": metrics,
        "best_model": best_model,
        "test_periods": int(test_size),
        "time_granularity": time_granularity,
        "interval_width": interval_width,
        "notes": [
            "Models were compared on the latest chronological holdout; no random split was used.",
            "Missing periods are not treated as zero.",
            "The range is a heuristic based on holdout absolute errors, not a calibrated prediction interval.",
            "Baseline results are experimental and should not be treated as business commitments.",
        ],
    }
