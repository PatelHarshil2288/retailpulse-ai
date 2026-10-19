
"""Revenue forecasting utilities for RetailPulse AI.

Uses chronological backtesting to compare a naive baseline with a
linear trend model. Forecasts are experimental because the available
monthly history is limited.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


MINIMUM_MONTHS = 6
BACKTEST_MONTHS = 3


def validate_monthly_revenue(monthly_revenue):
    """Validate and normalize a monthly revenue Series."""
    if not isinstance(monthly_revenue, pd.Series):
        raise TypeError("monthly_revenue must be a pandas Series.")

    if len(monthly_revenue) < MINIMUM_MONTHS:
        raise ValueError(
            f"At least {MINIMUM_MONTHS} monthly observations are required."
        )

    if not isinstance(monthly_revenue.index, pd.DatetimeIndex):
        raise TypeError("The Series index must be a DatetimeIndex.")

    values = pd.to_numeric(monthly_revenue, errors="raise")

    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Revenue values must be finite and non-missing.")

    values = values.sort_index().astype(float)
    values.index = values.index.to_period("M").to_timestamp()

    if values.index.has_duplicates:
        raise ValueError("Each month must have exactly one revenue value.")

    expected_months = pd.date_range(
        start=values.index.min(),
        periods=len(values),
        freq="MS",
    )

    if not values.index.equals(expected_months):
        raise ValueError("Monthly revenue must have consecutive months.")

    return values


def _predict_linear_trend(training_values, steps):
    """Predict future values using a linear trend."""
    x_train = np.arange(len(training_values)).reshape(-1, 1)
    y_train = np.asarray(training_values, dtype=float)

    model = LinearRegression()
    model.fit(x_train, y_train)

    x_future = np.arange(
        len(training_values),
        len(training_values) + steps,
    ).reshape(-1, 1)

    # Revenue forecasts cannot be negative.
    return np.maximum(model.predict(x_future), 0.0)


def evaluate_forecasting_models(monthly_revenue):
    """Compare models using the final three months as a rolling test."""
    values = validate_monthly_revenue(monthly_revenue)

    test_months = min(BACKTEST_MONTHS, len(values) - MINIMUM_MONTHS + 1)
    if test_months < 1:
        raise ValueError("Not enough history for chronological backtesting.")

    errors = {
        "Last Month Baseline": [],
        "Linear Trend": [],
    }

    for test_position in range(len(values) - test_months, len(values)):
        training = values.iloc[:test_position]
        actual = float(values.iloc[test_position])

        baseline_prediction = float(training.iloc[-1])
        trend_prediction = float(_predict_linear_trend(training, 1)[0])

        errors["Last Month Baseline"].append(
            abs(actual - baseline_prediction)
        )
        errors["Linear Trend"].append(
            abs(actual - trend_prediction)
        )

    results = pd.DataFrame(
        [
            {"Model": name, "MAE": float(np.mean(model_errors))}
            for name, model_errors in errors.items()
        ]
    ).sort_values("MAE", ignore_index=True)

    return results


def create_revenue_forecast(monthly_revenue, horizon=3):
    """Select the lower-MAE model and forecast future monthly revenue."""
    values = validate_monthly_revenue(monthly_revenue)

    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    evaluation = evaluate_forecasting_models(values)
    selected_model = evaluation.iloc[0]["Model"]

    if selected_model == "Last Month Baseline":
        predictions = np.repeat(float(values.iloc[-1]), horizon)
    else:
        predictions = _predict_linear_trend(values, horizon)

    future_months = pd.date_range(
        start=values.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    forecast = pd.DataFrame(
        {
            "Month": future_months,
            "Forecast Revenue": predictions,
            "Model": selected_model,
        }
    )

    return forecast, evaluation
