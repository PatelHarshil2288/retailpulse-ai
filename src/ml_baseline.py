"""Baseline supervised model evaluation for uploaded datasets."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.ml_limits import MAX_MODEL_ROWS


def _encode_boolean(series):
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "y": 1.0,
            "n": 0.0,
            "1": 1.0,
            "0": 0.0,
        }
    )
    return pd.to_numeric(mapped, errors="coerce")


def _build_feature_matrix(data, profile, target_column):
    column_profiles = {item["name"]: item for item in profile["columns"]}
    feature_frame = pd.DataFrame(index=data.index)
    dropped_features = []

    for name, column in column_profiles.items():
        if name == target_column:
            continue
        series = data[name]
        detected_type = column["detected_type"]

        if detected_type == "numeric":
            feature_frame[name] = pd.to_numeric(series, errors="coerce")
        elif detected_type == "date":
            dates = pd.to_datetime(
                series,
                errors="coerce",
                format="mixed",
                utc=True,
            )
            feature_frame[name] = (
                dates.dt.tz_convert(None).astype("int64") / 1_000_000_000
            )
            feature_frame[name] = feature_frame[name].where(
                dates.notna(),
                np.nan,
            )
        elif detected_type == "boolean":
            feature_frame[name] = _encode_boolean(series)
        elif detected_type == "categorical":
            if column["unique_count"] > 100:
                dropped_features.append(name)
                continue
            categorical = series.astype("string").fillna("<missing>")
            codes, _ = pd.factorize(categorical, sort=True)
            feature_frame[name] = pd.Series(codes, index=series.index).astype(
                float
            )
            feature_frame[name] = feature_frame[name].where(
                series.notna(),
                np.nan,
            )
        else:
            dropped_features.append(name)

    feature_frame = feature_frame.dropna(axis=1, how="all")
    if feature_frame.empty:
        raise ValueError(
            "No usable feature columns are available after preprocessing."
        )

    medians = feature_frame.median(numeric_only=True)
    feature_frame = feature_frame.fillna(medians)
    remaining_missing = feature_frame.isna().sum()
    if int(remaining_missing.sum()) > 0:
        feature_frame = feature_frame.fillna(0.0)

    return feature_frame, dropped_features


def _select_task_type(target_profile, target_values, requested_task):
    if requested_task in {"regression", "classification"}:
        return requested_task
    if target_profile["detected_type"] == "numeric" and target_values.nunique(
        dropna=True
    ) >= 10:
        return "regression"
    return "classification"


def run_baseline_model_evaluation(
    data,
    profile,
    target_column,
    requested_task="auto",
):
    """Run supervised baseline evaluation after an explicit target selection."""
    if target_column not in data.columns:
        raise ValueError(f"Unknown target column: {target_column}")
    if requested_task not in {"auto", "regression", "classification"}:
        raise ValueError(f"Unsupported ML task: {requested_task}")
    if len(data) > MAX_MODEL_ROWS:
        raise ValueError(
            f"Baseline model evaluation is limited to {MAX_MODEL_ROWS:,} "
            "rows per run. Upload a smaller dataset before training."
        )

    target_profile = next(
        (
            item
            for item in profile["columns"]
            if item["name"] == target_column
        ),
        None,
    )
    if not target_profile:
        raise ValueError(f"No profile metadata found for target: {target_column}")

    features, dropped_features = _build_feature_matrix(
        data,
        profile,
        target_column,
    )
    target_values = data[target_column]
    task_type = _select_task_type(
        target_profile,
        target_values,
        requested_task,
    )

    if task_type == "regression":
        y = pd.to_numeric(target_values, errors="coerce")
        valid = y.notna()
        y = y.loc[valid]
        x = features.loc[valid]
        if len(y) < 60:
            raise ValueError(
                "At least 60 valid target rows are required for regression evaluation."
            )
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
        )
        models = [
            ("Linear Regression", LinearRegression()),
            (
                "Random Forest Regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
        model_rows = []
        for name, model in models:
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            model_rows.append(
                {
                    "Model": name,
                    "MAE": float(mean_absolute_error(y_test, prediction)),
                    "RMSE": float(
                        np.sqrt(mean_squared_error(y_test, prediction))
                    ),
                    "R2": float(r2_score(y_test, prediction)),
                }
            )
        best_model = min(model_rows, key=lambda row: row["RMSE"])["Model"]
        return {
            "task_type": task_type,
            "target_column": target_column,
            "rows_used": int(len(y)),
            "feature_count": int(x.shape[1]),
            "dropped_features": dropped_features,
            "models": model_rows,
            "best_model": best_model,
            "notes": [
                "Regression metrics are based on a random 80/20 split.",
                "Use temporal splits for forecasting-style targets.",
            ],
            "confusion_matrix": None,
        }

    y = target_values.astype("string")
    valid = y.notna() & (y.str.len() > 0)
    y = y.loc[valid]
    x = features.loc[valid]
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            "Classification requires at least two target classes."
        )
    if int(class_counts.min()) < 2:
        raise ValueError(
            "Each target class needs at least two rows for train/test splitting."
        )
    if len(y) < 80:
        raise ValueError(
            "At least 80 labeled rows are required for classification evaluation."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    models = [
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1000),
        ),
        (
            "Random Forest Classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=1,
                class_weight="balanced_subsample",
            ),
        ),
    ]
    model_rows = []
    predictions_by_model = {}
    for name, model in models:
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        predictions_by_model[name] = prediction
        model_rows.append(
            {
                "Model": name,
                "Accuracy": float(accuracy_score(y_test, prediction)),
                "Precision": float(
                    precision_score(
                        y_test,
                        prediction,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "Recall": float(
                    recall_score(
                        y_test,
                        prediction,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "F1": float(
                    f1_score(
                        y_test,
                        prediction,
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }
        )
    best_model = max(model_rows, key=lambda row: row["F1"])["Model"]
    labels = sorted(y_test.unique())
    matrix = confusion_matrix(
        y_test,
        predictions_by_model[best_model],
        labels=labels,
    )
    return {
        "task_type": task_type,
        "target_column": target_column,
        "rows_used": int(len(y)),
        "feature_count": int(x.shape[1]),
        "dropped_features": dropped_features,
        "models": model_rows,
        "best_model": best_model,
        "notes": [
            "Classification metrics are weighted by class frequency.",
            "Confusion matrix is for the best-F1 baseline model.",
        ],
        "confusion_matrix": {
            "labels": [str(label) for label in labels],
            "matrix": matrix.tolist(),
        },
    }
