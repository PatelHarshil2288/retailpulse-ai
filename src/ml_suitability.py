"""Machine-learning suitability checks for uploaded tabular datasets."""

import pandas as pd

from src.ml_limits import MAX_MODEL_ROWS
from src.time_series import MINIMUM_FORECAST_HISTORY, period_starts


def _status_label(status):
    labels = {
        "pass": "Ready",
        "review": "Needs review",
        "block": "Blocked",
    }
    return labels.get(status, status.title())


def _add_check(checks, name, status, evidence):
    checks.append(
        {
            "name": name,
            "status": status,
            "label": _status_label(status),
            "evidence": evidence,
        }
    )


def _is_identifier_like(column):
    name = column["name"].lower().replace("_", " ")
    keywords = (" id", "id ", "invoice", "order", "transaction", "uuid")
    return column["high_cardinality"] or any(keyword in name for keyword in keywords)


def assess_ml_suitability(data, profile):
    """Assess whether uploaded data is suitable for common ML workflows."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("ML suitability assessment requires a pandas DataFrame.")

    summary = profile["summary"]
    columns = profile["columns"]
    row_count = int(summary["rows"])
    column_count = int(summary["columns"])
    issue_count = int(summary["issue_count"])
    missing_cells = int(summary["missing_cells"])

    numeric_columns = [
        item for item in columns if item["detected_type"] == "numeric"
    ]
    categorical_columns = [
        item for item in columns if item["detected_type"] == "categorical"
    ]
    date_columns = [
        item for item in columns if item["detected_type"] == "date"
    ]
    identifier_like_columns = [
        item["name"] for item in columns if _is_identifier_like(item)
    ]

    checks = []

    if row_count > MAX_MODEL_ROWS:
        _add_check(
            checks,
            "Dataset size",
            "review",
            f"{row_count:,} rows exceeds the {MAX_MODEL_ROWS:,}-row limit for model runs; upload a smaller dataset for ML analysis.",
        )
    elif row_count >= 1000:
        _add_check(
            checks,
            "Dataset size",
            "pass",
            f"{row_count:,} rows are usually adequate for baseline modeling.",
        )
    elif row_count >= 200:
        _add_check(
            checks,
            "Dataset size",
            "review",
            f"{row_count:,} rows are usable for experiments but may limit generalization.",
        )
    else:
        _add_check(
            checks,
            "Dataset size",
            "block",
            f"{row_count:,} rows are likely too few for reliable model validation.",
        )

    if column_count >= 3:
        _add_check(
            checks,
            "Feature availability",
            "pass",
            f"{column_count - 1:,} potential feature columns are available.",
        )
    elif column_count >= 2:
        _add_check(
            checks,
            "Feature availability",
            "review",
            "Only one likely feature column is available after choosing a target.",
        )
    else:
        _add_check(
            checks,
            "Feature availability",
            "block",
            "At least one feature and one target column are required for supervised models.",
        )

    total_cells = max(row_count * max(column_count, 1), 1)
    missing_percent = 100 * missing_cells / total_cells
    if missing_percent <= 5:
        _add_check(
            checks,
            "Missing values",
            "pass",
            f"Missingness is low ({missing_percent:.1f}% of cells).",
        )
    elif missing_percent <= 20:
        _add_check(
            checks,
            "Missing values",
            "review",
            f"Missingness is moderate ({missing_percent:.1f}% of cells).",
        )
    else:
        _add_check(
            checks,
            "Missing values",
            "block",
            f"Missingness is high ({missing_percent:.1f}% of cells).",
        )

    if numeric_columns and categorical_columns:
        _add_check(
            checks,
            "Feature types",
            "pass",
            "Both numeric and categorical signals are available.",
        )
    elif numeric_columns or categorical_columns:
        _add_check(
            checks,
            "Feature types",
            "review",
            "Only one feature type is strongly represented; model options are narrower.",
        )
    else:
        _add_check(
            checks,
            "Feature types",
            "block",
            "No clear numeric or categorical feature columns were detected.",
        )

    regression_targets = []
    for column in numeric_columns:
        valid_count = (
            column["non_missing_count"] - column["invalid_numeric_count"]
        )
        if valid_count >= max(50, int(0.7 * row_count)) and column["unique_count"] >= 10:
            regression_targets.append(column["name"])

    classification_targets = []
    classification_balance = []
    for column in categorical_columns:
        unique_count = int(column["unique_count"])
        non_missing_count = int(column["non_missing_count"])
        if unique_count < 2 or unique_count > max(20, int(row_count * 0.2)):
            continue
        if non_missing_count < max(50, int(0.7 * row_count)):
            continue
        value_counts = (
            data[column["name"]]
            .astype("string")
            .dropna()
            .value_counts()
        )
        if value_counts.empty:
            continue
        largest_share = float(value_counts.iloc[0] / value_counts.sum())
        classification_targets.append(column["name"])
        classification_balance.append((column["name"], largest_share))

    if regression_targets or classification_targets:
        candidate_text = []
        if regression_targets:
            candidate_text.append(
                "Regression: " + ", ".join(regression_targets[:3])
            )
        if classification_targets:
            candidate_text.append(
                "Classification: " + ", ".join(classification_targets[:3])
            )
        _add_check(
            checks,
            "Target availability",
            "review",
            "Potential supervised targets were detected. Confirm one target column before training. "
            + " | ".join(candidate_text),
        )
    else:
        _add_check(
            checks,
            "Target availability",
            "review",
            "No confident supervised target was auto-selected. User confirmation is required.",
        )

    if classification_balance:
        most_imbalanced = max(classification_balance, key=lambda item: item[1])
        if most_imbalanced[1] > 0.9:
            _add_check(
                checks,
                "Class balance",
                "review",
                f"{most_imbalanced[0]} is highly imbalanced (largest class {most_imbalanced[1] * 100:.1f}%).",
            )
        else:
            _add_check(
                checks,
                "Class balance",
                "pass",
                "Detected classification targets are not heavily dominated by one class.",
            )
    else:
        _add_check(
            checks,
            "Class balance",
            "review",
            "Class balance cannot be evaluated until a classification target is confirmed.",
        )

    if identifier_like_columns:
        _add_check(
            checks,
            "Potential data leakage",
            "review",
            "Identifier-like columns detected: "
            + ", ".join(identifier_like_columns[:3])
            + ". Exclude them from target and feature sets unless justified.",
        )
    else:
        _add_check(
            checks,
            "Potential data leakage",
            "pass",
            "No obvious identifier-style leakage columns were detected.",
        )

    if issue_count <= 3:
        _add_check(
            checks,
            "Data quality",
            "pass" if issue_count == 0 else "review",
            f"{issue_count} profile issue(s) detected; review findings before modeling.",
        )
    else:
        _add_check(
            checks,
            "Data quality",
            "review",
            f"{issue_count} profile issues were detected; cleaning should precede model training.",
        )

    forecast_readiness = None
    if date_columns and numeric_columns:
        date_name = date_columns[0]["name"]
        value_name = numeric_columns[0]["name"]
        dates = pd.to_datetime(
            data[date_name],
            errors="coerce",
            format="mixed",
            utc=True,
        )
        values = pd.to_numeric(data[value_name], errors="coerce")
        valid_pairs = pd.DataFrame({"d": dates, "v": values}).dropna()
        period_counts = {
            granularity: (
                int(
                    period_starts(
                        valid_pairs["d"],
                        granularity,
                    ).nunique()
                )
                if not valid_pairs.empty
                else 0
            )
            for granularity in ("daily", "weekly")
        }
        forecastable = [
            f"{granularity} ({count} periods)"
            for granularity, count in period_counts.items()
            if count >= MINIMUM_FORECAST_HISTORY[granularity]
        ]
        observed = [
            f"{granularity} ({count} periods)"
            for granularity, count in period_counts.items()
            if count >= 12
        ]
        if forecastable and len(valid_pairs) >= 100:
            forecast_readiness = (
                "ready",
                f"Forecasting has sufficient observed daily/weekly periods "
                f"in {date_name}: {', '.join(forecastable)}. "
                "The selected series must still have consecutive periods.",
            )
        elif observed:
            forecast_readiness = (
                "review",
                f"Forecasting coverage is limited in {date_name}: "
                f"{', '.join(observed)}. The selected series must have "
                "consecutive periods.",
            )
        else:
            forecast_readiness = (
                "blocked",
                "Not enough chronological coverage for meaningful forecasting.",
            )

    workflows = []
    if regression_targets:
        workflows.append(
            "Regression baseline: train/test split with MAE, RMSE, and R-squared."
        )
    if classification_targets:
        workflows.append(
            "Classification baseline: stratified split with accuracy, precision, recall, and F1-score."
        )
    if len(numeric_columns) >= 2 and row_count >= 100:
        workflows.append(
            "Clustering baseline: K-means with cluster summaries and silhouette score."
        )
        workflows.append(
            "Anomaly baseline: Isolation Forest on numeric features with review of top anomaly candidates."
        )
    if forecast_readiness:
        status, message = forecast_readiness
        if status == "ready":
            workflows.append(
                "Forecasting baseline: chronological split with MAE/RMSE and explicit forecast limitations."
            )
        elif status == "review":
            workflows.append(
                "Forecasting can be tested, but short history may reduce stability."
            )
        else:
            workflows.append(message)

    if not workflows:
        workflows.append(
            "No workflow is ready for automatic training yet; confirm target and improve data readiness first."
        )

    return {
        "checks": checks,
        "regression_targets": regression_targets,
        "classification_targets": classification_targets,
        "workflows": workflows,
        "requires_target_confirmation": bool(
            regression_targets or classification_targets
        ),
    }
