"""Reusable, non-destructive profiling and quality checks for datasets."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


MAX_TYPE_INFERENCE_VALUES = 1_000
TOP_VALUE_LIMIT = 5


def _finite_number(value):
    """Return a JSON-safe finite number or None."""
    if pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)


def _infer_string_type(values):
    """Classify text columns without modifying the source values."""
    normalized = values.astype("string").str.strip().str.lower()
    observed = normalized.dropna()
    unique_values = set(observed.unique())
    if unique_values and unique_values.issubset(
        {"true", "false", "yes", "no", "y", "n"}
    ):
        return "boolean"

    sample = values.dropna().head(MAX_TYPE_INFERENCE_VALUES)
    if sample.empty:
        return "unknown"

    numeric = pd.to_numeric(sample, errors="coerce")
    if numeric.notna().mean() >= 0.9:
        return "numeric"

    if not sample.astype("string").str.fullmatch(r"\s*\d+(\.\d+)?\s*").all():
        dates = pd.to_datetime(
            sample,
            errors="coerce",
            format="mixed",
            utc=True,
        )
        if dates.notna().mean() >= 0.8:
            return "date"

    unique_count = int(values.nunique(dropna=True))
    if unique_count <= max(50, int(len(values) * 0.5)):
        return "categorical"
    return "text"


def _detected_type(series):
    if is_bool_dtype(series.dtype):
        return "boolean"
    if is_numeric_dtype(series.dtype):
        return "numeric"
    if is_datetime64_any_dtype(series.dtype):
        return "date"
    return _infer_string_type(series)


def _column_profile(name, series):
    missing_count = int(series.isna().sum())
    non_missing = series.dropna()
    value_count = len(non_missing)
    unique_count = int(non_missing.nunique())
    detected_type = _detected_type(series)

    profile = {
        "name": str(name),
        "storage_type": str(series.dtype),
        "detected_type": detected_type,
        "row_count": len(series),
        "non_missing_count": value_count,
        "missing_count": missing_count,
        "missing_percent": (
            100 * missing_count / len(series) if len(series) else 0.0
        ),
        "unique_count": unique_count,
        "duplicate_values": max(value_count - unique_count, 0),
        "cardinality_percent": (
            100 * unique_count / value_count if value_count else 0.0
        ),
        "constant": bool(value_count and unique_count <= 1),
        "high_cardinality": bool(
            detected_type in {"categorical", "text"}
            and unique_count > 50
            and unique_count / max(value_count, 1) > 0.5
        ),
        "top_values": [],
        "rare_category_count": 0,
        "rare_categories": [],
        "statistics": {},
        "outlier_count": 0,
        "invalid_numeric_count": 0,
        "invalid_date_count": 0,
    }

    if detected_type == "numeric":
        numeric_values = pd.to_numeric(non_missing, errors="coerce")
        values = numeric_values.dropna()
        finite_mask = np.isfinite(values)
        profile["invalid_numeric_count"] = int(len(non_missing) - finite_mask.sum())
        values = values[finite_mask]
        if not values.empty:
            quantiles = values.quantile([0.25, 0.5, 0.75])
            q1, median, q3 = (
                float(quantiles.loc[0.25]),
                float(quantiles.loc[0.5]),
                float(quantiles.loc[0.75]),
            )
            profile["statistics"] = {
                "minimum": _finite_number(values.min()),
                "maximum": _finite_number(values.max()),
                "mean": _finite_number(values.mean()),
                "median": _finite_number(median),
                "standard_deviation": _finite_number(values.std()),
                "q1": _finite_number(q1),
                "q3": _finite_number(q3),
                "skewness": _finite_number(values.skew()),
                "kurtosis": _finite_number(values.kurt()),
            }
            iqr = q3 - q1
            if len(values) >= 4 and iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                profile["outlier_count"] = int(
                    ((values < lower_bound) | (values > upper_bound)).sum()
                )
    elif detected_type == "date":
        dates = pd.to_datetime(non_missing, errors="coerce", format="mixed", utc=True)
        profile["invalid_date_count"] = int(dates.isna().sum())
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            profile["statistics"] = {
                "earliest": valid_dates.min().isoformat(),
                "latest": valid_dates.max().isoformat(),
            }
    elif detected_type in {"categorical", "boolean"}:
        value_counts = non_missing.astype("string").value_counts()
        counts = value_counts.head(TOP_VALUE_LIMIT)
        profile["top_values"] = [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
        ]
        rare_categories = value_counts[value_counts == 1]
        profile["rare_category_count"] = int(len(rare_categories))
        profile["rare_categories"] = [
            str(value) for value in rare_categories.index[:TOP_VALUE_LIMIT]
        ]

    sample_values = non_missing.head(3).astype("string").fillna("")
    profile["sample_values"] = [str(value) for value in sample_values]
    return profile


def _make_issue(
    issue_type,
    description,
    column,
    affected_rows,
    row_count,
    severity,
    recommendation,
    automatic_correction_safe=False,
):
    return {
        "type": issue_type,
        "description": description,
        "column": column,
        "affected_rows": int(affected_rows),
        "affected_percent": (
            100 * affected_rows / row_count if row_count else 0.0
        ),
        "severity": severity,
        "recommendation": recommendation,
        "automatic_correction_safe": automatic_correction_safe,
    }


def profile_dataset(data):
    """Return dataset, column, and quality details without changing ``data``."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Profiling requires a pandas DataFrame.")
    if data.empty or data.shape[1] == 0:
        raise ValueError("Profiling requires a dataset with rows and columns.")

    row_count, column_count = data.shape
    columns = [
        _column_profile(name, data[name])
        for name in data.columns
    ]
    duplicate_rows = int(data.duplicated().sum())
    rows_with_missing = int(data.isna().any(axis=1).sum())
    missing_cells = int(data.isna().sum().sum())
    issues = []
    recommendations = []

    if duplicate_rows:
        issues.append(
            _make_issue(
                "duplicate_rows",
                "Exact duplicate records were found.",
                None,
                duplicate_rows,
                row_count,
                "medium",
                "Review duplicate records before deciding whether to remove them.",
            )
        )
        recommendations.append(
            "Review exact duplicate records; the original dataset has not been changed."
        )

    for column in columns:
        name = column["name"]
        missing_count = column["missing_count"]
        if missing_count:
            severity = "high" if column["missing_percent"] >= 50 else "medium"
            issues.append(
                _make_issue(
                    "missing_values",
                    f"{column['missing_percent']:.1f}% of values are missing.",
                    name,
                    missing_count,
                    row_count,
                    severity,
                    "Review the cause and choose whether to retain, correct, or impute missing values.",
                )
            )
            recommendations.append(
                f"Review missing values in {name}; no values were filled automatically."
            )

        if column["outlier_count"]:
            issues.append(
                _make_issue(
                    "potential_outliers",
                    "Values outside the 1.5 x IQR range were detected.",
                    name,
                    column["outlier_count"],
                    row_count,
                    "medium",
                    "Inspect the flagged values in context; an outlier is not necessarily an error.",
                )
            )
            recommendations.append(
                f"Inspect potential IQR outliers in {name} before applying any correction."
            )

        if column["invalid_date_count"]:
            invalid_count = column["invalid_date_count"]
            issues.append(
                _make_issue(
                    "invalid_dates",
                    "Some values in a date-like column could not be parsed.",
                    name,
                    invalid_count,
                    row_count,
                    "medium",
                    "Check the source date format and correct values only when the intended date is known.",
                )
            )
            recommendations.append(
                f"Check date formatting in {name}; invalid values were not replaced."
            )

        if column["invalid_numeric_count"]:
            invalid_count = column["invalid_numeric_count"]
            issues.append(
                _make_issue(
                    "invalid_numeric_values",
                    "Some values in a numeric-like column are not finite numbers.",
                    name,
                    invalid_count,
                    row_count,
                    "medium",
                    "Check the source values and units; invalid entries were not coerced or replaced.",
                )
            )
            recommendations.append(
                f"Review non-numeric or non-finite values in {name} before numerical analysis."
            )

        if column["constant"]:
            issues.append(
                _make_issue(
                    "constant_column",
                    "The column contains at most one distinct non-missing value.",
                    name,
                    row_count - missing_count,
                    row_count,
                    "low",
                    "Confirm whether this column is needed for downstream analysis.",
                )
            )
            recommendations.append(
                f"Confirm whether constant column {name} is needed for analysis."
            )

        if column["high_cardinality"]:
            issues.append(
                _make_issue(
                    "high_cardinality",
                    "Most non-missing values are distinct.",
                    name,
                    column["non_missing_count"],
                    row_count,
                    "low",
                    "Check whether these values are identifiers before using them as categories.",
                )
            )
            recommendations.append(
                f"Check whether {name} contains identifiers before treating it as a category."
            )

    dates = [
        column["statistics"]
        for column in columns
        if column["detected_type"] == "date"
        and column["statistics"].get("earliest")
    ]
    date_coverage = None
    if dates:
        date_coverage = {
            "earliest": min(item["earliest"] for item in dates),
            "latest": max(item["latest"] for item in dates),
        }

    if not issues:
        recommendations.append(
            "No missing values, exact duplicate rows, or IQR outliers were detected."
        )

    return {
        "summary": {
            "rows": row_count,
            "columns": column_count,
            "dimensions": f"{row_count:,} rows x {column_count:,} columns",
            "memory_bytes": int(data.memory_usage(index=True, deep=True).sum()),
            "duplicate_rows": duplicate_rows,
            "missing_cells": missing_cells,
            "rows_with_missing": rows_with_missing,
            "date_coverage": date_coverage,
            "issue_count": len(issues),
        },
        "columns": columns,
        "issues": issues,
        "recommendations": recommendations,
    }
