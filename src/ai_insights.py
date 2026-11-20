"""Deterministic, evidence-based insight generation for uploaded datasets."""

import pandas as pd


def _format_columns(columns):
    if not columns:
        return ""
    if len(columns) == 1:
        return columns[0]
    if len(columns) == 2:
        return f"{columns[0]} and {columns[1]}"
    return f"{', '.join(columns[:-1])}, and {columns[-1]}"


def generate_deterministic_ai_insights(profile, dataset_name):
    """Summarize profiled evidence into findings, recommendations, limits."""
    summary = profile["summary"]
    columns = profile["columns"]
    rows = int(summary["rows"])
    column_count = int(summary["columns"])
    issue_count = int(summary["issue_count"])
    duplicate_rows = int(summary["duplicate_rows"])
    missing_cells = int(summary["missing_cells"])
    rows_with_missing = int(summary["rows_with_missing"])
    date_coverage = summary.get("date_coverage")

    numeric_columns = [
        item for item in columns if item["detected_type"] == "numeric"
    ]
    categorical_columns = [
        item for item in columns if item["detected_type"] == "categorical"
    ]
    date_columns = [
        item for item in columns if item["detected_type"] == "date"
    ]
    outlier_columns = [
        item["name"] for item in numeric_columns if item["outlier_count"] > 0
    ]
    high_missing_columns = [
        item["name"]
        for item in columns
        if item["missing_percent"] >= 20
    ]
    high_cardinality_columns = [
        item["name"] for item in columns if item["high_cardinality"]
    ]

    findings = [
        (
            f"Dataset profile for {dataset_name}: {rows:,} rows and "
            f"{column_count:,} columns."
        )
    ]
    if issue_count == 0:
        findings.append(
            "No data-quality issues were flagged by the current checks."
        )
    else:
        findings.append(
            f"{issue_count:,} quality issue(s) were flagged during profiling."
        )
    if duplicate_rows:
        findings.append(
            f"{duplicate_rows:,} exact duplicate row(s) were detected."
        )
    if missing_cells:
        findings.append(
            f"{missing_cells:,} missing cell(s) were detected across "
            f"{rows_with_missing:,} row(s)."
        )
    if high_missing_columns:
        findings.append(
            "The highest missing-value impact is in "
            f"{_format_columns(high_missing_columns[:3])}."
        )
    if outlier_columns:
        findings.append(
            "Potential IQR outliers were detected in "
            f"{_format_columns(outlier_columns[:3])}."
        )
    if date_coverage:
        earliest = pd.to_datetime(date_coverage["earliest"]).date().isoformat()
        latest = pd.to_datetime(date_coverage["latest"]).date().isoformat()
        findings.append(
            f"Detected date coverage spans from {earliest} to {latest}."
        )
    if categorical_columns:
        top_category_column = categorical_columns[0]
        top_values = top_category_column.get("top_values", [])
        if top_values and top_category_column["non_missing_count"] > 0:
            top = top_values[0]
            share = (
                100
                * top["count"]
                / max(top_category_column["non_missing_count"], 1)
            )
            findings.append(
                f"Most frequent value in {top_category_column['name']}: "
                f"{top['value']} ({share:.1f}% of non-missing rows)."
            )

    recommendations = []
    if duplicate_rows:
        recommendations.append(
            "Review duplicate records and remove them only after confirming "
            "they are not valid repeated events."
        )
    if high_missing_columns:
        recommendations.append(
            "Prioritize missing-value treatment for "
            f"{_format_columns(high_missing_columns[:3])}."
        )
    elif missing_cells:
        recommendations.append(
            "Review missing-value treatment strategy before relying on "
            "comparative metrics."
        )
    if outlier_columns:
        recommendations.append(
            "Investigate outlier candidates with domain context before "
            "excluding or winsorizing values."
        )
    if high_cardinality_columns:
        recommendations.append(
            "Avoid direct group charts on high-cardinality fields such as "
            f"{_format_columns(high_cardinality_columns[:2])}; aggregate or "
            "bucket first."
        )
    if len(numeric_columns) >= 2:
        recommendations.append(
            "Use correlation and grouped summaries to test associations "
            "between numeric columns."
        )
    if date_columns and numeric_columns:
        recommendations.append(
            "Use daily or weekly trend views before selecting a forecasting "
            "model."
        )
    if not recommendations:
        recommendations.append(
            "Continue with grouped comparisons and trend checks to identify "
            "stable patterns before making decisions."
        )

    limitations = [
        "These AI insights are deterministic rule-based summaries from the "
        "current profile and do not infer causation.",
        "Recommendations are analytical guidance, not business decisions.",
    ]
    if rows < 100:
        limitations.append(
            "The current sample size is limited; validate key conclusions "
            "against a larger dataset when possible."
        )
    if not numeric_columns:
        limitations.append(
            "No numeric columns were detected, so numerical modeling "
            "workflows are currently limited."
        )
    if not date_columns:
        limitations.append(
            "No usable date column was detected, so time-series insights are "
            "not currently available."
        )

    return {
        "findings": findings,
        "recommendations": recommendations,
        "limitations": limitations,
    }
