"""Statistical and visualization helpers for arbitrary tabular datasets."""

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_profiling import profile_dataset
from src.pages.data_workspace import deserialize_workspace_dataset


ROW_COUNT_MEASURE = "__row_count__"
AGGREGATIONS = {
    "sum": "Sum",
    "count": "Count",
    "mean": "Mean",
    "median": "Median",
    "min": "Minimum",
    "max": "Maximum",
    "nunique": "Distinct count",
}
NUMERIC_AGGREGATIONS = {"sum", "mean", "median", "min", "max"}
MAX_GROUPS = 15


@lru_cache(maxsize=2)
def profile_workspace_payload(encoded_payload):
    """Profile a session payload once while the user explores its charts."""
    return profile_dataset(deserialize_workspace_dataset(encoded_payload))


def analysis_control_options(profile):
    """Build type-aware grouping, measure, and filter options."""
    columns = profile["columns"]
    dimensions = [
        {"label": f"{item['name']} ({item['detected_type']})",
         "value": item["name"]}
        for item in columns
    ]
    numeric_columns = [
        item["name"]
        for item in columns
        if item["detected_type"] == "numeric"
    ]
    measures = [
        {"label": "Rows", "value": ROW_COUNT_MEASURE},
        *[
            {
                "label": f"{item['name']} ({item['detected_type']})",
                "value": item["name"],
            }
            for item in columns
        ],
    ]
    date_column = next(
        (
            item["name"]
            for item in columns
            if item["detected_type"] == "date"
        ),
        None,
    )
    dimension = date_column or next(
        (
            item["name"]
            for item in columns
            if item["detected_type"] == "categorical"
        ),
        dimensions[0]["value"] if dimensions else None,
    )
    measure = numeric_columns[0] if numeric_columns else ROW_COUNT_MEASURE
    filter_options = [{"label": "No filter", "value": ""}] + [
        {"label": item["name"], "value": item["name"]}
        for item in columns
    ]
    return {
        "dimension_options": dimensions,
        "dimension": dimension,
        "measure_options": measures,
        "measure": measure,
        "filter_options": filter_options,
        "numeric_columns": numeric_columns,
        "date_columns": [
            item["name"]
            for item in columns
            if item["detected_type"] == "date"
        ],
    }


def filter_dataset(data, filter_column=None, filter_value=None):
    """Filter any uploaded column using a literal, case-insensitive match."""
    if not filter_column or filter_value is None or str(filter_value) == "":
        return data
    if filter_column not in data.columns:
        raise ValueError(f"Unknown filter column: {filter_column}")
    mask = data[filter_column].astype("string").str.contains(
        str(filter_value),
        case=False,
        regex=False,
        na=False,
    )
    return data.loc[mask].copy()


def aggregate_dataset(
    data,
    dimension,
    measure,
    aggregation,
    numeric_columns=(),
    date_columns=(),
):
    """Group a generic dataset using an explicit, type-valid aggregation."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Analytics data must be a pandas DataFrame.")
    if dimension not in data.columns:
        raise ValueError(f"Unknown grouping column: {dimension}")
    if measure != ROW_COUNT_MEASURE and measure not in data.columns:
        raise ValueError(f"Unknown measure column: {measure}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    if measure == ROW_COUNT_MEASURE and aggregation != "count":
        raise ValueError("Row counts support only the Count aggregation.")
    if (
        aggregation in NUMERIC_AGGREGATIONS
        and measure not in numeric_columns
    ):
        raise ValueError(
            f"{AGGREGATIONS[aggregation]} requires a numeric measure."
        )
    if data.empty:
        return pd.DataFrame(columns=["Category", "Value"])

    columns = [dimension]
    if measure != ROW_COUNT_MEASURE and measure != dimension:
        columns.append(measure)
    working = data.loc[:, columns].copy()
    if dimension in date_columns:
        dates = pd.to_datetime(
            working[dimension],
            errors="coerce",
            format="mixed",
            utc=True,
        )
        groups = (
            dates.dt.tz_convert(None)
            .dt.to_period("M")
            .astype("string")
            .fillna("Missing")
        )
    else:
        groups = working[dimension].astype("string").fillna("Missing")
    working["_group"] = groups

    if measure == ROW_COUNT_MEASURE:
        grouped = working.groupby("_group", dropna=False).size()
    else:
        values = working[measure]
        if aggregation in NUMERIC_AGGREGATIONS:
            values = pd.to_numeric(values, errors="coerce")
            values = values.where(np.isfinite(values), np.nan)
        working["_value"] = values
        group_by = working.groupby("_group", dropna=False)["_value"]
        if aggregation == "count":
            grouped = group_by.count()
        elif aggregation == "nunique":
            grouped = group_by.nunique()
        else:
            grouped = group_by.agg(aggregation)

    result = (
        grouped.rename("Value")
        .reset_index()
        .rename(columns={"_group": "Category"})
    )
    if dimension in date_columns:
        result = result.sort_values("Category", kind="stable")
    else:
        result = result.sort_values(
            "Value",
            ascending=False,
            kind="stable",
        ).head(MAX_GROUPS)
    return result.reset_index(drop=True)


def build_universal_figure(
    grouped,
    dimension,
    measure,
    aggregation,
    chart_type="bar",
    date_columns=(),
):
    """Build an interactive generic bar or time-series chart."""
    if chart_type not in {"bar", "line"}:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    measure_label = (
        "Rows" if measure == ROW_COUNT_MEASURE else str(measure)
    )
    title = f"{AGGREGATIONS[aggregation]} {measure_label} by {dimension}"
    if grouped.empty:
        return build_empty_figure("No records match the current selection.")

    if chart_type == "line":
        if dimension not in date_columns:
            raise ValueError("Line charts require a detected date column.")
        figure = px.line(
            grouped,
            x="Category",
            y="Value",
            markers=True,
            title=title,
            labels={"Category": dimension, "Value": measure_label},
        )
    else:
        figure = px.bar(
            grouped,
            x="Value",
            y="Category",
            orientation="h",
            title=title,
            labels={"Category": dimension, "Value": measure_label},
        )
        figure.update_yaxes(categoryorder="total ascending")
    figure.update_layout(
        height=430,
        margin={"l": 18, "r": 18, "t": 58, "b": 22},
    )
    return figure


def build_empty_figure(message):
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    figure.update_layout(
        height=430,
        margin={"l": 18, "r": 18, "t": 58, "b": 22},
    )
    return figure


def build_correlation_figure(data, profile):
    """Create a Pearson correlation heatmap from detected numeric columns."""
    numeric_columns = [
        item["name"]
        for item in profile["columns"]
        if item["detected_type"] == "numeric"
    ]
    if len(numeric_columns) < 2:
        return build_empty_figure(
            "At least two numeric columns are needed for correlation."
        )
    numeric_data = data[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    correlation = numeric_data.corr(min_periods=2)
    if correlation.count().sum() < 4:
        return build_empty_figure(
            "Not enough paired numeric values to calculate correlation."
        )
    figure = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Numeric correlation (Pearson)",
    )
    figure.update_layout(
        height=420,
        margin={"l": 18, "r": 18, "t": 58, "b": 22},
    )
    return figure


def build_trend_figure(data, profile):
    """Show a monthly mean trend when date and numeric columns are detected."""
    date_column = next(
        (
            item["name"]
            for item in profile["columns"]
            if item["detected_type"] == "date"
        ),
        None,
    )
    measure = next(
        (
            item["name"]
            for item in profile["columns"]
            if item["detected_type"] == "numeric"
        ),
        None,
    )
    if not date_column or not measure:
        return build_empty_figure(
            "A detected date column and numeric measure are needed for a trend."
        )

    dates = pd.to_datetime(
        data[date_column],
        errors="coerce",
        format="mixed",
        utc=True,
    )
    values = pd.to_numeric(data[measure], errors="coerce")
    values = values.where(np.isfinite(values), np.nan)
    trend = pd.DataFrame(
        {
            "Date": dates.dt.tz_convert(None),
            "Value": values,
        }
    ).dropna()
    if trend.empty:
        return build_empty_figure("No valid date and numeric pairs were found.")
    trend["Month"] = trend["Date"].dt.to_period("M").dt.to_timestamp()
    trend = trend.groupby("Month", as_index=False)["Value"].mean()
    figure = px.line(
        trend,
        x="Month",
        y="Value",
        markers=True,
        title=f"Monthly mean of {measure} by {date_column}",
    )
    figure.update_layout(
        height=420,
        margin={"l": 18, "r": 18, "t": 58, "b": 22},
    )
    return figure


def build_analysis_recommendations(profile):
    """Suggest data-supported analyses without guessing business semantics."""
    numeric_columns = [
        item["name"]
        for item in profile["columns"]
        if item["detected_type"] == "numeric"
    ]
    category_columns = [
        item["name"]
        for item in profile["columns"]
        if item["detected_type"] == "categorical"
    ]
    date_columns = [
        item["name"]
        for item in profile["columns"]
        if item["detected_type"] == "date"
    ]
    suggestions = []
    if numeric_columns:
        suggestions.append(
            "Review descriptive statistics and distributions for "
            + ", ".join(numeric_columns[:5])
            + "."
        )
    if category_columns and numeric_columns:
        suggestions.append(
            f"Compare a numeric measure across {category_columns[0]} groups."
        )
    if date_columns and numeric_columns:
        suggestions.append(
            f"Explore a monthly mean trend for {numeric_columns[0]} by "
            f"{date_columns[0]}; select another aggregation if appropriate."
        )
    if len(numeric_columns) >= 2:
        suggestions.append(
            "Inspect numeric correlations as associations, not causal evidence."
        )
    if not suggestions:
        suggestions.append(
            "This dataset has no clearly detected numeric measure yet; "
            "inspect its column types before selecting analyses."
        )
    return suggestions
