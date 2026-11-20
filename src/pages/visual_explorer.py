"""Interactive, data-driven charts for the Visual Explorer route."""

import pandas as pd
import plotly.express as px
from dash import dcc, html
from src.time_series import granularity_label, period_display, period_starts


DIMENSIONS = {
    "Country": ("Country", "Country"),
    "Time period": ("Time period", "InvoiceDate"),
    "Product": ("Product", "Description"),
    "Customer": ("Customer", "CustomerID"),
    "Stock code": ("Stock code", "StockCode"),
}

MEASURES = {
    "Revenue": ("Revenue", "Revenue", True),
    "Quantity": ("Units sold", "Quantity", True),
    "UnitPrice": ("Unit price", "UnitPrice", True),
    "InvoiceNo": ("Invoice number", "InvoiceNo", False),
    "CustomerID": ("Customer ID", "CustomerID", False),
}

AGGREGATIONS = {
    "sum": "Sum",
    "count": "Count",
    "mean": "Mean",
    "median": "Median",
    "min": "Minimum",
    "max": "Maximum",
    "nunique": "Distinct count",
}

DEFAULT_AGGREGATIONS = {
    "Revenue": "sum",
    "Quantity": "sum",
    "UnitPrice": "mean",
    "InvoiceNo": "nunique",
    "CustomerID": "nunique",
}

FILTER_COLUMNS = (
    "Country",
    "InvoiceDate",
    "Description",
    "StockCode",
    "CustomerID",
)

CHART_TYPES = {
    "bar": "Bar chart",
    "pie": "Donut chart",
    "line": "Line chart",
}

MAX_CATEGORIES = 15


def build_visual_explorer_page():
    """Create the controls and chart container for visual exploration."""
    return html.Div(
        className="main-content rp-data-workspace rp-visual-explorer",
        children=[
            html.Section(
                className="rp-overview-hero",
                children=[
                    html.P(
                        "EXPLORE  /  VISUAL EXPLORER",
                        className="rp-eyebrow",
                    ),
                    html.H1("Explore the patterns in your data."),
                    html.P(
                        "Explore the built-in retail dataset or analyze an "
                        "uploaded table with type-aware statistics, group "
                        "comparisons, correlations, and date trends.",
                        className="rp-overview-lede",
                    ),
                ],
            ),
            html.Section(
                id="visual-explorer-retail-panel",
                className="rp-explorer-panel",
                **{"aria-labelledby": "visual-explorer-heading"},
                children=[
                    html.Div(
                        className="rp-section-heading",
                        children=[
                            html.Div(
                                children=[
                                    html.P(
                                        "CHART BUILDER",
                                        className="rp-eyebrow",
                                    ),
                                    html.H2(
                                        "Choose what to compare",
                                        id="visual-explorer-heading",
                                    ),
                                ]
                            ),
                            html.P(
                                "Category charts show the top 15 groups by "
                                "the selected measure. Time charts support "
                                "daily or Monday-start weekly periods.",
                                className="rp-section-description",
                            ),
                        ],
                    ),
                    html.Div(
                        className="rp-analysis-workspace",
                        children=[
                            html.Aside(
                                className="rp-analysis-sidebar",
                                children=[
                                    html.Div(
                                        [
                                            html.P(
                                                "SET UP ANALYSIS",
                                                className="rp-eyebrow",
                                            ),
                                            html.H3("Chart controls"),
                                            html.P(
                                                "Adjust the grouping and "
                                                "metric to update the chart."
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        className="rp-explorer-controls",
                                        children=[
                                            _dropdown(
                                                "Group by",
                                                "visual-explorer-dimension",
                                                [
                                                    {
                                                        "label": label,
                                                        "value": value,
                                                    }
                                                    for value, (label, _) in DIMENSIONS.items()
                                                ],
                                                "Time period",
                                            ),
                                            _dropdown(
                                                "Time granularity",
                                                "visual-explorer-time-granularity",
                                                [
                                                    {
                                                        "label": "Daily",
                                                        "value": "daily",
                                                    },
                                                    {
                                                        "label": "Weekly (Monday start)",
                                                        "value": "weekly",
                                                    },
                                                ],
                                                "daily",
                                            ),
                                            _dropdown(
                                                "Measure",
                                                "visual-explorer-measure",
                                                [
                                                    {
                                                        "label": label,
                                                        "value": value,
                                                    }
                                                    for value, (label, _, _) in MEASURES.items()
                                                ],
                                                "Revenue",
                                            ),
                                            _dropdown(
                                                "Aggregation",
                                                "visual-explorer-aggregation",
                                                [
                                                    {
                                                        "label": label,
                                                        "value": value,
                                                    }
                                                    for value, label in AGGREGATIONS.items()
                                                ],
                                                "sum",
                                            ),
                                            _dropdown(
                                                "Chart style",
                                                "visual-explorer-chart-type",
                                                [
                                                    {
                                                        "label": label,
                                                        "value": value,
                                                    }
                                                    for value, label in CHART_TYPES.items()
                                                ],
                                                "line",
                                            ),
                                            html.Div(
                                                className="rp-explorer-field",
                                                children=[
                                                    html.Label(
                                                        "Filter column",
                                                        htmlFor=(
                                                            "visual-explorer-filter-column"
                                                        ),
                                                    ),
                                                    dcc.Dropdown(
                                                        id="visual-explorer-filter-column",
                                                        options=[
                                                            {
                                                                "label": "No filter",
                                                                "value": "",
                                                            }
                                                        ]
                                                        + [
                                                            {
                                                                "label": column,
                                                                "value": column,
                                                            }
                                                            for column in FILTER_COLUMNS
                                                        ],
                                                        value="",
                                                        clearable=False,
                                                        searchable=False,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="rp-explorer-field",
                                                children=[
                                                    html.Label(
                                                        "Match value",
                                                        htmlFor=(
                                                            "visual-explorer-filter-value"
                                                        ),
                                                    ),
                                                    dcc.Input(
                                                        id="visual-explorer-filter-value",
                                                        type="text",
                                                        value="",
                                                        placeholder=(
                                                            "Optional text match"
                                                        ),
                                                        debounce=True,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="rp-upload-actions",
                                        children=[
                                            html.Button(
                                                "Download grouped results",
                                                id="visual-explorer-export-button",
                                                className="rp-primary-action",
                                            ),
                                            dcc.Download(
                                                id="visual-explorer-download"
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="rp-analysis-canvas",
                                children=[
                                    html.Div(
                                        className="rp-chart-card",
                                        children=[
                                            html.Div(
                                                className=(
                                                    "rp-chart-card-header"
                                                ),
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.P(
                                                                "LIVE VIEW",
                                                                className="rp-eyebrow",
                                                            ),
                                                            html.H3(
                                                                "Sales performance"
                                                            ),
                                                        ]
                                                    ),
                                                    html.Span(
                                                        "Interactive",
                                                        className=(
                                                            "rp-status-pill"
                                                        ),
                                                    ),
                                                ],
                                            ),
                                            html.P(
                                                id="visual-explorer-summary",
                                                className="rp-explorer-summary",
                                                role="status",
                                                **{"aria-live": "polite"},
                                            ),
                                            dcc.Graph(
                                                id="visual-explorer-chart",
                                                className="rp-explorer-chart",
                                                config={
                                                    "displaylogo": False,
                                                    "responsive": True,
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className=(
                                            "rp-analysis-detail-card "
                                            "rp-explorer-preview"
                                        ),
                                        children=[
                                            html.H3(
                                                "Rows behind this view"
                                            ),
                                            html.Div(
                                                id="visual-explorer-preview"
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            build_universal_analytics_panel(),
        ],
    )


def build_universal_analytics_panel():
    """Create the generic analysis controls and evidence panels."""
    return html.Section(
        id="universal-analytics-panel",
        className="rp-explorer-panel rp-hidden",
        **{"aria-labelledby": "universal-analytics-heading"},
        children=[
            html.Div(
                className="rp-section-heading",
                children=[
                    html.Div(
                        children=[
                            html.P(
                                "UNIVERSAL ANALYTICS",
                                className="rp-eyebrow",
                            ),
                            html.H2(
                                "Analyze the uploaded dataset",
                                id="universal-analytics-heading",
                            ),
                        ]
                    ),
                    html.P(
                        "Column types are detected from values. Choose "
                        "grouping and aggregation explicitly; suggestions "
                        "are descriptive, not business conclusions.",
                        className="rp-section-description",
                    ),
                ],
            ),
            html.Div(
                className="rp-analysis-workspace",
                children=[
                    html.Aside(
                        className="rp-analysis-sidebar",
                        children=[
                            html.Div(
                                [
                                    html.P(
                                        "DATASET ANALYSIS",
                                        className="rp-eyebrow",
                                    ),
                                    html.H3("Build a view"),
                                    html.P(
                                        "Choose fields from the uploaded "
                                        "dataset. Available options adapt to "
                                        "detected column types."
                                    ),
                                ]
                            ),
                            html.Div(
                                className="rp-explorer-controls",
                                children=[
                                    _dropdown(
                                        "Time granularity",
                                        "universal-time-granularity",
                                        [
                                            {
                                                "label": "Daily",
                                                "value": "daily",
                                            },
                                            {
                                                "label": "Weekly (Monday start)",
                                                "value": "weekly",
                                            },
                                        ],
                                        "daily",
                                    ),
                                    _dropdown(
                                        "Group by",
                                        "universal-dimension",
                                        [],
                                        None,
                                    ),
                                    _dropdown(
                                        "Measure",
                                        "universal-measure",
                                        [],
                                        None,
                                    ),
                                    _dropdown(
                                        "Aggregation",
                                        "universal-aggregation",
                                        [
                                            {
                                                "label": label,
                                                "value": value,
                                            }
                                            for value, label in AGGREGATIONS.items()
                                        ],
                                        "count",
                                    ),
                                    _dropdown(
                                        "Chart style",
                                        "universal-chart-type",
                                        [
                                            {
                                                "label": "Bar chart",
                                                "value": "bar",
                                            },
                                            {
                                                "label": "Line chart",
                                                "value": "line",
                                            },
                                        ],
                                        "bar",
                                    ),
                                    html.Div(
                                        className="rp-explorer-field",
                                        children=[
                                            html.Label(
                                                "Filter column",
                                                htmlFor=(
                                                    "universal-filter-column"
                                                ),
                                            ),
                                            dcc.Dropdown(
                                                id="universal-filter-column",
                                                options=[
                                                    {
                                                        "label": "No filter",
                                                        "value": "",
                                                    }
                                                ],
                                                value="",
                                                clearable=False,
                                                searchable=False,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="rp-explorer-field",
                                        children=[
                                            html.Label(
                                                "Match value",
                                                htmlFor=(
                                                    "universal-filter-value"
                                                ),
                                            ),
                                            dcc.Input(
                                                id="universal-filter-value",
                                                type="text",
                                                value="",
                                                placeholder=(
                                                    "Optional literal match"
                                                ),
                                                debounce=True,
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="rp-upload-actions",
                                children=[
                                    html.Button(
                                        "Download grouped results",
                                        id="universal-analytics-export-button",
                                        className="rp-primary-action",
                                    ),
                                    dcc.Download(
                                        id="universal-analytics-download"
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="rp-analysis-canvas",
                        children=[
                            html.Div(
                                className="rp-chart-card",
                                children=[
                                    html.Div(
                                        className="rp-chart-card-header",
                                        children=[
                                            html.Div(
                                                [
                                                    html.P(
                                                        "UNIVERSAL VIEW",
                                                        className="rp-eyebrow",
                                                    ),
                                                    html.H3(
                                                        "Dataset visualization"
                                                    ),
                                                ]
                                            ),
                                            html.Span(
                                                "Type-aware",
                                                className="rp-status-pill",
                                            ),
                                        ],
                                    ),
                                    html.P(
                                        id="universal-chart-summary",
                                        className="rp-explorer-summary",
                                        role="status",
                                        **{"aria-live": "polite"},
                                    ),
                                    dcc.Graph(
                                        id="universal-analytics-chart",
                                        className="rp-explorer-chart",
                                        config={
                                            "displaylogo": False,
                                            "responsive": True,
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                className=(
                                    "rp-analysis-detail-card "
                                    "rp-explorer-preview"
                                ),
                                children=[
                                    html.H3("Rows behind this view"),
                                    html.Div(
                                        id="universal-analytics-preview"
                                    ),
                                ],
                            ),
                            html.Div(
                                id="universal-analysis-plan",
                                className="rp-quality-notice",
                            ),
                            html.Div(
                                className="rp-analysis-detail-card",
                                children=[
                                    html.H3(
                                        "AI insights (deterministic)"
                                    ),
                                    html.P(
                                        "These narrative insights are "
                                        "generated from measured profile "
                                        "evidence and do not use external "
                                        "model calls."
                                    ),
                                    html.Ul(id="universal-ai-insights"),
                                ],
                            ),
                            html.Div(
                                className="rp-analysis-detail-card",
                                children=[
                                    html.H3("Machine Learning suitability"),
                                    html.P(
                                        "This pre-training assessment checks "
                                        "readiness signals and suggests "
                                        "safe baseline workflows."
                                    ),
                                    html.Div(id="universal-ml-suitability"),
                                ],
                            ),
                            html.Div(
                                className="rp-analysis-detail-card",
                                children=[
                                    html.H3(
                                        "Baseline model evaluation"
                                    ),
                                    html.P(
                                        "Confirm a target column, choose an "
                                        "evaluation mode, and run a baseline "
                                        "comparison. Model workflows accept "
                                        "up to 20,000 rows per run."
                                    ),
                                    html.Div(
                                        className="rp-explorer-controls",
                                        children=[
                                            _dropdown(
                                                "Target column",
                                                "universal-ml-target",
                                                [],
                                                None,
                                            ),
                                            _dropdown(
                                                "Evaluation mode",
                                                "universal-ml-task",
                                                [
                                                    {
                                                        "label": "Auto-detect",
                                                        "value": "auto",
                                                    },
                                                    {
                                                        "label": "Regression",
                                                        "value": "regression",
                                                    },
                                                    {
                                                        "label": "Classification",
                                                        "value": "classification",
                                                    },
                                                ],
                                                "auto",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="rp-upload-actions",
                                        children=[
                                            html.Button(
                                                "Run baseline evaluation",
                                                id="universal-ml-run-button",
                                                className="rp-primary-action",
                                            ),
                                        ],
                                    ),
                                    html.Div(id="universal-ml-evaluation"),
                                ],
                            ),
                            html.Div(
                                className="rp-analysis-detail-card",
                                children=[
                                    html.H3(
                                        "Clustering and anomaly analysis"
                                    ),
                                    html.P(
                                        "Choose features explicitly. Results "
                                        "are exploratory signals, not "
                                        "automatic business decisions. Runs "
                                        "are limited to 20,000 rows."
                                    ),
                                    html.Div(
                                        className="rp-explorer-controls",
                                        children=[
                                            html.Div(
                                                className="rp-explorer-field",
                                                children=[
                                                    html.Label(
                                                        "Feature columns",
                                                        htmlFor=(
                                                            "universal-ml-features"
                                                        ),
                                                    ),
                                                    dcc.Dropdown(
                                                        id="universal-ml-features",
                                                        options=[],
                                                        value=[],
                                                        multi=True,
                                                        clearable=True,
                                                        searchable=True,
                                                    ),
                                                ],
                                            ),
                                            _dropdown(
                                                "Unsupervised workflow",
                                                "universal-ml-unsupervised-type",
                                                [
                                                    {
                                                        "label": "K-means clustering",
                                                        "value": "clustering",
                                                    },
                                                    {
                                                        "label": "Anomaly detection",
                                                        "value": "anomaly",
                                                    },
                                                ],
                                                "clustering",
                                            ),
                                            html.Div(
                                                className="rp-explorer-field",
                                                children=[
                                                    html.Label(
                                                        "Number of clusters",
                                                        htmlFor=(
                                                            "universal-ml-clusters"
                                                        ),
                                                    ),
                                                    dcc.Input(
                                                        id="universal-ml-clusters",
                                                        type="number",
                                                        min=2,
                                                        max=10,
                                                        step=1,
                                                        value=3,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="rp-upload-actions",
                                        children=[
                                            html.Button(
                                                "Run unsupervised analysis",
                                                id="universal-ml-unsupervised-run",
                                                className="rp-primary-action",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="universal-ml-unsupervised-results"
                                    ),
                                ],
                            ),
                            html.Div(
                                className="rp-analysis-detail-card",
                                children=[
                                    html.H3("Time-series forecasting"),
                                    html.P(
                                        "Choose a detected date and numeric "
                                        "measure. Models are evaluated on a "
                                        "chronological holdout."
                                    ),
                                    html.Div(
                                        className="rp-explorer-controls",
                                        children=[
                                            _dropdown(
                                                "Date column",
                                                "universal-forecast-date",
                                                [],
                                                None,
                                            ),
                                            _dropdown(
                                                "Numeric measure",
                                                "universal-forecast-measure",
                                                [],
                                                None,
                                            ),
                                            _dropdown(
                                                "Time granularity",
                                                "universal-forecast-granularity",
                                                [
                                                    {
                                                        "label": "Daily",
                                                        "value": "daily",
                                                    },
                                                    {
                                                        "label": "Weekly (Monday start)",
                                                        "value": "weekly",
                                                    },
                                                ],
                                                "daily",
                                            ),
                                            _dropdown(
                                                "Period aggregation",
                                                "universal-forecast-aggregation",
                                                [
                                                    {
                                                        "label": "Sum",
                                                        "value": "sum",
                                                    },
                                                    {
                                                        "label": "Mean",
                                                        "value": "mean",
                                                    },
                                                ],
                                                "sum",
                                            ),
                                            html.Div(
                                                className="rp-explorer-field",
                                                children=[
                                                    html.Label(
                                                        "Forecast horizon (days or weeks)",
                                                        id=(
                                                            "universal-forecast-horizon-label"
                                                        ),
                                                        htmlFor=(
                                                            "universal-forecast-horizon"
                                                        ),
                                                    ),
                                                    dcc.Input(
                                                        id="universal-forecast-horizon",
                                                        type="number",
                                                        min=1,
                                                        max=28,
                                                        step=1,
                                                        value=28,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="rp-upload-actions",
                                        children=[
                                            html.Button(
                                                "Evaluate and forecast",
                                                id="universal-forecast-run",
                                                className="rp-primary-action",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="universal-forecast-results"
                                    ),
                                ],
                            ),
                            html.Div(
                                className="rp-analysis-detail-card",
                                children=[
                                    html.H3("Numeric summary"),
                                    html.Div(
                                        id="universal-analysis-statistics",
                                        className="rp-upload-table-wrap",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="rp-universal-analysis-grid",
                                children=[
                                    html.Div(
                                        className="rp-quality-categories",
                                        children=[
                                            html.H2("Correlation analysis"),
                                            html.P(
                                                "Pearson correlations use "
                                                "paired numeric values; "
                                                "correlation does not imply "
                                                "causation."
                                            ),
                                            dcc.Graph(
                                                id="universal-correlation-chart",
                                                config={
                                                    "displaylogo": False
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="rp-quality-categories",
                                        children=[
                                            html.H2("Date trend"),
                                            html.P(
                                                "When available, this chart "
                                                "shows daily or Monday-start "
                                                "weekly means for the first "
                                                "detected numeric column. "
                                                "Choose the time granularity "
                                                "in the analysis controls."
                                            ),
                                            dcc.Graph(
                                                id="universal-trend-chart",
                                                config={
                                                    "displaylogo": False
                                                },
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _dropdown(label, element_id, options, value):
    return html.Div(
        className="rp-explorer-field",
        children=[
            html.Label(label, htmlFor=element_id),
            dcc.Dropdown(
                id=element_id,
                options=options,
                value=value,
                clearable=False,
                searchable=False,
            ),
        ],
    )


def filter_sales(sales, filter_column=None, filter_value=None):
    """Filter active transactions using a literal, case-insensitive match."""
    if not filter_column or not filter_value:
        return sales
    if filter_column not in FILTER_COLUMNS or filter_column not in sales.columns:
        raise ValueError(f"Unsupported visual-explorer filter: {filter_column}")

    matches = sales[filter_column].astype("string").str.contains(
        str(filter_value),
        case=False,
        regex=False,
        na=False,
    )
    return sales.loc[matches].copy()


def aggregate_sales(
    sales,
    dimension,
    measure,
    aggregation=None,
    time_granularity="daily",
):
    """Aggregate sales into chart-ready groups, limiting category cardinality."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"Unsupported visual-explorer dimension: {dimension}")
    if measure not in MEASURES:
        raise ValueError(f"Unsupported visual-explorer measure: {measure}")
    aggregation = aggregation or DEFAULT_AGGREGATIONS[measure]
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"Unsupported visual-explorer aggregation: {aggregation}"
        )
    if not isinstance(sales, pd.DataFrame):
        raise TypeError("Sales data must be a pandas DataFrame.")
    if sales.empty:
        return pd.DataFrame(columns=["Category", "Value"])

    _, dimension_column = DIMENSIONS[dimension]
    _, measure_column, is_numeric = MEASURES[measure]
    required_columns = {dimension_column, measure_column}
    missing_columns = required_columns.difference(sales.columns)
    if missing_columns:
        raise ValueError(
            "Sales data is missing columns required for this chart: "
            f"{', '.join(sorted(missing_columns))}"
        )

    data = sales[[dimension_column, measure_column]].copy()
    if dimension == "Time period":
        data[dimension_column] = period_starts(
            data[dimension_column],
            time_granularity,
        )
        data = data.dropna(subset=[dimension_column])
    else:
        data[dimension_column] = (
            data[dimension_column].fillna("Unknown").astype(str)
        )

    if aggregation in {"sum", "mean", "median", "min", "max"}:
        if not is_numeric:
            raise ValueError(
                f"{AGGREGATIONS[aggregation]} requires a numeric measure."
            )
        data[measure_column] = pd.to_numeric(
            data[measure_column],
            errors="coerce",
        )
        data = data.dropna(subset=[measure_column])
    else:
        data = data.dropna(subset=[measure_column])

    grouped = (
        data.groupby(dimension_column, dropna=False)
        .agg(Value=(measure_column, aggregation))
        .reset_index()
        .rename(columns={dimension_column: "Category"})
    )
    if dimension == "Time period":
        grouped = grouped.sort_values("Category")
        grouped["Category"] = grouped["Category"].map(
            lambda value: period_display(value, time_granularity)
        )
    else:
        grouped = grouped.sort_values(
            "Value",
            ascending=False,
            kind="stable",
        ).head(MAX_CATEGORIES)

    grouped = grouped.reset_index(drop=True)
    return grouped


def build_visual_explorer_figure(
    sales,
    dimension,
    measure,
    chart_type,
    aggregation=None,
    time_granularity="daily",
):
    """Build a chart after grouping the selected active sales data."""
    grouped = aggregate_sales(
        sales,
        dimension,
        measure,
        aggregation,
        time_granularity,
    )
    return build_visual_explorer_chart(
        grouped,
        dimension,
        measure,
        chart_type,
        aggregation or DEFAULT_AGGREGATIONS[measure],
        time_granularity,
    )


def build_visual_explorer_chart(
    grouped,
    dimension,
    measure,
    chart_type,
    aggregation,
    time_granularity="daily",
):
    """Build a Plotly chart from grouped data."""
    if chart_type not in CHART_TYPES:
        raise ValueError(f"Unsupported visual-explorer chart type: {chart_type}")

    dimension_label = (
        (
            "Date"
            if time_granularity == "daily"
            else "Week starting Monday"
        )
        if dimension == "Time period"
        else DIMENSIONS[dimension][0]
    )
    measure_label = MEASURES[measure][0]
    aggregation_label = AGGREGATIONS[aggregation]
    title = (
        f"{aggregation_label} {measure_label} by "
        f"{dimension_label} ({granularity_label(time_granularity)})"
        if dimension == "Time period"
        else f"{aggregation_label} {measure_label} by {dimension_label}"
    )

    if chart_type == "pie":
        figure = px.pie(
            grouped,
            names="Category",
            values="Value",
            hole=0.56,
            title=title,
        )
        figure.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )
        if measure in {"Revenue", "UnitPrice"}:
            figure.update_traces(
                hovertemplate="%{label}<br>£%{value:,.2f}<extra></extra>"
            )
    elif chart_type == "line":
        figure = px.line(
            grouped,
            x="Category",
            y="Value",
            markers=True,
            title=title,
            labels={"Category": dimension_label, "Value": measure_label},
        )
        figure.update_traces(
            line={"width": 3, "color": "#58c7ff"},
            marker={
                "size": 8,
                "color": "#58c7ff",
                "line": {"width": 2, "color": "#e9f7ff"},
            },
        )
    elif dimension == "Time period":
        figure = px.bar(
            grouped,
            x="Category",
            y="Value",
            title=title,
            labels={"Category": dimension_label, "Value": measure_label},
        )
    else:
        figure = px.bar(
            grouped,
            x="Value",
            y="Category",
            orientation="h",
            title=title,
            labels={"Category": dimension_label, "Value": measure_label},
        )
        figure.update_yaxes(categoryorder="total ascending")

    if measure in {"Revenue", "UnitPrice"} and chart_type in {"bar", "line"}:
        if dimension == "Time period" or chart_type == "line":
            figure.update_yaxes(tickprefix="£")
        else:
            figure.update_xaxes(tickprefix="£")

    figure.update_layout(
        height=470,
        margin={"l": 18, "r": 18, "t": 58, "b": 22},
        legend_title_text=dimension_label,
    )
    return figure


def build_filtered_preview(sales, row_limit=10):
    """Render a bounded preview of filtered transactions."""
    if sales.empty:
        return html.P("No transactions match the current filter.")

    columns = list(sales.columns[:8])
    return html.Div(
        className="rp-upload-table-wrap",
        children=html.Table(
            [
                html.Thead(html.Tr([html.Th(column) for column in columns])),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(str(value))
                                for value in row
                            ]
                        )
                        for row in sales.loc[:, columns]
                        .head(row_limit)
                        .itertuples(index=False, name=None)
                    ]
                ),
            ]
        ),
    )
