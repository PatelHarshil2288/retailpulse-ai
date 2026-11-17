"""Data-quality views for the active RetailPulse dataset."""

from dash import html


QUALITY_METRICS = (
    ("Rows received", "rows"),
    ("Duplicate rows", "duplicate_rows"),
    ("Missing descriptions", "missing_descriptions"),
    ("Missing customer IDs", "missing_customer_ids"),
    ("Cancellation rows", "cancellation_invoice_rows"),
    ("Negative quantities", "negative_quantity_rows"),
    ("Zero quantities", "zero_quantity_rows"),
    ("Negative prices", "negative_price_rows"),
    ("Zero-price rows", "zero_price_rows"),
)


def build_quality_lab_page():
    """Create the route container for the active dataset's quality report."""
    return html.Div(
        className="main-content rp-data-workspace",
        children=[
            html.Section(
                className="rp-overview-hero",
                children=[
                    html.P(
                        "DATA QUALITY  /  DIAGNOSTICS",
                        className="rp-eyebrow",
                    ),
                    html.H1("Understand the data behind your analysis."),
                    html.P(
                        "Review detected column types, missing values, "
                        "duplicates, potential outliers, date coverage, and "
                        "recommendations. Uploaded data is never changed "
                        "automatically.",
                        className="rp-overview-lede",
                    ),
                ],
            ),
            html.Div(id="quality-lab-content"),
        ],
    )


def build_dataset_quality_content(profile, filename, sheet_name=None):
    """Render a generic dataset profile and transparent quality findings."""
    summary = profile["summary"]
    outlier_count = sum(
        column["outlier_count"] for column in profile["columns"]
    )
    memory_mib = summary["memory_bytes"] / (1024 * 1024)
    cards = [
        ("Rows", summary["rows"]),
        ("Columns", summary["columns"]),
        ("Missing cells", summary["missing_cells"]),
        ("Rows with missing values", summary["rows_with_missing"]),
        ("Exact duplicate rows", summary["duplicate_rows"]),
        ("Potential IQR outliers", outlier_count),
    ]
    date_coverage_text = ""
    if summary["date_coverage"]:
        date_coverage_text = (
            "Detected date coverage: "
            f"{summary['date_coverage']['earliest']} to "
            f"{summary['date_coverage']['latest']}."
        )

    column_rows = []
    for column in profile["columns"]:
        statistics = column["statistics"]
        if column["detected_type"] == "numeric":
            detail = (
                f"min {statistics.get('minimum')}; "
                f"median {statistics.get('median')}; "
                f"max {statistics.get('maximum')}; "
                f"outliers {column['outlier_count']:,}; "
                f"invalid {column['invalid_numeric_count']:,}"
            )
        elif column["detected_type"] == "date":
            detail = (
                f"{statistics.get('earliest', 'No valid dates')} to "
                f"{statistics.get('latest', 'No valid dates')}"
            )
        else:
            detail = (
                ", ".join(
                    f"{item['value']} ({item['count']:,})"
                    for item in column["top_values"]
                )
                + (
                    f"; {column['rare_category_count']:,} singleton categories"
                    if column["detected_type"] == "categorical"
                    else ""
                )
            )
        column_rows.append(
            html.Tr(
                [
                    html.Td(column["name"]),
                    html.Td(
                        f"{column['detected_type']} "
                        f"({column['storage_type']})"
                    ),
                    html.Td(f"{column['unique_count']:,}"),
                    html.Td(
                        f"{column['missing_count']:,} "
                        f"({column['missing_percent']:.1f}%)"
                    ),
                    html.Td(detail or "-"),
                ]
            )
        )

    issue_rows = [
        html.Tr(
            [
                html.Td(issue["severity"].title()),
                html.Td(issue["description"]),
                html.Td(issue["column"] or "Dataset"),
                html.Td(
                    f"{issue['affected_rows']:,} "
                    f"({issue['affected_percent']:.1f}%)"
                ),
                html.Td(issue["recommendation"]),
                html.Td(
                    "No - requires review"
                    if not issue["automatic_correction_safe"]
                    else "Yes"
                ),
            ]
        )
        for issue in profile["issues"]
    ]
    if not issue_rows:
        issue_rows = [
            html.Tr(
                [
                    html.Td("No flagged issues", colSpan=6),
                ]
            )
        ]

    heading = f"Profile and quality report: {filename}"
    if sheet_name:
        heading = f"{heading} ({sheet_name})"

    return html.Section(
        className="rp-quality-panel",
        children=[
            html.Div(
                className="rp-quality-source",
                children=[
                    html.Strong(heading),
                    html.Span(
                        f"Dataset dimensions: {summary['dimensions']} "
                        f"| Approx. memory: {memory_mib:.2f} MiB"
                    ),
                    html.Span(
                        "Type detection and IQR outliers are screening hints. "
                        "The uploaded source is not modified automatically."
                    ),
                    html.Span(date_coverage_text),
                ],
            ),
            html.Div(
                className="rp-upload-metrics",
                children=[
                    _metric_card(label, value)
                    for label, value in cards
                ],
            ),
            html.Div(
                className="rp-quality-panel rp-quality-categories",
                children=[
                    html.H2("Column profile"),
                    html.Div(
                        className="rp-upload-table-wrap",
                        children=html.Table(
                            [
                                html.Thead(
                                    html.Tr(
                                        [
                                            html.Th("Column"),
                                            html.Th("Detected type"),
                                            html.Th("Unique values"),
                                            html.Th("Missing values"),
                                            html.Th("Summary / frequent values"),
                                        ]
                                    )
                                ),
                                html.Tbody(column_rows),
                            ]
                        ),
                    ),
                ],
            ),
            html.Div(
                className="rp-quality-panel rp-quality-categories",
                children=[
                    html.H2("Quality issues"),
                    html.P(
                        f"{summary['issue_count']:,} potential issue(s). "
                        "Severity is a heuristic: high means at least 50% "
                        "missing in a column; medium covers duplicate, parse, "
                        "and IQR findings; low covers constant or high-cardinality "
                        "columns. Findings are not automatic corrections."
                    ),
                    html.Div(
                        className="rp-upload-table-wrap",
                        children=html.Table(
                            [
                                html.Thead(
                                    html.Tr(
                                        [
                                            html.Th("Severity"),
                                            html.Th("Finding"),
                                            html.Th("Affected column"),
                                            html.Th("Affected records"),
                                            html.Th("Suggested investigation"),
                                            html.Th("Safe to auto-fix?"),
                                        ]
                                    )
                                ),
                                html.Tbody(issue_rows),
                            ]
                        ),
                    ),
                ],
            ),
            html.Div(
                className="rp-quality-panel rp-quality-categories",
                children=[
                    html.H2("Cleaning recommendations"),
                    html.Ul(
                        [html.Li(item) for item in profile["recommendations"]]
                    ),
                ],
            ),
        ],
    )


def build_quality_lab_content(
    uploaded_data=None,
    default_sales_rows=0,
    default_cleaned_rows=0,
    default_transaction_counts=None,
):
    """Render uploaded-file diagnostics or an honest preprocessed-data state."""
    if not uploaded_data:
        counts = default_transaction_counts or {}
        return html.Section(
            className="rp-quality-panel",
            children=[
                html.Div(
                    className="rp-quality-notice",
                    children=[
                        html.Strong("Built-in dataset: preprocessed UCI data"),
                        html.P(
                            "The bundled dataset is already cleaned. Raw-file "
                            "checks such as original duplicate rows and "
                            "missing-field counts are not available for this "
                            "preprocessed copy. Upload a compatible file on "
                            "Your Data to inspect those checks."
                        ),
                    ],
                ),
                html.Div(
                    className="rp-upload-metrics",
                    children=[
                        _metric_card("Processed rows", default_cleaned_rows),
                        _metric_card("Ordinary sale rows", default_sales_rows),
                        _metric_card(
                            "Cancellations",
                            counts.get("cancellation", 0),
                        ),
                        _metric_card(
                            "Returns / reversals",
                            counts.get("return_or_reversal", 0),
                        ),
                        _metric_card(
                            "Accounting adjustments",
                            counts.get("accounting_adjustment", 0),
                        ),
                    ],
                ),
                html.Div(
                    className="rp-quality-panel",
                    children=[
                        html.H2("Processed transaction categories"),
                        _transaction_table(counts),
                    ],
                ),
            ],
        )

    report = uploaded_data["quality_report"]
    transaction_counts = uploaded_data["transaction_counts"]
    cleaned_rows = uploaded_data["cleaned_rows"]
    sales_rows = uploaded_data["sales_rows"]

    return html.Section(
        className="rp-quality-panel",
        children=[
            html.Div(
                className="rp-quality-source",
                children=[
                    html.Strong("Active upload"),
                    html.Span(uploaded_data["filename"]),
                    html.Span(
                        "Issue counts can overlap; a row may trigger more "
                        "than one check."
                    ),
                ],
            ),
            html.Div(
                className="rp-upload-metrics",
                children=[
                    _metric_card(label, report[key])
                    for label, key in QUALITY_METRICS
                ],
            ),
            html.Div(
                className="rp-quality-funnel",
                children=[
                    _metric_card("Rows after cleaning", cleaned_rows),
                    html.Span("→", className="rp-quality-arrow"),
                    _metric_card("Ordinary sales for analysis", sales_rows),
                    _metric_card(
                        "Rows excluded from sales analytics",
                        cleaned_rows - sales_rows,
                    ),
                ],
            ),
            html.Div(
                className="rp-quality-panel rp-quality-categories",
                children=[
                    html.H2("Cleaned transaction classifications"),
                    _transaction_table(transaction_counts),
                ],
            ),
        ],
    )


def _metric_card(label, value):
    return html.Article(
        className="rp-upload-metric",
        children=[
            html.P(label),
            html.Strong(f"{value:,}"),
        ],
    )


def _transaction_table(counts):
    categories = (
        ("sale", "Ordinary sales"),
        ("cancellation", "Cancellations"),
        ("return_or_reversal", "Returns / reversals"),
        ("zero_price_review", "Zero-price reviews"),
        ("negative_price_review", "Negative-price reviews"),
        ("accounting_adjustment", "Accounting adjustments"),
    )
    return html.Div(
        className="rp-upload-table-wrap",
        children=html.Table(
            [
                html.Thead(
                    html.Tr([html.Th("Category"), html.Th("Rows")])
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(label),
                                html.Td(f"{counts.get(key, 0):,}"),
                            ]
                        )
                        for key, label in categories
                    ]
                ),
            ]
        ),
    )
