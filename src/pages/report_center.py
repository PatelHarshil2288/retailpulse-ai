"""Downloadable report centre for uploaded datasets."""

from dash import dcc, html


def build_report_center_page():
    return html.Div(
        className="main-content rp-data-workspace rp-report-center",
        children=[
            html.Section(
                className="rp-overview-hero",
                children=[
                    html.P("REPORTS  /  REPORT CENTRE", className="rp-eyebrow"),
                    html.H1("Turn your analysis into a clear report."),
                    html.P(
                        "Generate downloadable PDF, Excel, and CSV outputs "
                        "from the currently uploaded dataset. Reports include "
                        "measured profile results and disclose their "
                        "methodology and limitations. Excel exports sanitize "
                        "text and truncate cell values above Excel's limit; "
                        "the uploaded dataset remains unchanged.",
                        className="rp-overview-lede",
                    ),
                ],
            ),
            html.Section(
                className="rp-analysis-detail-card",
                **{"aria-labelledby": "report-download-heading"},
                children=[
                    html.H2(
                        "Download analytical reports",
                        id="report-download-heading",
                    ),
                    html.P(
                        "Uploads are held in this browser session. Select "
                        "Your Data and upload a file before generating "
                        "reports."
                    ),
                    html.Label(
                        "Time-series granularity",
                        htmlFor="report-time-granularity",
                    ),
                    dcc.Dropdown(
                        id="report-time-granularity",
                        options=[
                            {"label": "Daily", "value": "daily"},
                            {
                                "label": "Weekly (Monday start)",
                                "value": "weekly",
                            },
                        ],
                        value="daily",
                        clearable=False,
                        className="rp-select",
                    ),
                    html.P(
                        "PDF and Excel reports include an automatically "
                        "detected date-and-numeric trend at this cadence."
                    ),
                    html.P(
                        id="report-center-status",
                        role="status",
                        **{"aria-live": "polite"},
                    ),
                    html.Div(
                        className="rp-upload-actions",
                        children=[
                            html.Button(
                                "Download PDF report",
                                id="report-pdf-button",
                                className="rp-primary-action",
                                disabled=True,
                            ),
                            dcc.Download(id="report-pdf-download"),
                            html.Button(
                                "Download Excel workbook",
                                id="report-excel-button",
                                className="rp-primary-action",
                                disabled=True,
                            ),
                            dcc.Download(id="report-excel-download"),
                            html.Button(
                                "Download summary CSV",
                                id="report-summary-button",
                                className="rp-primary-action",
                                disabled=True,
                            ),
                            dcc.Download(id="report-summary-download"),
                            html.Button(
                                "Download dataset CSV",
                                id="report-data-button",
                                className="rp-primary-action",
                                disabled=True,
                            ),
                            dcc.Download(id="report-data-download"),
                        ],
                    ),
                ],
            ),
        ],
    )
