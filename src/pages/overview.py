"""Overview and not-found pages for the RetailPulse AI application."""

from dash import dcc, html


def build_overview_page():
    """Build the product overview without fabricated operational metrics."""
    return html.Div(
        className="main-content rp-overview",
        children=[
            html.Section(
                className="rp-overview-hero",
                children=[
                    html.P(
                        "RETAILPULSE AI  /  OVERVIEW",
                        className="rp-eyebrow",
                    ),
                    html.H1("Your data. Your intelligence. Your decisions."),
                    html.P(
                        "A workspace for exploring retail performance today "
                        "and building toward broader data intelligence.",
                        className="rp-overview-lede",
                    ),
                ],
            ),
            html.Section(
                className="rp-overview-section",
                **{"aria-labelledby": "workspace-heading"},
                children=[
                    html.Div(
                        className="rp-section-heading",
                        children=[
                            html.Div(
                                children=[
                                    html.P(
                                        "WORKSPACE",
                                        className="rp-eyebrow",
                                    ),
                                    html.H2(
                                        "Choose an experience",
                                        id="workspace-heading",
                                    ),
                                ]
                            ),
                            html.P(
                                "Available experiences are clearly "
                                "separated from planned capabilities.",
                                className="rp-section-description",
                            ),
                        ],
                    ),
                    html.Div(
                        className="rp-overview-grid",
                        children=[
                            html.Article(
                                className=(
                                    "rp-overview-card "
                                    "rp-overview-card-active"
                                ),
                                children=[
                                    html.Div(
                                        className="rp-card-topline",
                                        children=[
                                            html.Span(
                                                "RETAIL ANALYTICS",
                                                className="rp-eyebrow",
                                            ),
                                            html.Span(
                                                "Available",
                                                className=(
                                                    "rp-status-pill "
                                                    "rp-status-available"
                                                ),
                                            ),
                                        ],
                                    ),
                                    html.H3("Retail Intelligence"),
                                    html.P(
                                        "Explore revenue, products, "
                                        "customers, purchase patterns, "
                                        "segments, and an experimental "
                                        "forecast using the UCI Online "
                                        "Retail dataset."
                                    ),
                                    dcc.Link(
                                        "Open Retail Intelligence",
                                        href="/retail",
                                        className="rp-primary-action",
                                    ),
                                ],
                            ),
                            html.Article(
                                className=(
                                    "rp-overview-card "
                                    "rp-overview-card-active"
                                ),
                                children=[
                                    html.Div(
                                        className="rp-card-topline",
                                        children=[
                                            html.Span(
                                                "UPLOAD AND ANALYZE",
                                                className="rp-eyebrow",
                                            ),
                                            html.Span(
                                                "Available",
                                                className=(
                                                    "rp-status-pill "
                                                    "rp-status-available"
                                                ),
                                            ),
                                        ],
                                    ),
                                    html.H3("Your Data"),
                                    html.P(
                                        "Upload a retail-compatible CSV or "
                                        "XLSX file, review data-quality "
                                        "checks, and download cleaned rows."
                                    ),
                                    dcc.Link(
                                        "Open Your Data",
                                        href="/data",
                                        className="rp-primary-action",
                                    ),
                                ],
                            ),
                            html.Article(
                                className=(
                                    "rp-overview-card "
                                    "rp-overview-card-active"
                                ),
                                children=[
                                    html.Div(
                                        className="rp-card-topline",
                                        children=[
                                            html.Span(
                                                "DATA DIAGNOSTICS",
                                                className="rp-eyebrow",
                                            ),
                                            html.Span(
                                                "Available",
                                                className=(
                                                    "rp-status-pill "
                                                    "rp-status-available"
                                                ),
                                            ),
                                        ],
                                    ),
                                    html.H3("Data Quality Lab"),
                                    html.P(
                                        "Review upload-specific data issues, "
                                        "cleaning outcomes, and transaction "
                                        "classification counts."
                                    ),
                                    dcc.Link(
                                        "Open Data Quality Lab",
                                        href="/quality",
                                        className="rp-primary-action",
                                    ),
                                ],
                            ),
                            html.Article(
                                className=(
                                    "rp-overview-card "
                                    "rp-overview-card-active"
                                ),
                                children=[
                                    html.Div(
                                        className="rp-card-topline",
                                        children=[
                                            html.Span(
                                                "CUSTOM CHARTS",
                                                className="rp-eyebrow",
                                            ),
                                            html.Span(
                                                "Available",
                                                className=(
                                                    "rp-status-pill "
                                                    "rp-status-available"
                                                ),
                                            ),
                                        ],
                                    ),
                                    html.H3("Visual Explorer"),
                                    html.P(
                                        "Group retail transactions, choose "
                                        "measures and valid aggregations, "
                                        "filter matching records, and export "
                                        "the summarized results."
                                    ),
                                    dcc.Link(
                                        "Open Visual Explorer",
                                        href="/explorer",
                                        className="rp-primary-action",
                                    ),
                                ],
                            ),
                            html.Article(
                                className=(
                                    "rp-overview-card "
                                    "rp-overview-card-active"
                                ),
                                children=[
                                    html.Div(
                                        className="rp-card-topline",
                                        children=[
                                            html.Span(
                                                "REPORTING",
                                                className="rp-eyebrow",
                                            ),
                                            html.Span(
                                                "Available",
                                                className=(
                                                    "rp-status-pill "
                                                    "rp-status-available"
                                                ),
                                            ),
                                        ],
                                    ),
                                    html.H3("Report Centre"),
                                    html.P(
                                        "Download evidence-based PDF and "
                                        "Excel reports, analytical summaries, "
                                        "and the uploaded dataset."
                                    ),
                                    dcc.Link(
                                        "Open Report Centre",
                                        href="/reports",
                                        className="rp-primary-action",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_not_found_page(pathname):
    """Show an explicit not-found state for unsupported routes."""
    return html.Div(
        className="main-content rp-not-found",
        children=[
            html.P("404  /  PAGE NOT FOUND", className="rp-eyebrow"),
            html.H1("This destination is not available."),
            html.P(
                f"No application page is configured for {pathname!r}."
            ),
            dcc.Link(
                "Return to Overview",
                href="/",
                className="rp-primary-action",
            ),
        ],
    )
