"""Shared navigation shell for the RetailPulse AI Dash application."""

from dash import dcc, html


def _route_link(label, href, element_id):
    return dcc.Link(
        label,
        href=href,
        id=element_id,
        className="rp-nav-link",
    )


def _planned_link(label):
    return html.Div(
        className="rp-nav-link rp-nav-planned",
        **{"aria-disabled": "true"},
        children=[
            html.Span(label),
            html.Span("Planned", className="rp-planned-badge"),
        ],
    )


def _navigation_links(suffix):
    return [
        _route_link("Overview", "/", f"nav-overview-{suffix}"),
        _route_link(
            "Retail Intelligence",
            "/retail",
            f"nav-retail-{suffix}",
        ),
        _route_link("Your Data", "/data", f"nav-data-{suffix}"),
        html.Div(
            className="rp-nav-section-label",
            children="Explore",
        ),
        _route_link(
            "Data Quality Lab",
            "/quality",
            f"nav-quality-{suffix}",
        ),
        _route_link(
            "Visual Explorer",
            "/explorer",
            f"nav-explorer-{suffix}",
        ),
        _route_link(
            "Report Centre",
            "/reports",
            f"nav-reports-{suffix}",
        ),
        _planned_link("Data Workspace"),
        _planned_link("AI Insights"),
        _planned_link("Forecast Studio"),
        _planned_link("Model Evaluation"),
        _planned_link("Documentation"),
        _planned_link("Settings"),
    ]


def build_app_shell():
    """Create the shared desktop and mobile application frame."""
    brand = html.Span(
        children=[
            html.Span("RP", className="rp-brand-mark"),
            html.Span(
                children=[
                    html.Strong("RetailPulse"),
                    html.Small("DATA INTELLIGENCE"),
                ],
                className="rp-brand-copy",
            ),
        ],
        className="rp-brand",
    )

    return html.Div(
        className="app-shell rp-app-shell",
        children=[
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="uploaded-sales-store", storage_type="memory"),
            dcc.Store(id="uploaded-dataset-store", storage_type="memory"),
            dcc.Store(id="uploaded-file-store", storage_type="memory"),
            html.A(
                "Skip to main content",
                href="#page-content",
                className="rp-skip-link",
            ),
            html.Header(
                className="rp-mobile-header",
                children=[
                    dcc.Link(
                        brand,
                        href="/",
                        className="rp-brand-link",
                        title="RetailPulse AI home",
                    ),
                    html.Details(
                        id="mobile-nav-details",
                        className="rp-mobile-menu",
                        children=[
                            html.Summary(
                                "Menu",
                                **{"aria-label": "Open navigation menu"},
                            ),
                            html.Nav(
                                className="rp-mobile-panel",
                                **{"aria-label": "Mobile primary navigation"},
                                children=_navigation_links("mobile"),
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="rp-app-frame",
                children=[
                    html.Aside(
                        className="rp-sidebar",
                        children=[
                            dcc.Link(
                                brand,
                                href="/",
                                className="rp-brand-link",
                                title="RetailPulse AI home",
                            ),
                            html.P(
                                "WORKSPACE",
                                className="rp-nav-section-label",
                            ),
                            html.Nav(
                                className="rp-navigation",
                                **{"aria-label": "Primary navigation"},
                                children=_navigation_links("desktop"),
                            ),
                            html.Div(
                                className="rp-sidebar-footer",
                                children="Built-in: UCI Online Retail",
                            ),
                        ],
                    ),
                    html.Div(
                        className="rp-main-column",
                        children=[
                            html.Header(
                                className="rp-topbar",
                                children=[
                                    html.Div(
                                        children=[
                                            html.P(
                                                "RETAILPULSE AI  /  WORKSPACE",
                                                className="rp-topbar-eyebrow",
                                            ),
                                            html.Strong(
                                                "Analytics workspace"
                                            ),
                                        ],
                                    ),
                                    dcc.Link(
                                        "Manage data",
                                        href="/data",
                                        className="rp-topbar-action",
                                    ),
                                ],
                            ),
                            html.Main(
                                id="page-content",
                                className="rp-page-container",
                                tabIndex="-1",
                                children=[
                                    html.Div(
                                        id="dataset-source-notice",
                                        className=(
                                            "rp-data-source-notice rp-hidden"
                                        ),
                                        children=html.Span(
                                            id="dataset-source-label"
                                        ),
                                    ),
                                    html.Div(id="route-page-content"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
