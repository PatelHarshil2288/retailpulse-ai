from src.app import app


def test_dashboard_app_initializes():
    """Verify that the Dash application initializes correctly."""
    assert app is not None
    assert app.layout is not None


def test_dashboard_layout_contains_main_components():
    """Verify that the dashboard layout contains its main sections."""
    layout_text = str(app.layout)

    assert "Dataset Overview" in layout_text
    assert "Transaction Quality Overview" in layout_text
    assert "Business Insights" in layout_text


def test_dashboard_registers_callbacks():
    """Verify that the dashboard registers interactive callbacks."""
    assert len(app.callback_map) > 0



def test_dashboard_has_expected_interactive_callbacks():
    """Verify that key dashboard interactions are registered."""
    callback_keys = list(app.callback_map.keys())

    assert len(callback_keys) >= 3

    callback_text = " ".join(callback_keys)

    assert "monthly-chart.figure" in callback_text
    assert "country-selector.value" in callback_text
    assert "download-filtered-csv.data" in callback_text


def test_main_callback_returns_expected_outputs():
    """Verify that the main dashboard callback returns all 14 outputs."""
    callback_key = next(
        key for key in app.callback_map
        if "monthly-chart.figure" in key
    )

    callback_function = app.callback_map[callback_key]["callback"]

    # Dash wraps registered callbacks; access the original function.
    if hasattr(callback_function, "__wrapped__"):
        callback_function = callback_function.__wrapped__

    results = callback_function("Revenue", "All", None, None)

    assert len(results) == 14

    # Verify the four chart outputs are Plotly figures.
    assert all(
        hasattr(results[index], "to_dict")
        for index in (0, 5, 6, 7)
    )

    # Verify the KPI outputs are populated.
    assert results[1].startswith("£")
    assert results[2]
    assert results[3].startswith("£")
    assert results[4]


def test_main_callback_calculates_kpis_from_sales_data():
    """Verify that the main callback's KPI values match the sales data."""
    import pandas as pd
    from src.app import sales

    callback_key = next(
        key for key in app.callback_map
        if "monthly-chart.figure" in key
    )

    callback_function = app.callback_map[callback_key]["callback"]

    if hasattr(callback_function, "__wrapped__"):
        callback_function = callback_function.__wrapped__

    results = callback_function("Revenue", "All", None, None)

    expected_revenue = sales["Revenue"].sum()
    expected_orders = sales["InvoiceNo"].nunique()
    expected_average = (
        expected_revenue / expected_orders if expected_orders else 0
    )

    assert results[1] == f"£{expected_revenue:,.2f}"
    assert results[2] == f"{expected_orders:,}"
    assert results[3] == f"£{expected_average:,.2f}"


def test_country_filter_updates_kpis():
    """Verify that selecting a country filters KPI calculations."""
    from src.app import sales

    available_countries = sorted(sales["Country"].dropna().unique())
    assert available_countries, "No countries found in sales data"

    selected_country = available_countries[0]
    country_sales = sales.loc[sales["Country"] == selected_country]

    callback_key = next(
        key for key in app.callback_map
        if "monthly-chart.figure" in key
    )

    callback_function = app.callback_map[callback_key]["callback"]

    if hasattr(callback_function, "__wrapped__"):
        callback_function = callback_function.__wrapped__

    results = callback_function(
        "Revenue", selected_country, None, None
    )

    expected_revenue = country_sales["Revenue"].sum()
    expected_orders = country_sales["InvoiceNo"].nunique()
    expected_average = (
        expected_revenue / expected_orders if expected_orders else 0
    )

    assert results[1] == f"£{expected_revenue:,.2f}"
    assert results[2] == f"{expected_orders:,}"
    assert results[3] == f"£{expected_average:,.2f}"


def test_date_filter_updates_kpis():
    """Verify that date filtering calculates KPIs for the selected period."""
    import pandas as pd
    from src.app import sales

    start_date = sales["InvoiceDate"].min().strftime("%Y-%m-%d")
    end_date = sales["InvoiceDate"].min().strftime("%Y-%m-%d")

    filtered_sales = sales.loc[
        (sales["InvoiceDate"] >= pd.Timestamp(start_date))
        & (
            sales["InvoiceDate"]
            < pd.Timestamp(end_date) + pd.Timedelta(days=1)
        )
    ]

    callback_key = next(
        key for key in app.callback_map
        if "monthly-chart.figure" in key
    )

    callback_function = app.callback_map[callback_key]["callback"]

    if hasattr(callback_function, "__wrapped__"):
        callback_function = callback_function.__wrapped__

    results = callback_function(
        "Revenue", "All", start_date, end_date
    )

    expected_revenue = filtered_sales["Revenue"].sum()
    expected_orders = filtered_sales["InvoiceNo"].nunique()
    expected_average = (
        expected_revenue / expected_orders if expected_orders else 0
    )

    assert results[1] == f"£{expected_revenue:,.2f}"
    assert results[2] == f"{expected_orders:,}"
    assert results[3] == f"£{expected_average:,.2f}"


def test_reset_filters_restores_default_values():
    """Verify that reset restores All countries and the full date range."""
    from src.app import reset_filters, sales

    default_values = reset_filters(0)

    assert default_values[0] == "All"
    assert default_values[1] == sales["InvoiceDate"].min().date()
    assert default_values[2] == sales["InvoiceDate"].max().date()



def test_reset_filters_restores_default_values():
    """Verify that a reset click restores All countries and the full date range."""
    from src.app import reset_filters, sales

    default_values = reset_filters(1)

    assert default_values[0] == "All"
    assert default_values[1] == sales["InvoiceDate"].min().date()
    assert default_values[2] == sales["InvoiceDate"].max().date()


def test_csv_export_returns_none_without_clicks():
    """Verify that CSV export does nothing before the button is clicked."""
    from src.app import export_filtered_csv

    assert export_filtered_csv(None, "All", None, None) is None


def test_csv_export_returns_download_data():
    """Verify that clicking export generates downloadable CSV data."""
    from src.app import export_filtered_csv

    result = export_filtered_csv(1, "All", None, None)

    assert result is not None
    assert isinstance(result, dict)
    assert "content" in result
    assert "filename" in result
    assert result["filename"] == "retailpulse_filtered_sales.csv"
