from src.app import app


def test_dashboard_app_initializes():
    """Verify that the Dash application initializes correctly."""
    assert app is not None
    assert app.layout is not None


def test_dash_callback_requests_have_a_server_side_size_limit(monkeypatch):
    from src.app import MAX_DASH_REQUEST_BYTES

    assert app.server.config["MAX_CONTENT_LENGTH"] == MAX_DASH_REQUEST_BYTES

    monkeypatch.setitem(app.server.config, "MAX_CONTENT_LENGTH", 16)
    response = app.server.test_client().post(
        "/_dash-update-component",
        data=b"x" * 17,
        content_type="application/json",
    )

    assert response.status_code == 413


def test_dashboard_layout_contains_main_components():
    """Verify that the dashboard layout contains its main sections."""
    from src.app import render_page

    layout_text = str(render_page("/retail")[0])

    assert "Dataset Overview" in layout_text
    assert "Transaction Quality Overview" in layout_text
    assert "Business Insights" in layout_text
    assert "period-revenue-chart" in layout_text
    assert "period-orders-chart" in layout_text
    assert "declining-products-chart" in layout_text


def test_url_routes_render_overview_and_retail_pages():
    """Verify route content and active navigation stay in sync."""
    from src.app import render_page

    routes = (
        ("/", "Choose an experience", 0),
        ("/retail", "Dataset Overview", 1),
        ("/data", "Upload transactions", 2),
        ("/quality", "DATA QUALITY", 3),
        ("/explorer", "CHART BUILDER", 4),
        ("/reports", "Download analytical reports", 5),
    )
    for pathname, expected_content, active_link in routes:
        page = render_page(pathname)
        assert expected_content in str(page[0])
        expected_classes = ["rp-nav-link"] * 6
        expected_classes[active_link] = "rp-nav-link rp-nav-link-active"
        assert page[1:7] == tuple(expected_classes)
        assert page[7:13] == tuple(expected_classes)
        assert page[13] is False


def test_unknown_route_returns_an_explicit_not_found_page():
    """Do not silently route unsupported paths to a product page."""
    from src.app import render_page

    page = render_page("/not-a-feature")

    assert "PAGE NOT FOUND" in str(page[0])
    assert all(value == "rp-nav-link" for value in page[1:13])
    assert page[13] is False


def test_shell_marks_unavailable_areas_as_planned():
    """Keep future destinations visible without making them active links."""
    layout_text = str(app.layout)

    assert "Your Data" in layout_text
    assert "Visual Explorer" in layout_text
    assert "Report Centre" in layout_text
    assert "/data" in layout_text
    assert "/explorer" in layout_text
    assert "Planned" in layout_text
    assert "aria-disabled" in layout_text


def test_shell_keeps_uploads_in_browser_session_stores():
    layout_text = str(app.layout)

    assert "uploaded-dataset-store" in layout_text
    assert "uploaded-file-store" in layout_text


def test_generic_upload_does_not_claim_to_power_retail_analytics():
    from src import app as app_module

    callback = app_module.update_dataset_source_notice
    if hasattr(callback, "__wrapped__"):
        callback = callback.__wrapped__

    class_name, notice = callback(
        None,
        {"payload": "data", "filename": "inventory.csv"},
        "/explorer",
    )

    assert class_name == "rp-data-source-notice"
    assert "built-in UCI dataset" in str(notice)
    assert "inventory.csv" in str(notice)


def test_dataset_source_notice_reports_invalid_browser_stores():
    from src import app as app_module

    retail_class, retail_notice = (
        app_module.update_dataset_source_notice(
            "invalid-retail-store",
            None,
            "/retail",
        )
    )
    workspace_class, workspace_notice = (
        app_module.update_dataset_source_notice(
            None,
            [],
            "/quality",
        )
    )

    assert retail_class == "rp-data-source-notice"
    assert "Retail Intelligence store is invalid" in str(retail_notice)
    assert workspace_class == "rp-data-source-notice"
    assert "workspace store is invalid" in str(workspace_notice)


def test_retail_callbacks_surface_malformed_dashboard_payload():
    from src import app as app_module

    invalid_store = {
        "payload": "not-base64",
        "filename": "broken.csv",
        "transaction_counts": {},
        "cleaned_rows": 0,
    }

    explorer = app_module.update_visual_explorer(
        "Country",
        "Revenue",
        "sum",
        "bar",
        "",
        "",
        invalid_store,
    )
    visual_export = app_module.export_visual_explorer_results(
        1,
        "Country",
        "Revenue",
        "sum",
        "",
        "",
        invalid_store,
    )
    filters = app_module.update_dashboard_filter_controls(invalid_store)
    dataset_summary = app_module.update_dataset_summary(invalid_store)
    dashboard = app_module.update_chart(
        "Revenue",
        "Revenue",
        "All",
        None,
        None,
        invalid_store,
    )
    summary_cards = app_module.update_summary_cards(
        "Revenue",
        "Revenue",
        "All",
        None,
        None,
        invalid_store,
    )
    csv_export = app_module.export_filtered_csv(
        1,
        "All",
        None,
        None,
        invalid_store,
    )
    cross_sell = app_module.update_cross_sell(
        "All",
        None,
        None,
        invalid_store,
    )
    customer_insights = app_module.update_customer_insights(
        "All",
        None,
        None,
        invalid_store,
    )
    revenue_forecast = app_module.update_revenue_forecast(
        "All",
        None,
        None,
        invalid_store,
    )
    daily_trends =     app_module.update_period_sales_trends(
        "All",
        None,
        None,
        invalid_store,
    )
    product_declines = app_module.update_declining_products(
        "All",
        None,
        None,
        invalid_store,
    )

    assert "invalid" in str(explorer[1]).lower()
    assert visual_export[0] is app_module.no_update
    assert "invalid" in str(visual_export[1]).lower()
    assert "Dataset error" in filters[0][0]["label"]
    assert "invalid" in str(dataset_summary[0]).lower()
    assert "invalid" in str(dashboard[1]).lower()
    assert "invalid" in str(summary_cards[0]).lower()
    assert csv_export[0] is app_module.no_update
    assert "invalid" in str(csv_export[1]).lower()
    assert "invalid" in str(cross_sell[1]).lower()
    assert "invalid" in str(customer_insights[1]).lower()
    assert "invalid" in str(revenue_forecast[1]).lower()
    assert "invalid" in daily_trends[0].layout.annotations[0].text.lower()
    assert "invalid" in daily_trends[1].layout.annotations[0].text.lower()
    assert "invalid" in product_declines.layout.annotations[0].text.lower()


def test_workspace_callbacks_guard_non_mapping_stores():
    from src import app as app_module

    mode = app_module.update_visual_explorer_mode(42)
    controls = app_module.update_universal_analytics_controls(42)
    chart, summary, preview = app_module.update_universal_analytics_chart(
        "Group",
        "Amount",
        "sum",
        "bar",
        "",
        "",
        42,
    )

    assert mode == ("rp-explorer-panel", "rp-explorer-panel rp-hidden")
    assert controls[0] == []
    assert summary == "No uploaded dataset is active."
    assert preview is not None
    assert len(chart.data) == 0


def test_universal_analytics_callbacks_use_uploaded_dataset():
    import pandas as pd

    from src import app as app_module
    from src.pages.data_workspace import serialize_workspace_dataset

    data = pd.DataFrame(
        {
            "Group": ["A", "A", "B"],
            "Amount": [10.0, 20.0, 30.0],
            "Other": [2.0, 4.0, 6.0],
            "EventDate": ["2024-01-02", "2024-01-20", "2024-02-02"],
        }
    )
    dataset_store = {
        "payload": serialize_workspace_dataset(data),
        "filename": "sample.csv",
        "rows": len(data),
    }

    controls = app_module.update_universal_analytics_controls(dataset_store)
    chart, summary, preview = app_module.update_universal_analytics_chart(
        "Group",
        "Amount",
        "sum",
        "bar",
        "",
        "",
        dataset_store,
    )
    plan, ai_insights, ml_suitability, statistics, correlation, trend = (
        app_module.render_universal_analysis_overview(dataset_store)
    )

    assert controls[1] == "EventDate"
    assert controls[3] == "Amount"
    assert chart.data[0].type == "bar"
    assert "3 of 3 rows" in summary
    assert preview is not None
    assert "Automatic analysis plan" in str(plan)
    assert "Recommendation:" in str(ai_insights)
    assert "target" in str(ml_suitability).lower()
    assert "Numeric column" in str(statistics)
    assert correlation.data[0].type == "heatmap"
    assert trend.data[0].type == "scatter"


def test_universal_chart_deserializes_uploaded_dataset_once(monkeypatch):
    import pandas as pd

    import src.universal_analytics as universal_analytics
    from src import app as app_module
    from src.pages.data_workspace import serialize_workspace_dataset

    dataset_store = {
        "payload": serialize_workspace_dataset(
            pd.DataFrame({"Group": ["A", "B"], "Amount": [10, 20]})
        ),
        "filename": "sample.csv",
    }
    original_deserializer = app_module.deserialize_workspace_dataset
    deserialized_payloads = []

    def track_deserialization(payload):
        deserialized_payloads.append(payload)
        return original_deserializer(payload)

    monkeypatch.setattr(
        app_module,
        "deserialize_workspace_dataset",
        track_deserialization,
    )
    monkeypatch.setattr(
        universal_analytics,
        "deserialize_workspace_dataset",
        track_deserialization,
    )

    figure, summary, preview = app_module.update_universal_analytics_chart(
        "Group",
        "Amount",
        "sum",
        "bar",
        "",
        "",
        dataset_store,
    )

    assert deserialized_payloads == [dataset_store["payload"]]
    assert figure.data[0].type == "bar"
    assert "2 of 2 rows" in summary
    assert preview is not None


def test_universal_ml_baseline_requires_target_confirmation_and_evaluates():
    import pandas as pd

    from src import app as app_module
    from src.pages.data_workspace import serialize_workspace_dataset

    rows = 220
    data = pd.DataFrame(
        {
            "FeatureA": [float(index % 9) for index in range(rows)],
            "FeatureB": [float(index) * 0.4 for index in range(rows)],
            "Category": [
                "A" if index % 2 == 0 else "B" for index in range(rows)
            ],
            "TargetValue": [100 + index * 1.5 for index in range(rows)],
        }
    )
    dataset_store = {
        "payload": serialize_workspace_dataset(data),
        "filename": "ml.csv",
        "rows": len(data),
    }

    options, selected_target = app_module.update_ml_target_options(
        dataset_store
    )
    assert options
    assert selected_target in {item["value"] for item in options}

    baseline_output = app_module._render_universal_ml_baseline(
        dataset_store,
        1,
        selected_target,
        "auto",
        "universal-ml-run-button",
    )
    assert "Best baseline model" in str(baseline_output)
    assert "Rows used" in str(baseline_output)


def test_universal_unsupervised_callback_renders_cluster_report():
    import pandas as pd

    from src import app as app_module
    from src.pages.data_workspace import serialize_workspace_dataset

    rows = 80
    data = pd.DataFrame(
        {
            "MetricA": [float(index % 10) for index in range(rows)],
            "MetricB": [float(index // 10) for index in range(rows)],
        }
    )
    dataset_store = {
        "payload": serialize_workspace_dataset(data),
        "filename": "clusters.csv",
        "rows": len(data),
    }

    options, default_features = (
        app_module.update_unsupervised_feature_options(dataset_store)
    )
    result = app_module.run_universal_unsupervised_analysis(
        1,
        default_features,
        "clustering",
        3,
        dataset_store,
    )

    assert options
    assert "K-means clustering" in str(result)
    assert "Silhouette score" in str(result)


def test_malformed_dataset_stores_return_visible_callback_errors():
    from src import app as app_module

    invalid_store = {"payload": "not-base64", "filename": "broken.csv"}

    target_options, selected_target = app_module.update_ml_target_options(
        invalid_store
    )
    feature_options, selected_features = (
        app_module.update_unsupervised_feature_options(invalid_store)
    )
    baseline_result = app_module._render_universal_ml_baseline(
        invalid_store,
        1,
        "Target",
        "auto",
        "universal-ml-run-button",
    )
    unsupervised_result = app_module.run_universal_unsupervised_analysis(
        1,
        ["Feature"],
        "clustering",
        2,
        invalid_store,
    )
    download_result = app_module.download_workspace_dataset(1, invalid_store)

    assert "Dataset error" in target_options[0]["label"]
    assert selected_target is None
    assert "Dataset error" in feature_options[0]["label"]
    assert selected_features == []
    assert "invalid" in str(baseline_result).lower()
    assert "invalid" in str(unsupervised_result).lower()
    assert download_result[0] is app_module.no_update
    assert "invalid" in str(download_result[1]).lower()


def test_universal_controls_and_export_report_malformed_dataset_store():
    from src import app as app_module

    invalid_store = {"payload": "not-base64", "filename": "broken.csv"}

    controls = app_module.update_universal_analytics_controls(invalid_store)
    aggregations = app_module.update_universal_aggregation_options(
        "Amount",
        invalid_store,
        "sum",
    )
    forecast_options = app_module.update_universal_forecast_options(
        invalid_store
    )
    export_result = app_module.export_universal_analytics(
        1,
        "Group",
        "Amount",
        "sum",
        None,
        None,
        invalid_store,
    )

    assert "Dataset error" in controls[0][0]["label"]
    assert controls[1] is None
    assert "Dataset error" in aggregations[0][0]["label"]
    assert aggregations[1] is None
    assert "Dataset error" in forecast_options[0][0]["label"]
    assert forecast_options[1] is None
    assert export_result[0] is app_module.no_update
    assert "invalid" in str(export_result[1]).lower()


def test_upload_callbacks_report_malformed_store_and_worksheet_errors():
    from src import app as app_module

    malformed_parse = app_module.update_uploaded_dataset(
        {"filename": "missing-content.csv"},
        None,
    )
    malformed_workspace = app_module.render_workspace_upload(
        "invalid-store",
        None,
    )
    invalid_workbook = app_module.restore_workspace_worksheet(
        {
            "contents": "data:application/octet-stream;base64,Zm9v",
            "filename": "invalid.xlsx",
        }
    )

    assert malformed_parse[0] is None
    assert "invalid" in malformed_parse[1]["error"].lower()
    assert "invalid" in str(malformed_workspace[0]).lower()
    assert invalid_workbook[2] == "rp-worksheet-field rp-hidden"
    assert "worksheet list could not be loaded" in str(
        invalid_workbook[3]
    ).lower()


def test_report_callbacks_handle_malformed_store_shapes_and_filename():
    import pandas as pd

    from src import app as app_module
    from src.pages.data_workspace import serialize_workspace_dataset

    disabled_status = app_module.update_report_center_status(42)
    malformed_download = app_module._build_analytical_report_download(
        "report-pdf-button",
        42,
    )
    invalid_filename_download = app_module._build_analytical_report_download(
        "report-pdf-button",
        {
            "payload": serialize_workspace_dataset(
                pd.DataFrame({"Amount": [1, 2, 3]})
            ),
            "filename": [],
        },
    )

    assert disabled_status[:4] == (True, True, True, True)
    assert "No uploaded dataset" in disabled_status[4]
    assert malformed_download[:4] == (app_module.no_update,) * 4
    assert "Upload a CSV" in malformed_download[4]
    assert invalid_filename_download[:4] == (app_module.no_update,) * 4
    assert "filename is invalid" in invalid_filename_download[4]


def test_dataset_csv_report_skips_profile_generation(monkeypatch):
    import pandas as pd

    from src import app as app_module
    from src.pages.data_workspace import serialize_workspace_dataset

    dataset_store = {
        "payload": serialize_workspace_dataset(
            pd.DataFrame({"Amount": [10, 20]})
        ),
        "filename": "sales.csv",
    }

    def reject_profile_generation(*args):
        raise AssertionError(
            "Dataset CSV download must not build an analytical report."
        )

    monkeypatch.setattr(
        app_module,
        "build_report_context",
        reject_profile_generation,
    )
    result = app_module._build_analytical_report_download(
        "report-data-button",
        dataset_store,
    )

    assert result[:3] == (app_module.no_update,) * 3
    assert result[3]["filename"] == "sales-dataset.csv"
    assert result[4] == "Dataset CSV generated."


def test_xlsx_upload_parsing_reuses_returned_worksheet_metadata(monkeypatch):
    import base64
    from io import BytesIO

    import pandas as pd

    from src import app as app_module

    workbook_bytes = BytesIO()
    with pd.ExcelWriter(workbook_bytes, engine="openpyxl") as writer:
        pd.DataFrame({"First": [1]}).to_excel(
            writer,
            sheet_name="First",
            index=False,
        )
        pd.DataFrame({"Second": [2]}).to_excel(
            writer,
            sheet_name="Second",
            index=False,
        )
    uploaded_file = {
        "contents": (
            "data:application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet;base64,"
            + base64.b64encode(workbook_bytes.getvalue()).decode("ascii")
        ),
        "filename": "worksheets.xlsx",
    }

    def reject_redundant_open(*args):
        raise AssertionError(
            "Worksheet metadata should come from dataset parsing."
        )

    monkeypatch.setattr(
        app_module,
        "list_uploaded_worksheet_names",
        reject_redundant_open,
    )
    response = app_module.app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": (
                "..uploaded-sales-store.data..."
                "uploaded-dataset-store.data.."
            ),
            "outputs": [
                {"id": "uploaded-sales-store", "property": "data"},
                {"id": "uploaded-dataset-store", "property": "data"},
            ],
            "inputs": [
                {
                    "id": "uploaded-file-store",
                    "property": "data",
                    "value": uploaded_file,
                },
                {
                    "id": "retail-worksheet-selector",
                    "property": "value",
                    "value": None,
                },
            ],
            "state": [],
            "changedPropIds": ["uploaded-file-store.data"],
        },
    )

    assert response.status_code == 200
    dataset_store = response.get_json()["response"][
        "uploaded-dataset-store"
    ]["data"]
    assert dataset_store["sheet_names"] == ["First", "Second"]
    assert dataset_store["selected_sheet"] == "First"


def test_generic_upload_flows_into_the_quality_lab():
    import base64

    client = app.server.test_client()
    contents = "data:text/csv;base64," + base64.b64encode(
        b"Group,Amount\nA,1\nA,2\nB,3\n,100\n"
    ).decode("ascii")
    file_response = client.post(
        "/_dash-update-component",
        json={
            "output": "uploaded-file-store.data",
            "outputs": {"id": "uploaded-file-store", "property": "data"},
            "inputs": [
                {
                    "id": "retail-dataset-upload",
                    "property": "contents",
                    "value": contents,
                },
                {
                    "id": "restore-default-data",
                    "property": "n_clicks",
                    "value": 0,
                },
            ],
            "state": [
                {
                    "id": "retail-dataset-upload",
                    "property": "filename",
                    "value": "quality.csv",
                }
            ],
            "changedPropIds": ["retail-dataset-upload.contents"],
        },
    )
    assert file_response.status_code == 200
    uploaded_file = file_response.get_json()["response"]["uploaded-file-store"][
        "data"
    ]

    parse_response = client.post(
        "/_dash-update-component",
        json={
            "output": (
                "..uploaded-sales-store.data..."
                "uploaded-dataset-store.data.."
            ),
            "outputs": [
                {"id": "uploaded-sales-store", "property": "data"},
                {"id": "uploaded-dataset-store", "property": "data"},
            ],
            "inputs": [
                {
                    "id": "uploaded-file-store",
                    "property": "data",
                    "value": uploaded_file,
                },
                {
                    "id": "retail-worksheet-selector",
                    "property": "value",
                    "value": None,
                },
            ],
            "state": [],
            "changedPropIds": ["uploaded-file-store.data"],
        },
    )
    assert parse_response.status_code == 200
    parsed = parse_response.get_json()["response"]
    dataset_store = parsed["uploaded-dataset-store"]["data"]
    assert parsed["uploaded-sales-store"]["data"] is None
    assert dataset_store["rows"] == 4
    assert dataset_store["filename"] == "quality.csv"

    quality_response = client.post(
        "/_dash-update-component",
        json={
            "output": "quality-lab-content.children",
            "outputs": {
                "id": "quality-lab-content",
                "property": "children",
            },
            "inputs": [
                {
                    "id": "uploaded-sales-store",
                    "property": "data",
                    "value": parsed["uploaded-sales-store"]["data"],
                },
                {
                    "id": "uploaded-dataset-store",
                    "property": "data",
                    "value": dataset_store,
                },
            ],
            "state": [],
            "changedPropIds": ["uploaded-dataset-store.data"],
        },
    )
    assert quality_response.status_code == 200
    quality_content = str(
        quality_response.get_json()["response"]["quality-lab-content"][
            "children"
        ]
    )
    assert "Quality issues" in quality_content
    assert "Cleaning recommendations" in quality_content


def test_dashboard_registers_callbacks():
    """Verify that the dashboard registers interactive callbacks."""
    assert len(app.callback_map) > 0



def test_dashboard_has_expected_interactive_callbacks():
    """Verify that key dashboard interactions are registered."""
    callback_keys = list(app.callback_map.keys())

    assert len(callback_keys) >= 3

    callback_text = " ".join(callback_keys)

    assert "period-performance-chart.figure" in callback_text
    assert "country-selector.value" in callback_text
    assert "download-filtered-csv.data" in callback_text
    assert "visual-explorer-chart.figure" in callback_text


def test_visual_explorer_limits_aggregations_for_categorical_measures():
    from src.app import update_visual_explorer_aggregations

    options, selected_value = update_visual_explorer_aggregations(
        "InvoiceNo",
        "sum",
    )

    assert {option["value"] for option in options} == {"count", "nunique"}
    assert selected_value == "nunique"


def test_visual_explorer_export_contains_grouped_results(monkeypatch):
    import pandas as pd
    from src import app as app_module

    sample_sales = pd.DataFrame(
        {
            "Country": ["Exampleland", "Exampleland", "Elsewhere"],
            "Revenue": [12.0, 8.0, 4.0],
        }
    )
    monkeypatch.setattr(app_module, "sales", sample_sales)

    result, status = app_module.export_visual_explorer_results(
        1,
        "Country",
        "Revenue",
        "sum",
        "",
        "",
    )

    assert result["filename"] == "visual-explorer-results.csv"
    assert "Exampleland,20.0" in result["content"]
    assert status is app_module.no_update


def test_main_callback_returns_expected_outputs():
    """Verify that the main dashboard callback returns all 14 outputs."""
    callback_key = next(
        key for key in app.callback_map
        if "period-performance-chart.figure" in key
    )

    callback_function = app.callback_map[callback_key]["callback"]

    # Dash wraps registered callbacks; access the original function.
    if hasattr(callback_function, "__wrapped__"):
        callback_function = callback_function.__wrapped__

    results = callback_function("Revenue", "Revenue", "All", None, None)

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


def test_period_chart_hover_highlights_selected_measure(monkeypatch):
    import pandas as pd

    from src import app as app_module

    monkeypatch.setattr(
        app_module,
        "sales",
        pd.DataFrame(
            {
                "InvoiceNo": ["I-1", "I-2", "I-3"],
                "InvoiceDate": pd.to_datetime(
                    ["2024-01-02", "2024-01-15", "2024-02-02"]
                ),
                "Revenue": [50.0, 40.0, 40.0],
                "Orders": [1, 1, 1],
                "Quantity": [2, 1, 1],
                "UnitPrice": [25.0, 40.0, 40.0],
                "Product": ["Widget A", "Widget B", "Widget A"],
                "Country": ["Exampleland"] * 3,
                "CustomerID": [501, 502, 501],
            }
        ),
    )

    revenue_figure = app_module.update_chart(
        "Revenue",
        "Revenue",
        "All",
        None,
        None,
    )[0]
    orders_figure = app_module.update_chart(
        "Orders",
        "Revenue",
        "All",
        None,
        None,
    )[0]

    assert "Revenue: £%{y:,.2f}" in revenue_figure.data[0].hovertemplate
    assert "Orders: %{customdata[1]:,.0f}" in revenue_figure.data[0].hovertemplate
    assert "Orders: %{y:,.0f}" in orders_figure.data[0].hovertemplate
    assert "Revenue: £%{customdata[0]:,.2f}" in orders_figure.data[0].hovertemplate
    assert revenue_figure.data[0].hoverlabel.font.size == 16


def test_analytics_charts_share_the_dark_theme(monkeypatch):
    """Keep charts from each analytics callback visually consistent."""
    import pandas as pd
    from src import app as app_module

    months = pd.date_range("2020-01-01", periods=8, freq="MS")
    rows = []
    for month_index, month in enumerate(months):
        for product_index, product in enumerate(("Alpha", "Beta")):
            rows.append(
                {
                    "InvoiceNo": f"INV-{month_index}",
                    "StockCode": product,
                    "Description": product,
                    "Product": f"{product} - {product}",
                    "InvoiceDate": month,
                    "Quantity": 1,
                    "UnitPrice": 10 + product_index,
                    "Revenue": 10 + product_index + month_index,
                    "Country": "United Kingdom",
                    "CustomerID": 10001 + month_index % 2,
                }
            )

    monkeypatch.setattr(app_module, "sales", pd.DataFrame(rows))
    start_date = months[0].date().isoformat()
    end_date = months[-1].date().isoformat()

    figures = [
        app_module.update_chart(
            "Revenue", "Revenue", "All", start_date, end_date
        )[0],
        app_module.update_cross_sell("All", start_date, end_date)[0],
        app_module.update_customer_insights(
            "All", start_date, end_date
        )[0],
        app_module.update_revenue_forecast(
            "All", start_date, end_date
        )[0],
    ]

    assert all(
        figure.layout.paper_bgcolor == "#111a29"
        and figure.layout.plot_bgcolor == "#111a29"
        for figure in figures
    )


def test_cross_sell_callback_computes_exact_pairs_only_once(monkeypatch):
    import pandas as pd

    from src import app as app_module
    from src.product_association import build_product_associations

    sales_data = pd.DataFrame(
        {
            "InvoiceNo": ["A", "A", "B", "B"],
            "Product": ["Tea", "Coffee", "Tea", "Coffee"],
            "Country": ["UK"] * 4,
            "InvoiceDate": pd.to_datetime(["2024-01-01"] * 4),
        }
    )
    original_sales = sales_data.copy(deep=True)
    monkeypatch.setattr(app_module, "sales", sales_data)
    calls = []

    def count_pair_calculations(data, max_pairs=10):
        calls.append(1)
        return build_product_associations(data, max_pairs)

    monkeypatch.setattr(
        app_module,
        "build_product_associations",
        count_pair_calculations,
    )

    _, summary = app_module.update_cross_sell("All", None, None)

    assert len(calls) == 1
    assert "Strongest pair: Coffee + Tea" in str(summary)
    pd.testing.assert_frame_equal(sales_data, original_sales)


def test_cross_sell_callback_surfaces_the_pair_work_limit(monkeypatch):
    import pandas as pd

    from src import app as app_module
    from src import product_association

    monkeypatch.setattr(product_association, "MAX_PAIR_CANDIDATES", 2)
    monkeypatch.setattr(
        app_module,
        "sales",
        pd.DataFrame(
            {
                "InvoiceNo": ["A"] * 3,
                "Product": ["Tea", "Coffee", "Cake"],
                "Country": ["UK"] * 3,
                "InvoiceDate": pd.to_datetime(["2024-01-01"] * 3),
            }
        ),
    )

    figure, summary = app_module.update_cross_sell("All", None, None)

    assert "exceeds the 2 pair-work limit" in str(summary)
    assert "Narrow the country or date filters" in figure.layout.annotations[0].text


def test_customer_insights_reuses_its_customer_aggregation(monkeypatch):
    import pandas as pd

    from src import app as app_module
    from src import customer_insights

    sales_data = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2, 2],
            "InvoiceNo": ["A", "B", "C", "D"],
            "Revenue": [10.0, 20.0, 5.0, 5.0],
            "InvoiceDate": pd.to_datetime(["2024-01-01"] * 4),
            "Country": ["UK"] * 4,
        }
    )
    original_sales = sales_data.copy(deep=True)
    monkeypatch.setattr(app_module, "sales", sales_data)
    original = customer_insights.build_customer_value_summary
    calls = []

    def count_aggregations(data, top_n=10):
        calls.append(1)
        return original(data, top_n)

    monkeypatch.setattr(
        customer_insights,
        "build_customer_value_summary",
        count_aggregations,
    )
    monkeypatch.setattr(
        app_module,
        "build_customer_value_summary",
        count_aggregations,
    )

    _, summary = app_module.update_customer_insights("All", None, None)

    assert len(calls) == 1
    assert "Top customer: Customer 1 with £30.00" in str(summary)
    assert "top 2 customer accounts" in str(summary)
    pd.testing.assert_frame_equal(sales_data, original_sales)


def test_summary_cards_group_customer_metrics_once(monkeypatch):
    import pandas as pd

    from src import app as app_module

    monkeypatch.setattr(
        app_module,
        "sales",
        pd.DataFrame(
            {
                "CustomerID": [1, 1, 2],
                "InvoiceNo": ["A", "B", "C"],
                "Revenue": [10.0, 20.0, 5.0],
                "InvoiceDate": pd.to_datetime(
                    ["2024-01-01", "2024-02-01", "2024-02-02"]
                ),
                "Country": ["UK"] * 3,
                "Product": ["Tea", "Coffee", "Tea"],
            }
        ),
    )
    original_groupby = pd.DataFrame.groupby
    customer_groupby_calls = []

    def count_customer_groupby(data, by=None, *args, **kwargs):
        if isinstance(by, str) and by == "CustomerID":
            customer_groupby_calls.append(1)
        return original_groupby(data, by=by, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", count_customer_groupby)

    result = app_module.update_summary_cards(
        "Revenue",
        "Revenue",
        "All",
        None,
        None,
    )

    assert len(customer_groupby_calls) == 1
    assert result[5] == "Customer 1"
    assert result[6] == "Revenue: £30.00"
    assert result[8] == "85.7% of gross sales"


def test_main_callback_calculates_kpis_from_sales_data():
    """Verify that the main callback's KPI values match the sales data."""
    import pandas as pd
    from src.app import sales

    callback_key = next(
        key for key in app.callback_map
        if "period-performance-chart.figure" in key
    )

    callback_function = app.callback_map[callback_key]["callback"]

    if hasattr(callback_function, "__wrapped__"):
        callback_function = callback_function.__wrapped__

    results = callback_function("Revenue", "Revenue", "All", None, None)

    expected_revenue = sales["Revenue"].sum()
    expected_orders = sales["InvoiceNo"].nunique()
    expected_average = (
        expected_revenue / expected_orders if expected_orders else 0
    )

    assert results[1] == f"£{expected_revenue:,.2f}"
    assert results[2] == f"{expected_orders:,}"
    assert results[3] == f"£{expected_average:,.2f}"


def test_dashboard_customer_count_reuses_customer_order_groups(monkeypatch):
    import pandas as pd

    from src import app as app_module

    monkeypatch.setattr(
        app_module,
        "sales",
        pd.DataFrame(
            {
                "InvoiceNo": ["A", "B", "C"],
                "StockCode": ["A1", "A1", "B1"],
                "Description": ["Widget A", "Widget A", "Widget B"],
                "InvoiceDate": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-02-01"]
                ),
                "Quantity": [1, 2, 1],
                "UnitPrice": [10.0, 10.0, 5.0],
                "Revenue": [10.0, 20.0, 5.0],
                "Country": ["UK"] * 3,
                "CustomerID": [101, 101, 202],
                "Product": ["A1 - Widget A", "A1 - Widget A", "B1 - Widget B"],
            }
        ),
    )
    original_nunique = pd.Series.nunique
    customer_unique_scans = []
    original_groupby = pd.DataFrame.groupby
    product_groupby_calls = []

    def count_customer_unique_scans(series, *args, **kwargs):
        if series.name == "CustomerID":
            customer_unique_scans.append(1)
        return original_nunique(series, *args, **kwargs)

    def count_product_groupby(data, by=None, *args, **kwargs):
        if isinstance(by, str) and by == "Product":
            product_groupby_calls.append(1)
        return original_groupby(data, by=by, *args, **kwargs)

    monkeypatch.setattr(
        pd.Series,
        "nunique",
        count_customer_unique_scans,
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "groupby",
        count_product_groupby,
    )
    result = app_module.update_chart(
        "Revenue",
        "Revenue",
        "All",
        None,
        None,
    )

    assert result[12] == "2"
    assert customer_unique_scans == []
    assert len(product_groupby_calls) == 1


def test_uploaded_dataset_drives_retail_dashboard_metrics_and_filters():
    import pandas as pd
    from src import app as app_module
    from src.data_processing import clean_transactions
    from src.pages.data_workspace import serialize_dashboard_sales

    raw_transactions = pd.DataFrame(
        [
            {
                "InvoiceNo": "U-1001",
                "StockCode": "A1",
                "Description": "Widget A",
                "Quantity": 2,
                "InvoiceDate": "2024-01-15",
                "UnitPrice": 10.0,
                "CustomerID": 501,
                "Country": "Exampleland",
            },
            {
                "InvoiceNo": "U-1002",
                "StockCode": "B1",
                "Description": "Widget B",
                "Quantity": 1,
                "InvoiceDate": "2024-02-15",
                "UnitPrice": 5.0,
                "CustomerID": 501,
                "Country": "Exampleland",
            },
        ]
    )
    cleaned = clean_transactions(raw_transactions)
    uploaded_data = {
        "payload": serialize_dashboard_sales(cleaned),
        "filename": "sample-retail.csv",
        "transaction_counts": cleaned["TransactionType"].value_counts().to_dict(),
        "cleaned_rows": len(cleaned),
    }

    dashboard = app_module.update_chart(
        "Revenue",
        "Revenue",
        "All",
        None,
        None,
        uploaded_data,
    )
    filters = app_module.update_dashboard_filter_controls(uploaded_data)
    summary = app_module.update_dataset_summary(uploaded_data)
    explorer_figure, explorer_summary, explorer_preview = (
        app_module.update_visual_explorer(
        "Country",
        "Revenue",
        "sum",
        "bar",
        "",
        "",
        uploaded_data,
        )
    )

    assert dashboard[1] == "£25.00"
    assert dashboard[2] == "2"
    assert dashboard[13] == "Exampleland"
    assert filters[0] == [
        {"label": "All Countries", "value": "All"},
        {"label": "Exampleland", "value": "Exampleland"},
    ]
    assert str(filters[4]) == "2024-01-15"
    assert str(filters[7]) == "2024-02-15"
    assert summary[1:4] == ("2", "2", "1")
    assert summary[5] == "2"
    assert list(explorer_figure.data[0].x) == [25.0]
    assert list(explorer_figure.data[0].y) == ["Exampleland"]
    assert "sample-retail.csv" in explorer_summary
    assert "Widget A" in str(explorer_preview)


def test_daily_period_sales_trends_follow_country_and_date_filters(monkeypatch):
    import pandas as pd

    from src import app as app_module

    monkeypatch.setattr(
        app_module,
        "sales",
        pd.DataFrame(
            {
                "InvoiceNo": ["A", "A", "B", "C", "D"],
                "Country": ["UK", "UK", "UK", "UK", "France"],
                "InvoiceDate": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-02",
                    ]
                ),
                "Revenue": [10.0, 5.0, 20.0, 7.0, 100.0],
            }
        ),
    )

    revenue_figure, orders_figure = app_module.update_period_sales_trends(
        "UK",
        "2024-01-01",
        "2024-01-02",
    )

    assert list(revenue_figure.data[0].y) == [15.0, 20.0]
    assert list(orders_figure.data[0].y) == [1, 1]


def test_weekly_retail_summary_and_forecast_use_monday_periods(monkeypatch):
    import pandas as pd

    from src import app as app_module

    weeks = pd.date_range("2022-01-03", periods=80, freq="W-MON")
    sample_sales = pd.DataFrame(
        {
            "InvoiceNo": [f"INV-{index}" for index in range(len(weeks))],
            "InvoiceDate": weeks,
            "Revenue": [100.0 + index for index in range(len(weeks))],
            "Country": ["UK"] * len(weeks),
            "Product": ["Widget"] * len(weeks),
            "CustomerID": [501] * len(weeks),
        }
    )
    monkeypatch.setattr(app_module, "sales", sample_sales)

    trend, orders = app_module.update_period_sales_trends(
        "UK",
        None,
        None,
        time_granularity="weekly",
    )
    summary = app_module.update_summary_cards(
        "Revenue",
        "Revenue",
        "UK",
        None,
        None,
        time_granularity="weekly",
    )
    forecast, forecast_summary = app_module.update_revenue_forecast(
        "UK",
        None,
        None,
        time_granularity="weekly",
    )

    assert trend.layout.title.text == "Weekly Revenue"
    assert trend.data[0].x[0] == pd.Timestamp("2022-01-03")
    assert orders.data[0].y[0] == 1
    assert summary[1] == "Week of 10 Jul 2023"
    assert len(forecast.data[1].x) == 4
    assert all(pd.Timestamp(period).weekday() == 0 for period in forecast.data[1].x)
    assert "Forecast horizon: 4 weeks" in forecast_summary


def test_country_filter_updates_kpis():
    """Verify that selecting a country filters KPI calculations."""
    from src.app import sales

    available_countries = sorted(sales["Country"].dropna().unique())
    assert available_countries, "No countries found in sales data"

    selected_country = available_countries[0]
    country_sales = sales.loc[sales["Country"] == selected_country]

    callback_key = next(
        key for key in app.callback_map
        if "period-performance-chart.figure" in key
    )

    callback_function = app.callback_map[callback_key]["callback"]

    if hasattr(callback_function, "__wrapped__"):
        callback_function = callback_function.__wrapped__

    results = callback_function(
        "Revenue", "Revenue", selected_country, None, None
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
            < pd.Timestamp(end_date) + pd.offsets.Day(1)
        )
    ]

    callback_key = next(
        key for key in app.callback_map
        if "period-performance-chart.figure" in key
    )

    callback_function = app.callback_map[callback_key]["callback"]

    if hasattr(callback_function, "__wrapped__"):
        callback_function = callback_function.__wrapped__

    results = callback_function(
        "Revenue", "Revenue", "All", start_date, end_date
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
    from src import app as app_module

    assert app_module.export_filtered_csv(
        None,
        "All",
        None,
        None,
    ) == (None, app_module.no_update)



def test_csv_export_returns_download_data(monkeypatch):
    """Verify that clicking export generates downloadable CSV data."""
    import pandas as pd
    from src import app as app_module

    sample_sales = pd.DataFrame(
        {
            "InvoiceNo": ["536365", "536366"],
            "StockCode": ["85123A", "71053"],
            "Description": ["Product A", "Product B"],
            "InvoiceDate": pd.to_datetime(
                ["2010-12-01", "2010-12-02"]
            ),
            "Quantity": [2, 1],
            "UnitPrice": [3.50, 5.00],
            "Revenue": [7.00, 5.00],
            "Country": ["United Kingdom", "France"],
            "CustomerID": [17850, 12583],
        }
    )

    monkeypatch.setattr(app_module, "sales", sample_sales)

    result, status = app_module.export_filtered_csv(
        1,
        "All",
        None,
        None,
    )

    assert result is not None
    assert isinstance(result, dict)
    assert "content" in result
    assert "filename" in result
    assert result["filename"] == "retailpulse_filtered_sales.csv"
    assert "InvoiceNo" in result["content"]
    assert "536365" in result["content"]
    assert "536366" in result["content"]
    pd.testing.assert_frame_equal(app_module.sales, sample_sales)
    assert status is app_module.no_update



def test_revenue_forecast_chart_exists_in_layout():
    """Verify that the forecast chart and model summary are in the layout."""
    from src.app import render_page

    layout_text = str(render_page("/retail")[0])

    assert "revenue-forecast-chart" in layout_text
    assert "forecast-model-summary" in layout_text
    assert "Revenue Forecast" in layout_text


def test_revenue_forecast_callback_is_registered():
    """Verify that the forecast has its own independent callback."""
    from src.app import app

    callback_keys = list(app.callback_map.keys())
    callback_text = " ".join(callback_keys)

    assert "revenue-forecast-chart.figure" in callback_text
    assert "forecast-model-summary.children" in callback_text


def test_revenue_forecast_callback_returns_valid_figure():
    """Verify that the forecast callback returns a chart and model summary."""
    from src.app import app, update_revenue_forecast

    figure, summary = update_revenue_forecast("All", None, None)

    assert hasattr(figure, "to_dict")
    assert "Selected model:" in summary
    assert "Backtesting MAE:" in summary

    figure_data = figure.to_dict()["data"]

    assert len(figure_data) >= 2
    assert any(
        trace.get("name") == "Forecast revenue"
        for trace in figure_data
    )


def test_revenue_forecast_handles_short_date_range():
    """Verify that short date selections produce a helpful empty state."""
    from src.app import update_revenue_forecast

    from src.app import sales

    first_date = sales["InvoiceDate"].min().strftime("%Y-%m-%d")
    figure, summary = update_revenue_forecast(
        "All",
        first_date,
        first_date,
    )

    assert hasattr(figure, "to_dict")
    assert "unavailable" in summary.lower()


def test_product_metric_selector_contains_expected_options():
    """Verify that the product explorer exposes all three metrics."""
    from src.app import render_page

    layout_text = str(render_page("/retail")[0])

    assert "product-metric-selector" in layout_text
    assert "Units Sold" in layout_text
    assert "Average Selling Price" in layout_text
    assert "Revenue" in layout_text


def test_product_explorer_metrics_and_rankings(monkeypatch):
    """Verify that each product metric uses the correct aggregation."""
    import pandas as pd
    from src import app as app_module

    sample_sales = pd.DataFrame(
        {
            "InvoiceNo": ["I1", "I2", "I3", "I4", "I5"],
            "Product": [
                "Product A",
                "Product A",
                "Product B",
                "Product C",
                "Product C",
            ],
            "Quantity": [4, 6, 2, 3, 3],
            "Revenue": [8.0, 12.0, 40.0, 15.0, 15.0],
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-01-02",
                    "2011-01-03",
                    "2011-01-04",
                    "2011-01-05",
                ]
            ),
            "Country": [
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
            ],
        }
    )

    monkeypatch.setattr(app_module, "sales", sample_sales)

    callback_key = next(
        key
        for key in app_module.app.callback_map
        if "period-performance-chart.figure" in key
    )
    callback_function = app_module.app.callback_map[callback_key]["callback"]

    if hasattr(callback_function, "__wrapped__"):
        callback_function = callback_function.__wrapped__

    expected_rankings = {
        "Revenue": ["Product A", "Product C", "Product B"],
        "Units Sold": ["Product B", "Product C", "Product A"],
        "Average Selling Price": [
            "Product A",
            "Product C",
            "Product B",
        ],
    }

    expected_values = {
        "Revenue": [20.0, 30.0, 40.0],
        "Units Sold": [2, 6, 10],
        "Average Selling Price": [2.0, 5.0, 20.0],
    }

    for metric, expected_products in expected_rankings.items():
        results = callback_function(
            "Revenue",
            metric,
            "All",
            None,
            None,
        )

        product_figure = results[5]

        assert len(results) == 14
        assert list(product_figure.data[0].y) == expected_products
        assert list(product_figure.data[0].x) == expected_values[metric]
        assert metric in product_figure.layout.title.text


def test_product_explorer_respects_country_and_date_filters(monkeypatch):
    """Verify product metrics use the selected country and date range."""
    import pandas as pd
    from src import app as app_module

    sample_sales = pd.DataFrame(
        {
            "InvoiceNo": ["I1", "I2", "I3"],
            "Product": ["Product A", "Product B", "Product A"],
            "Quantity": [5, 10, 100],
            "Revenue": [50.0, 20.0, 1000.0],
            "InvoiceDate": pd.to_datetime(
                ["2011-01-10", "2011-01-12", "2011-02-10"]
            ),
            "Country": [
                "United Kingdom",
                "France",
                "United Kingdom",
            ],
        }
    )

    monkeypatch.setattr(app_module, "sales", sample_sales)

    callback_key = next(
        key
        for key in app_module.app.callback_map
        if "period-performance-chart.figure" in key
    )
    callback_function = app_module.app.callback_map[callback_key]["callback"]

    if hasattr(callback_function, "__wrapped__"):
        callback_function = callback_function.__wrapped__

    results = callback_function(
        "Revenue",
        "Revenue",
        "United Kingdom",
        "2011-01-01",
        "2011-01-31",
    )

    product_figure = results[5]

    assert list(product_figure.data[0].y) == ["Product A"]
    assert list(product_figure.data[0].x) == [50.0]
    assert "United Kingdom" in product_figure.layout.title.text


def test_declining_products_compare_latest_two_daily_periods_and_filters(monkeypatch):
    import pandas as pd

    from src import app as app_module

    monkeypatch.setattr(
        app_module,
        "sales",
        pd.DataFrame(
            {
                "InvoiceNo": ["A", "B", "C", "D", "E", "F", "G"],
                "Product": [
                    "Product A",
                    "Product A",
                    "Product A",
                    "Product B",
                    "Product C",
                    "Product A",
                    "Product A",
                ],
                "Country": [
                    "UK",
                    "UK",
                    "UK",
                    "UK",
                    "UK",
                    "France",
                    "France",
                ],
                "InvoiceDate": pd.to_datetime(
                    [
                        "2024-03-01",
                        "2024-03-02",
                        "2024-03-03",
                        "2024-03-01",
                        "2024-03-02",
                        "2024-03-01",
                        "2024-03-02",
                    ]
                ),
                "Revenue": [
                    100.0,
                    40.0,
                    5.0,
                    50.0,
                    30.0,
                    500.0,
                    1000.0,
                ],
            }
        ),
    )

    figure = app_module.update_declining_products(
        "UK",
        "2024-03-01",
        "2024-03-03",
    )

    assert "02 Mar 2024 to 03 Mar 2024" in figure.layout.title.text
    assert dict(zip(figure.data[0].y, figure.data[0].x)) == {
        "Product A": -35.0,
        "Product C": -30.0,
    }

    one_day_figure = app_module.update_declining_products(
        "UK",
        "2024-03-02",
        "2024-03-02",
    )
    assert "at least two daily periods" in (
        one_day_figure.layout.annotations[0].text.lower()
    )
    