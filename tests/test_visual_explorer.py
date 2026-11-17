import pandas as pd
import pytest

from src.pages.visual_explorer import (
    aggregate_sales,
    build_universal_analytics_panel,
    build_visual_explorer_figure,
    filter_sales,
)


@pytest.fixture
def sample_sales():
    return pd.DataFrame(
        [
            {
                "Country": "A-land",
                "InvoiceDate": "2024-01-05",
                "Description": "Widget A",
                "CustomerID": 101,
                "InvoiceNo": "A-1",
                "Quantity": 2,
                "UnitPrice": 5.0,
                "Revenue": 10.0,
            },
            {
                "Country": "A-land",
                "InvoiceDate": "2024-01-06",
                "Description": "Widget B",
                "CustomerID": 102,
                "InvoiceNo": "A-1",
                "Quantity": 1,
                "UnitPrice": 3.0,
                "Revenue": 3.0,
            },
            {
                "Country": "B-land",
                "InvoiceDate": "2024-02-05",
                "Description": "Widget A",
                "CustomerID": 101,
                "InvoiceNo": "B-1",
                "Quantity": 4,
                "UnitPrice": 5.0,
                "Revenue": 20.0,
            },
        ]
    )


def test_aggregates_measures_by_selected_dimension(sample_sales):
    revenue = aggregate_sales(sample_sales, "Country", "Revenue")
    orders = aggregate_sales(
        sample_sales,
        "Country",
        "InvoiceNo",
        "nunique",
    )
    customers = aggregate_sales(
        sample_sales,
        "Country",
        "CustomerID",
        "nunique",
    )

    assert revenue.to_dict("records") == [
        {"Category": "B-land", "Value": 20.0},
        {"Category": "A-land", "Value": 13.0},
    ]
    assert orders.to_dict("records") == [
        {"Category": "A-land", "Value": 1},
        {"Category": "B-land", "Value": 1},
    ]
    assert customers.to_dict("records") == [
        {"Category": "A-land", "Value": 2},
        {"Category": "B-land", "Value": 1},
    ]


def test_month_dimension_keeps_all_months_in_order(sample_sales):
    grouped = aggregate_sales(sample_sales, "Month", "Revenue")

    assert grouped.to_dict("records") == [
        {"Category": "Jan 2024", "Value": 13.0},
        {"Category": "Feb 2024", "Value": 20.0},
    ]


def test_category_dimension_is_limited_to_top_fifteen():
    sales = pd.DataFrame(
        {
            "Country": [f"Country {index}" for index in range(20)],
            "Revenue": [float(index) for index in range(20)],
        }
    )

    grouped = aggregate_sales(sales, "Country", "Revenue")

    assert len(grouped) == 15
    assert grouped.iloc[0].to_dict() == {
        "Category": "Country 19",
        "Value": 19.0,
    }


@pytest.mark.parametrize("chart_type", ["bar", "pie"])
def test_builds_selected_chart_type(sample_sales, chart_type):
    figure = build_visual_explorer_figure(
        sample_sales,
        "Country",
        "Revenue",
        chart_type,
    )

    assert figure.layout.title.text == "Sum Revenue by Country"
    assert len(figure.data) == 1
    if chart_type == "pie":
        assert figure.data[0].type == "pie"
    else:
        assert figure.data[0].type == "bar"


def test_builds_monthly_revenue_line_chart(sample_sales):
    figure = build_visual_explorer_figure(
        sample_sales,
        "Month",
        "Revenue",
        "line",
    )

    assert figure.layout.title.text == "Sum Revenue by Month"
    assert figure.data[0].type == "scatter"
    assert list(figure.data[0].x) == ["Jan 2024", "Feb 2024"]
    assert list(figure.data[0].y) == [13.0, 20.0]
    assert figure.data[0].line.width == 3
    assert figure.data[0].marker.size == 8


def test_rejects_unknown_dimension(sample_sales):
    with pytest.raises(
        ValueError,
        match="Unsupported visual-explorer dimension",
    ):
        aggregate_sales(sample_sales, "Invoice", "Revenue")


def test_rejects_numeric_aggregation_on_categorical_measure(sample_sales):
    with pytest.raises(ValueError, match="requires a numeric measure"):
        aggregate_sales(sample_sales, "Country", "InvoiceNo", "sum")


def test_filter_matches_values_case_insensitively(sample_sales):
    result = filter_sales(sample_sales, "Description", "widget a")

    assert len(result) == 2
    assert result["Description"].tolist() == ["Widget A", "Widget A"]


def test_universal_analytics_panel_contains_generic_interactions():
    page_text = str(build_universal_analytics_panel())

    assert "universal-dimension" in page_text
    assert "universal-measure" in page_text
    assert "universal-analytics-chart" in page_text
    assert "universal-correlation-chart" in page_text
    assert "universal-trend-chart" in page_text
