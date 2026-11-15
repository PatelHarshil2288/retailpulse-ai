import pandas as pd

from src.customer_insights import build_customer_value_summary, summarize_top_customers


def test_build_customer_value_summary_ranks_revenue():
    sales = pd.DataFrame(
        {
            "CustomerID": [101, 101, 102, 102, 102, 103],
            "InvoiceNo": ["A", "B", "C", "D", "E", "F"],
            "Revenue": [50.0, 150.0, 30.0, 20.0, 10.0, 200.0],
            "InvoiceDate": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-10", "2020-01-12", "2020-01-15", "2020-01-11"]),
            "Country": ["United Kingdom", "United Kingdom", "France", "France", "France", "Germany"],
        }
    )

    summary = build_customer_value_summary(sales, top_n=10)

    assert not summary.empty
    assert list(summary.columns) == [
        "CustomerID",
        "TotalRevenue",
        "InvoiceCount",
        "AverageOrderValue",
        "Country",
        "LastPurchaseDate",
    ]
    assert summary["TotalRevenue"].max() == 200.0
    assert set(summary["CustomerID"].tolist()) == {101, 102, 103}


def test_summarize_top_customers_returns_expected_snapshot():
    sales = pd.DataFrame(
        {
            "CustomerID": [10, 10, 20, 20, 20, 30],
            "InvoiceNo": ["1", "2", "3", "4", "5", "6"],
            "Revenue": [25.0, 75.0, 60.0, 40.0, 20.0, 90.0],
            "InvoiceDate": pd.to_datetime(["2020-01-01", "2020-01-20", "2020-01-03", "2020-01-11", "2020-01-20", "2020-01-05"]),
            "Country": ["United Kingdom", "United Kingdom", "France", "France", "France", "Germany"],
        }
    )

    summary = summarize_top_customers(sales, top_n=3)

    assert summary["top_customer"] == 20
    assert summary["top_revenue"] == 120.0
    assert summary["customer_count"] == 3
