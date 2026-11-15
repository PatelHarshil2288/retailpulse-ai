"""Customer-level performance summaries for RetailPulse AI."""

from __future__ import annotations

import pandas as pd


def build_customer_value_summary(sales, top_n=10):
    """Summarize customer revenue, order count, and value.

    Parameters
    ----------
    sales : pandas.DataFrame
        Sales data containing CustomerID, InvoiceNo, Revenue, InvoiceDate, Country.
    top_n : int, default 10
        Number of customers to return in the ranked summary.
    """
    required_columns = {"CustomerID", "InvoiceNo", "Revenue", "InvoiceDate", "Country"}
    missing_columns = required_columns - set(sales.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns for customer insights: {sorted(missing_columns)}"
        )

    if sales.empty:
        return pd.DataFrame(
            columns=[
                "CustomerID",
                "TotalRevenue",
                "InvoiceCount",
                "AverageOrderValue",
                "Country",
                "LastPurchaseDate",
            ]
        )

    data = sales[list(required_columns)].copy()
    data["CustomerID"] = pd.to_numeric(data["CustomerID"], errors="coerce")
    data["Revenue"] = pd.to_numeric(data["Revenue"], errors="coerce")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.dropna(subset=["CustomerID", "InvoiceNo", "Revenue", "InvoiceDate"]).copy()
    data = data.loc[data["CustomerID"] > 0].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "CustomerID",
                "TotalRevenue",
                "InvoiceCount",
                "AverageOrderValue",
                "Country",
                "LastPurchaseDate",
            ]
        )

    customer_summary = (
        data.groupby("CustomerID", as_index=False)
        .agg(
            TotalRevenue=("Revenue", "sum"),
            InvoiceCount=("InvoiceNo", "nunique"),
            LastPurchaseDate=("InvoiceDate", "max"),
            Country=("Country", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        )
        .sort_values("TotalRevenue", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    customer_summary["AverageOrderValue"] = (
        customer_summary["TotalRevenue"] / customer_summary["InvoiceCount"]
    )
    customer_summary["Country"] = customer_summary["Country"].fillna("Unknown")
    customer_summary["LastPurchaseDate"] = pd.to_datetime(
        customer_summary["LastPurchaseDate"], errors="coerce"
    )
    customer_summary = customer_summary[
        [
            "CustomerID",
            "TotalRevenue",
            "InvoiceCount",
            "AverageOrderValue",
            "Country",
            "LastPurchaseDate",
        ]
    ]
    return customer_summary


def summarize_top_customers(sales, top_n=5):
    """Return a concise summary of the strongest customer accounts."""
    summary = build_customer_value_summary(sales, top_n=top_n)
    if summary.empty:
        return {
            "top_customer": "No identifiable customers",
            "top_revenue": 0.0,
            "avg_order_value": 0.0,
            "customer_count": 0,
        }

    top_customer = summary.iloc[0]
    return {
        "top_customer": int(top_customer["CustomerID"]),
        "top_revenue": float(top_customer["TotalRevenue"]),
        "avg_order_value": float(top_customer["AverageOrderValue"]),
        "customer_count": int(len(summary)),
    }
