import pandas as pd

from src.product_association import (
    build_product_associations,
    summarize_top_product_pair,
)


def test_build_product_associations_counts_pairs():
    sales = pd.DataFrame(
        {
            "InvoiceNo": ["A", "A", "B", "B", "B", "C", "D"],
            "Product": ["Tea", "Coffee", "Tea", "Coffee", "Cake", "Tea", "Book"],
        }
    )

    pair_data = build_product_associations(sales, max_pairs=10)

    assert not pair_data.empty
    assert set(pair_data.columns) == {
        "ProductA",
        "ProductB",
        "CoPurchaseCount",
        "Support",
        "ProductPair",
    }
    tea_coffee = pair_data.loc[
        (pair_data["ProductA"] == "Coffee")
        & (pair_data["ProductB"] == "Tea")
    ]
    assert not tea_coffee.empty
    assert tea_coffee.iloc[0]["CoPurchaseCount"] == 2


def test_summarize_top_product_pair_returns_co_purchase_summary():
    sales = pd.DataFrame(
        {
            "InvoiceNo": ["1", "1", "2", "2", "3", "3", "4"],
            "Product": ["Pen", "Notebook", "Pen", "Notebook", "Pen", "Marker", "Marker"],
        }
    )

    summary = summarize_top_product_pair(sales)

    assert summary["pair_name"] == "Notebook + Pen"
    assert summary["count"] == 2
    assert summary["support"] > 0
