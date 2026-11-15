"""Product association analysis for RetailPulse AI.

This module identifies which products are commonly purchased together in the
same invoice and highlights the strongest cross-sell signals in the dataset.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd


def build_product_associations(sales, max_pairs=10):
    """Return the strongest product-to-product purchase combinations.

    Parameters
    ----------
    sales : pandas.DataFrame
        Sales data containing at least InvoiceNo and Product columns.
    max_pairs : int, default 10
        Maximum number of product pairs to return.
    """
    required_columns = {"InvoiceNo", "Product"}
    missing_columns = required_columns - set(sales.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns for product associations: {sorted(missing_columns)}"
        )

    if sales.empty:
        return pd.DataFrame(
            columns=["ProductA", "ProductB", "CoPurchaseCount", "Support", "ProductPair"]
        )

    data = sales[["InvoiceNo", "Product"]].copy()
    data = data.dropna(subset=["InvoiceNo", "Product"]).copy()
    data["Product"] = data["Product"].astype(str).str.strip()
    data = data.loc[data["Product"] != ""].copy()

    if data.empty:
        return pd.DataFrame(
            columns=["ProductA", "ProductB", "CoPurchaseCount", "Support", "ProductPair"]
        )

    invoice_products = (
        data.groupby("InvoiceNo")["Product"]
        .apply(lambda products: sorted(set(products)))
        .to_list()
    )

    pair_rows = []
    for products in invoice_products:
        if len(products) < 2:
            continue
        for product_a, product_b in combinations(products, 2):
            pair_rows.append((product_a, product_b))

    if not pair_rows:
        return pd.DataFrame(
            columns=["ProductA", "ProductB", "CoPurchaseCount", "Support", "ProductPair"]
        )

    pair_data = pd.DataFrame(pair_rows, columns=["ProductA", "ProductB"])
    pair_summary = (
        pair_data.groupby(["ProductA", "ProductB"], as_index=False)
        .size()
        .rename(columns={"size": "CoPurchaseCount"})
    )

    total_invoices = data["InvoiceNo"].nunique()
    if total_invoices:
        pair_summary["Support"] = (
            pair_summary["CoPurchaseCount"] / total_invoices * 100
        )
    else:
        pair_summary["Support"] = 0.0

    pair_summary["ProductPair"] = (
        pair_summary["ProductA"] + " + " + pair_summary["ProductB"]
    )

    return (
        pair_summary.sort_values(
            ["CoPurchaseCount", "Support"], ascending=[False, False]
        )
        .head(max_pairs)
        .reset_index(drop=True)
    )


def summarize_top_product_pair(sales, max_pairs=10):
    """Return a lightweight summary for the strongest co-purchase pair."""
    pair_data = build_product_associations(sales, max_pairs=max_pairs)

    if pair_data.empty:
        return {
            "pair_name": "No product pairs detected",
            "count": 0,
            "support": 0.0,
        }

    top_pair = pair_data.iloc[0]
    return {
        "pair_name": top_pair["ProductPair"],
        "count": int(top_pair["CoPurchaseCount"]),
        "support": float(top_pair["Support"]),
    }
