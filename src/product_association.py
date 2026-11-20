"""Product association analysis for RetailPulse AI.

This module identifies which products are commonly purchased together in the
same invoice and highlights the strongest cross-sell signals in the dataset.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pandas as pd


MAX_PAIR_CANDIDATES = 1_000_000


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

    pair_candidates = sum(
        len(products) * (len(products) - 1) // 2
        for products in invoice_products
    )
    if pair_candidates > MAX_PAIR_CANDIDATES:
        raise ValueError(
            f"Exact cross-sell analysis exceeds the "
            f"{MAX_PAIR_CANDIDATES:,} pair-work "
            "limit. Narrow the country or date filters and try again."
        )

    pair_counts = Counter()
    for products in invoice_products:
        if len(products) < 2:
            continue
        pair_counts.update(combinations(products, 2))

    if not pair_counts:
        return pd.DataFrame(
            columns=["ProductA", "ProductB", "CoPurchaseCount", "Support", "ProductPair"]
        )

    total_invoices = data["InvoiceNo"].nunique()
    top_pairs = sorted(
        pair_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )[:max_pairs]
    pair_summary = pd.DataFrame(
        [
            {
                "ProductA": product_a,
                "ProductB": product_b,
                "CoPurchaseCount": count,
                "Support": (
                    count / total_invoices * 100 if total_invoices else 0.0
                ),
                "ProductPair": f"{product_a} + {product_b}",
            }
            for (product_a, product_b), count in top_pairs
        ],
        columns=[
            "ProductA",
            "ProductB",
            "CoPurchaseCount",
            "Support",
            "ProductPair",
        ],
    )
    return pair_summary.reset_index(drop=True)


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
