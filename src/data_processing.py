
"""Data loading, quality checks, and cleaning for RetailPulse AI."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


def load_retail_data(file_path):
    """Load the original Excel or CSV dataset."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)

    raise ValueError("Supported formats are .xlsx, .xls, and .csv.")


def validate_schema(df):
    """Check that all required dataset columns are present."""
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return True


def create_quality_report(df):
    """Return key data-quality counts without modifying the data."""
    validate_schema(df)

    invoice = df["InvoiceNo"].astype("string").str.strip()
    quantity = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_descriptions": int(df["Description"].isna().sum()),
        "missing_customer_ids": int(df["CustomerID"].isna().sum()),
        "cancellation_invoice_rows": int(
            invoice.str.upper().str.startswith("C", na=False).sum()
        ),
        "negative_quantity_rows": int((quantity < 0).sum()),
        "zero_quantity_rows": int((quantity == 0).sum()),
        "negative_price_rows": int((price < 0).sum()),
        "zero_price_rows": int((price == 0).sum()),
    }


def clean_transactions(df):
    """
    Create a cleaned transaction copy for analysis.

    Exact duplicate rows are removed. Returns, cancellations,
    zero-price items, and unusual accounting records are retained
    for audit purposes and classified rather than silently discarded.
    """
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    validate_schema(df)

    df = df.drop_duplicates().copy()

    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("string").str.strip()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["Country"] = df["Country"].astype("string").str.strip()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")

    # Exclude records that cannot support transaction-level analysis.
    df = df.dropna(
        subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "Country"]
    ).copy()

    
        # Classify each row using explicit precedence.
    is_cancellation = (
        df["InvoiceNo"].str.upper().str.startswith("C", na=False)
    )
    is_adjustment = (
        df["StockCode"].str.upper().eq("B")
        | df["Description"].str.lower().eq("adjust bad debt")
    )
    is_return = df["Quantity"] < 0
    is_invalid_price = df["UnitPrice"] < 0
    is_zero_price = df["UnitPrice"] == 0

    df["TransactionType"] = "sale"
    df.loc[is_zero_price, "TransactionType"] = "zero_price_review"
    df.loc[is_return, "TransactionType"] = "return_or_reversal"
    df.loc[is_cancellation, "TransactionType"] = "cancellation"
    df.loc[is_adjustment, "TransactionType"] = "accounting_adjustment"
    df.loc[is_invalid_price, "TransactionType"] = "negative_price_review"


    # Signed line value preserves the direction of quantity.
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Customer IDs remain missing where unavailable; do not invent IDs.
    df["CustomerID"] = df["CustomerID"].astype("Int64")

    return df.reset_index(drop=True)



def get_sales_transactions(df):
    """
    Return ordinary positive-quantity sales with positive prices.

    Keep only columns needed for retail analytics to reduce memory use.
    This is a gross-sales view, not net revenue after returns.
    """
    required_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
        "Revenue",
    ]

    mask = (
        df["TransactionType"].eq("sale")
        & df["Quantity"].gt(0)
        & df["UnitPrice"].gt(0)
    )

    return df.loc[mask, required_columns]



if __name__ == "__main__":
    dataset_path = Path("data/raw/Online Retail.xlsx")

    raw_df = load_retail_data(dataset_path)
    print("DATA QUALITY REPORT")
    for metric, value in create_quality_report(raw_df).items():
        print(f"{metric}: {value}")

    cleaned_df = clean_transactions(raw_df)
    sales_df = get_sales_transactions(cleaned_df)

   
    print("\nPROCESSING SUMMARY")
    print(f"Rows after duplicate removal and validation: {len(cleaned_df):,}")
    print(f"Ordinary sales rows: {len(sales_df):,}")
    print("\nTransaction types:")
    print(cleaned_df["TransactionType"].value_counts().to_string())

    # Save processed datasets locally.
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / "cleaned_transactions.csv"
    sales_path = output_dir / "sales_transactions.csv"

    cleaned_df.to_csv(cleaned_path, index=False)
    sales_df.to_csv(sales_path, index=False)

    print("\nOUTPUT FILES")
    print(f"Cleaned transactions: {cleaned_path}")
    print(f"Sales transactions: {sales_path}")
    print(f"Cleaned file size: {cleaned_path.stat().st_size / (1024**2):.2f} MB")
    print(f"Sales file size: {sales_path.stat().st_size / (1024**2):.2f} MB")

