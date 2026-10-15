
import pandas as pd
import pytest

from src.data_processing import (
    validate_schema,
    create_quality_report,
    clean_transactions,
    get_sales_transactions,
)


@pytest.fixture
def sample_transactions():
    """Create a small dataset covering common transaction types."""
    return pd.DataFrame(
        [
            {
                "InvoiceNo": "1001",
                "StockCode": "A1",
                "Description": "Product A",
                "Quantity": 2,
                "InvoiceDate": "2011-01-01",
                "UnitPrice": 10.0,
                "CustomerID": 101,
                "Country": "United Kingdom",
            },
            {
                "InvoiceNo": "1002",
                "StockCode": "A2",
                "Description": "Product B",
                "Quantity": 1,
                "InvoiceDate": "2011-01-02",
                "UnitPrice": 5.0,
                "CustomerID": 102,
                "Country": "France",
            },
            {
                "InvoiceNo": "1003",
                "StockCode": "A3",
                "Description": "Returned Product",
                "Quantity": -1,
                "InvoiceDate": "2011-01-03",
                "UnitPrice": 10.0,
                "CustomerID": 101,
                "Country": "United Kingdom",
            },
            {
                "InvoiceNo": "C1004",
                "StockCode": "A4",
                "Description": "Cancelled Product",
                "Quantity": 1,
                "InvoiceDate": "2011-01-04",
                "UnitPrice": 8.0,
                "CustomerID": 103,
                "Country": "France",
            },
            {
                "InvoiceNo": "1005",
                "StockCode": "A5",
                "Description": "Free Item",
                "Quantity": 1,
                "InvoiceDate": "2011-01-05",
                "UnitPrice": 0.0,
                "CustomerID": None,
                "Country": "France",
            },
        ]
    )


def test_validate_schema_accepts_required_columns(sample_transactions):
    assert validate_schema(sample_transactions) is True


def test_validate_schema_rejects_missing_column(sample_transactions):
    incomplete = sample_transactions.drop(columns=["Country"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(incomplete)


def test_quality_report_counts_rows_and_special_cases(sample_transactions):
    report = create_quality_report(sample_transactions)

    assert report["rows"] == 5
    assert report["cancellation_invoice_rows"] == 1
    assert report["negative_quantity_rows"] == 1
    assert report["zero_price_rows"] == 1
    assert report["missing_customer_ids"] == 1


def test_clean_transactions_classifies_records(sample_transactions):
    cleaned = clean_transactions(sample_transactions)

    assert len(cleaned) == 5
    assert cleaned["TransactionType"].tolist() == [
        "sale",
        "sale",
        "return_or_reversal",
        "cancellation",
        "zero_price_review",
    ]


def test_clean_transactions_removes_exact_duplicates(sample_transactions):
    duplicated = pd.concat(
        [sample_transactions, sample_transactions.iloc[[0]]],
        ignore_index=True,
    )

    cleaned = clean_transactions(duplicated)

    assert len(cleaned) == 5


def test_clean_transactions_calculates_signed_revenue(sample_transactions):
    cleaned = clean_transactions(sample_transactions)

    assert cleaned.loc[2, "Revenue"] == -10.0
    assert cleaned.loc[0, "Revenue"] == 20.0


def test_sales_transactions_keep_only_ordinary_sales(sample_transactions):
    cleaned = clean_transactions(sample_transactions)
    sales = get_sales_transactions(cleaned)

    assert len(sales) == 2
    assert sales["InvoiceNo"].tolist() == ["1001", "1002"]
    assert (sales["Quantity"] > 0).all()
    assert (sales["UnitPrice"] > 0).all()



def test_clean_transactions_removes_invalid_dates(sample_transactions):
    sample_transactions.loc[0, "InvoiceDate"] = "invalid-date"

    cleaned = clean_transactions(sample_transactions)

    assert len(cleaned) == 4
    assert "1001" not in cleaned["InvoiceNo"].tolist()



def test_clean_transactions_removes_missing_required_fields(sample_transactions):
    sample_transactions.loc[0, "Country"] = None

    cleaned = clean_transactions(sample_transactions)

    assert len(cleaned) == 4
    assert "1001" not in cleaned["InvoiceNo"].tolist()



def test_clean_transactions_classifies_negative_price(sample_transactions):
    sample_transactions.loc[0, "UnitPrice"] = -5.0

    cleaned = clean_transactions(sample_transactions)

    first_transaction = cleaned.loc[
        cleaned["InvoiceNo"] == "1001"
    ].iloc[0]

    assert first_transaction["TransactionType"] == "negative_price_review"
    assert first_transaction["Revenue"] == -10.0



def test_clean_transactions_classifies_accounting_adjustment(sample_transactions):
    sample_transactions.loc[0, "StockCode"] = "B"

    cleaned = clean_transactions(sample_transactions)

    first_transaction = cleaned.loc[
        cleaned["InvoiceNo"] == "1001"
    ].iloc[0]

    assert first_transaction["TransactionType"] == "accounting_adjustment"



def test_sales_transactions_exclude_zero_quantity(sample_transactions):
    sample_transactions.loc[0, "Quantity"] = 0

    cleaned = clean_transactions(sample_transactions)
    sales = get_sales_transactions(cleaned)

    assert "1001" not in sales["InvoiceNo"].astype(str).values
