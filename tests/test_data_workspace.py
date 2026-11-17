import base64
from io import BytesIO

import pandas as pd
import pytest

from src.pages.data_workspace import (
    MAX_UPLOAD_BASE64_LENGTH,
    build_dataset_upload_results,
    build_data_workspace_page,
    deserialize_dashboard_sales,
    deserialize_workspace_dataset,
    list_uploaded_worksheet_names,
    process_uploaded_dataset,
    process_uploaded_data,
    serialize_dashboard_sales,
    serialize_workspace_dataset,
)


def _data_uri(payload, mime_type="text/csv"):
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _retail_csv():
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
                "InvoiceNo": "1001",
                "StockCode": "A1",
                "Description": "Product A",
                "Quantity": 2,
                "InvoiceDate": "2011-01-01",
                "UnitPrice": 10.0,
                "CustomerID": 101,
                "Country": "United Kingdom",
            },
        ]
    )


def test_process_uploaded_csv_reports_quality_and_cleans_duplicates():
    content = _data_uri(_retail_csv().to_csv(index=False).encode("utf-8"))

    analysis = process_uploaded_data(content, "retail.csv")

    assert analysis.quality_report["rows"] == 2
    assert analysis.quality_report["duplicate_rows"] == 1
    assert len(analysis.cleaned_transactions) == 1
    assert analysis.sales_rows == 1


def test_process_uploaded_xlsx_file():
    excel_file = BytesIO()
    _retail_csv().drop_duplicates().to_excel(excel_file, index=False)
    content = _data_uri(
        excel_file.getvalue(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    analysis = process_uploaded_data(content, "retail.xlsx")

    assert len(analysis.cleaned_transactions) == 1
    assert analysis.sales_rows == 1


def test_dashboard_sales_payload_round_trips_cleaned_sales():
    analysis = process_uploaded_data(
        _data_uri(_retail_csv().to_csv(index=False).encode("utf-8")),
        "retail.csv",
    )

    payload = serialize_dashboard_sales(analysis.cleaned_transactions)
    restored = deserialize_dashboard_sales(payload)

    assert len(restored) == analysis.sales_rows
    assert restored.iloc[0]["Country"] == "United Kingdom"
    assert restored.iloc[0]["Revenue"] == 20
    assert restored.iloc[0]["Product"] == "A1 - Product A"


def test_dashboard_sales_payload_rejects_malformed_data():
    with pytest.raises(ValueError, match="invalid"):
        deserialize_dashboard_sales("not-base64")


def test_process_uploaded_file_rejects_non_retail_schema():
    content = _data_uri(b"Item,Price\nA,10\n")

    with pytest.raises(ValueError, match="Missing required columns"):
        process_uploaded_data(content, "other.csv")


def test_process_uploaded_dataset_accepts_generic_csv_and_normalizes_headers():
    content = _data_uri(b" Item ;Item; ;Value\nA;B;C;1\n")

    upload = process_uploaded_dataset(content, "generic.csv")

    assert list(upload.data.columns) == [
        "Item",
        "Item (2)",
        "Column 3",
        "Value",
    ]
    assert upload.data.iloc[0].tolist() == ["A", "B", "C", 1]


def test_process_uploaded_dataset_reads_cp1252_csv():
    content = _data_uri("Product,Price\nCafé,2\n".encode("cp1252"))

    upload = process_uploaded_dataset(content, "generic.csv")

    assert upload.data.iloc[0]["Product"] == "Café"


def test_process_uploaded_dataset_selects_excel_worksheet():
    excel_file = BytesIO()
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        pd.DataFrame({"first": [1]}).to_excel(
            writer,
            sheet_name="First",
            index=False,
        )
        pd.DataFrame({"second": [2]}).to_excel(
            writer,
            sheet_name="Second",
            index=False,
        )
    content = _data_uri(
        excel_file.getvalue(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert list_uploaded_worksheet_names(content, "generic.xlsx") == (
        "First",
        "Second",
    )
    upload = process_uploaded_dataset(content, "generic.xlsx", "Second")

    assert upload.selected_sheet == "Second"
    assert upload.data.iloc[0]["second"] == 2


def test_process_uploaded_dataset_rejects_empty_file():
    with pytest.raises(ValueError, match="empty"):
        process_uploaded_dataset(_data_uri(b""), "empty.csv")


def test_workspace_dataset_payload_round_trips_generic_data():
    data = pd.DataFrame(
        {
            "Product": ["Café", "Tea"],
            "Quantity": [2, 1],
            "Optional": [None, "present"],
        }
    )

    restored = deserialize_workspace_dataset(
        serialize_workspace_dataset(data)
    )

    pd.testing.assert_frame_equal(restored, data)


def test_workspace_dataset_payload_rejects_malformed_data():
    with pytest.raises(ValueError, match="invalid"):
        deserialize_workspace_dataset("not-base64")


def test_workspace_dataset_preview_is_bounded_and_includes_profile():
    data = pd.DataFrame(
        {f"Column {index}": range(12) for index in range(12)}
    )

    rendered = str(build_dataset_upload_results(data, "Data"))

    assert "Column profile" in rendered
    assert "Dataset preview" in rendered
    assert "Showing the first 10 rows and up to 10 columns." in rendered
    assert "Th('Column 10')" not in rendered


def test_process_uploaded_file_rejects_unsupported_extensions():
    with pytest.raises(ValueError, match="Unsupported file type"):
        process_uploaded_data(_data_uri(b"data"), "retail.json")


def test_process_uploaded_file_rejects_oversized_content_before_decoding():
    content = "data:text/csv;base64," + ("A" * (MAX_UPLOAD_BASE64_LENGTH + 4))

    with pytest.raises(ValueError, match="10 MB upload limit"):
        process_uploaded_data(content, "retail.csv")


def test_data_workspace_has_upload_and_size_limit():
    page_text = str(build_data_workspace_page())

    assert "retail-dataset-upload" in page_text
    assert "10 MB" in page_text
    assert "InvoiceNo" in page_text
    assert "retail-worksheet-selector" in page_text
    assert "generic" not in page_text.lower()
