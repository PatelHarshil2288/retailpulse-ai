import pandas as pd

from src.data_profiling import profile_dataset
from src.pages.quality_lab import (
    build_dataset_quality_content,
    build_quality_lab_content,
    build_quality_lab_page,
)


def test_quality_lab_explains_preprocessed_builtin_dataset():
    page_text = str(
        build_quality_lab_content(
            default_sales_rows=120,
            default_cleaned_rows=145,
            default_transaction_counts={"sale": 120, "cancellation": 25},
        )
    )

    assert "already cleaned" in page_text
    assert "Processed rows" in page_text
    assert "145" in page_text
    assert "Cancellations" in page_text


def test_quality_lab_shows_upload_quality_checks_and_cleaning_funnel():
    uploaded_data = {
        "filename": "retail.csv",
        "quality_report": {
            "rows": 4,
            "duplicate_rows": 1,
            "missing_descriptions": 1,
            "missing_customer_ids": 1,
            "cancellation_invoice_rows": 1,
            "negative_quantity_rows": 1,
            "zero_quantity_rows": 0,
            "negative_price_rows": 0,
            "zero_price_rows": 1,
        },
        "transaction_counts": {"sale": 1, "cancellation": 1},
        "cleaned_rows": 3,
        "sales_rows": 1,
    }

    page_text = str(build_quality_lab_content(uploaded_data))

    assert "retail.csv" in page_text
    assert "Duplicate rows" in page_text
    assert "Rows after cleaning" in page_text
    assert "Rows excluded from sales analytics" in page_text
    assert "Issue counts can overlap" in page_text


def test_quality_lab_page_has_a_dynamic_report_region():
    page_text = str(build_quality_lab_page())

    assert "quality-lab-content" in page_text
    assert "DATA QUALITY" in page_text


def test_quality_lab_renders_generic_dataset_findings_and_recommendations():
    data = pd.DataFrame(
        {
            "Amount": [1, 2, 3, 4, 100],
            "Group": ["A", "A", None, "B", "B"],
        }
    )
    rendered = str(
        build_dataset_quality_content(
            profile_dataset(data),
            "generic.csv",
        )
    )

    assert "Profile and quality report: generic.csv" in rendered
    assert "Detected type" in rendered
    assert "Potential IQR outliers" in rendered
    assert "Cleaning recommendations" in rendered
    assert "not modified automatically" in rendered
    assert "Safe to auto-fix?" in rendered
