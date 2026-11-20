from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from src.pages.report_center import build_report_center_page
from src.reporting import (
    build_report_context,
    create_excel_report,
    create_pdf_report,
)


def _report_data():
    return pd.DataFrame(
        {
            "Category": ["A", "B", "A", "=1+1", None, "B"],
            "Amount": [10.0, 15.0, 20.0, 25.0, 30.0, None],
            "EventDate": [
                "2024-01-01",
                "2024-01-02",
                "2024-02-01",
                "2024-02-02",
                "2024-03-01",
                "2024-03-02",
            ],
        }
    )


def test_report_context_uses_real_profile_and_analysis_results():
    data = _report_data()

    context = build_report_context(data, "sales.csv")

    assert context["profile"]["summary"]["rows"] == len(data)
    assert "Amount" in context["statistics"]["Column"].tolist()
    assert "Recommendation" in context["findings"]["Type"].tolist()
    assert not context["processed"].empty
    assert context["time_granularity"] == "daily"
    assert context["time_series"]["Period"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-02-02"),
        pd.Timestamp("2024-03-01"),
    ]
    assert "sales.csv" in context["insights"]["findings"][0]


def test_report_context_uses_monday_start_weekly_periods():
    context = build_report_context(
        _report_data(),
        "sales.csv",
        time_granularity="weekly",
    )

    assert context["time_series"]["Period"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-29"),
        pd.Timestamp("2024-02-26"),
    ]


def test_excel_report_contains_required_sheets_and_safe_text():
    context = build_report_context(_report_data(), "sales.csv")

    workbook_bytes = create_excel_report(context)
    workbook = load_workbook(BytesIO(workbook_bytes), data_only=False)

    assert workbook.sheetnames == [
        "Dataset Summary",
        "Data Quality",
        "Statistical Analysis",
        "Categorical Analysis",
        "Analytical Findings",
        "Processed Results",
        "Time Series",
    ]
    category_values = [
        cell.value
        for row in workbook["Categorical Analysis"].iter_rows()
        for cell in row
    ]
    assert "'=1+1" in category_values
    assert workbook["Dataset Summary"].freeze_panes == "A2"


def test_excel_report_sanitizes_invalid_and_overlong_cell_text():
    context = build_report_context(_report_data(), "sales.csv")
    context["findings"] = pd.DataFrame(
        {
            "Type": ["Finding", "Finding"],
            "Statement": ["\x00valid", "x" * 40_000],
        }
    )

    workbook_bytes = create_excel_report(context)
    workbook = load_workbook(BytesIO(workbook_bytes), data_only=False)
    statements = [
        row[1].value
        for row in workbook["Analytical Findings"].iter_rows(min_row=2)
    ]

    assert statements[0] == "valid"
    assert len(statements[1]) == 32_767
    assert statements[1].endswith("...")


def test_pdf_report_generates_a_nonempty_pdf_with_chart():
    data = _report_data()
    context = build_report_context(data, "sales.csv")

    pdf_bytes = create_pdf_report(data, "sales.csv", context)

    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2_000


def test_report_center_exposes_download_controls():
    page_text = str(build_report_center_page())

    assert "report-pdf-button" in page_text
    assert "report-excel-button" in page_text
    assert "report-summary-button" in page_text
    assert "report-data-button" in page_text
    assert "report-time-granularity" in page_text
