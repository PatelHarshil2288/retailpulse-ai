"""PDF and Excel reports built from measured dataset profile evidence."""

from datetime import datetime, timezone
from io import BytesIO
import re
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.ai_insights import generate_deterministic_ai_insights
from src.data_profiling import profile_dataset
from src.time_series import granularity_label, period_starts, validate_granularity


EXCEL_CELL_CHARACTER_LIMIT = 32_767
EXCEL_ILLEGAL_CHARACTERS = re.compile(r"[\000-\010\013\014\016-\037]")


def _safe_excel_text(value):
    if not isinstance(value, str):
        return value
    safe_value = EXCEL_ILLEGAL_CHARACTERS.sub("", value)
    if safe_value.startswith(("=", "+", "-", "@", "\t", "\r")):
        safe_value = "'" + safe_value
    if len(safe_value) > EXCEL_CELL_CHARACTER_LIMIT:
        safe_value = (
            safe_value[: EXCEL_CELL_CHARACTER_LIMIT - 3] + "..."
        )
    return safe_value


def _excel_safe_frame(frame):
    result = frame.copy()
    result.columns = [_safe_excel_text(str(column)) for column in result.columns]
    for column in result.columns:
        result[column] = result[column].map(_safe_excel_text)
    return result


def build_report_context(data, dataset_name, time_granularity="daily"):
    """Build report-ready analytical tables from the uploaded dataset."""
    validate_granularity(time_granularity)
    profile = profile_dataset(data)
    insights = generate_deterministic_ai_insights(profile, dataset_name)
    analyzed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary = profile["summary"]

    summary_frame = pd.DataFrame(
        [
            {"Metric": "Dataset", "Value": str(dataset_name)},
            {"Metric": "Rows", "Value": int(summary["rows"])},
            {"Metric": "Columns", "Value": int(summary["columns"])},
            {"Metric": "Memory bytes", "Value": int(summary["memory_bytes"])},
            {"Metric": "Missing cells", "Value": int(summary["missing_cells"])},
            {"Metric": "Rows with missing values", "Value": int(summary["rows_with_missing"])},
            {"Metric": "Exact duplicate rows", "Value": int(summary["duplicate_rows"])},
            {"Metric": "Profiled quality issues", "Value": int(summary["issue_count"])},
            {"Metric": "Analysis date (UTC)", "Value": analyzed_at},
        ]
    )
    quality_frame = pd.DataFrame(
        [
            {
                "Issue": issue["type"],
                "Column": issue["column"] or "",
                "Description": issue["description"],
                "Affected rows": issue["affected_rows"],
                "Affected percent": issue["affected_percent"],
                "Severity": issue["severity"],
                "Recommendation": issue["recommendation"],
            }
            for issue in profile["issues"]
        ],
        columns=[
            "Issue",
            "Column",
            "Description",
            "Affected rows",
            "Affected percent",
            "Severity",
            "Recommendation",
        ],
    )
    statistical_rows = []
    categorical_rows = []
    for column in profile["columns"]:
        if column["detected_type"] == "numeric":
            statistical_rows.append(
                {
                    "Column": column["name"],
                    "Valid values": column["non_missing_count"] - column["invalid_numeric_count"],
                    "Missing values": column["missing_count"],
                    "Minimum": column["statistics"].get("minimum"),
                    "Q1": column["statistics"].get("q1"),
                    "Median": column["statistics"].get("median"),
                    "Q3": column["statistics"].get("q3"),
                    "Maximum": column["statistics"].get("maximum"),
                    "Mean": column["statistics"].get("mean"),
                    "Std. dev.": column["statistics"].get("standard_deviation"),
                    "Outlier candidates": column["outlier_count"],
                }
            )
        if column["detected_type"] in {"categorical", "boolean"}:
            for value in column["top_values"]:
                categorical_rows.append(
                    {
                        "Column": column["name"],
                        "Value": value["value"],
                        "Count": value["count"],
                        "Non-missing column values": column["non_missing_count"],
                    }
                )
    statistical_frame = pd.DataFrame(
        statistical_rows,
        columns=[
            "Column",
            "Valid values",
            "Missing values",
            "Minimum",
            "Q1",
            "Median",
            "Q3",
            "Maximum",
            "Mean",
            "Std. dev.",
            "Outlier candidates",
        ],
    )
    categorical_frame = pd.DataFrame(
        categorical_rows,
        columns=[
            "Column",
            "Value",
            "Count",
            "Non-missing column values",
        ],
    )

    numeric_columns = [
        item["name"]
        for item in profile["columns"]
        if item["detected_type"] == "numeric"
    ]
    category_columns = [
        item["name"]
        for item in profile["columns"]
        if item["detected_type"] in {"categorical", "boolean"}
    ]
    processed_rows = []
    for category in category_columns[:5]:
        for measure in numeric_columns[:5]:
            grouped_data = data[[category, measure]].copy()
            grouped_data[measure] = pd.to_numeric(
                grouped_data[measure],
                errors="coerce",
            )
            grouped_data[measure] = grouped_data[measure].where(
                np.isfinite(grouped_data[measure]),
                np.nan,
            )
            grouped = (
                grouped_data.groupby(category, dropna=False)[measure]
                .agg(["count", "mean", "sum"])
                .dropna(subset=["mean", "sum"])
                .sort_values("sum", ascending=False)
                .head(15)
            )
            for category_value, row in grouped.iterrows():
                processed_rows.append(
                    {
                        "Dimension": category,
                        "Category": (
                            "<missing>"
                            if pd.isna(category_value)
                            else str(category_value)
                        ),
                        "Measure": measure,
                        "Count": int(row["count"]),
                        "Mean": float(row["mean"]),
                        "Sum": float(row["sum"]),
                        "Aggregation": "Mean and Sum",
                    }
                )
    processed_frame = pd.DataFrame(
        processed_rows,
        columns=[
            "Dimension",
            "Category",
            "Measure",
            "Count",
            "Mean",
            "Sum",
            "Aggregation",
        ],
    )

    finding_rows = []
    for kind, items in (
        ("Finding", insights["findings"]),
        ("Recommendation", insights["recommendations"]),
        ("Limitation", insights["limitations"]),
    ):
        finding_rows.extend(
            {"Type": kind, "Statement": item} for item in items
        )
    findings_frame = pd.DataFrame(
        finding_rows,
        columns=["Type", "Statement"],
    )
    date_column = next(
        (
            item["name"]
            for item in profile["columns"]
            if item["detected_type"] == "date"
        ),
        None,
    )
    measure_column = next(
        (
            item["name"]
            for item in profile["columns"]
            if item["detected_type"] == "numeric"
        ),
        None,
    )
    time_series = pd.DataFrame(columns=["Period", "Mean"])
    time_series_title = None
    if date_column and measure_column:
        dates = pd.to_datetime(
            data[date_column],
            errors="coerce",
            format="mixed",
            utc=True,
        ).dt.tz_convert(None)
        values = pd.to_numeric(data[measure_column], errors="coerce")
        values = values.where(np.isfinite(values), np.nan)
        pairs = pd.DataFrame(
            {
                "Date": dates,
                "Value": values,
            }
        ).dropna()
        if not pairs.empty:
            pairs["Period"] = period_starts(
                pairs["Date"],
                time_granularity,
            )
            time_series = (
                pairs.groupby("Period", as_index=False)["Value"]
                .mean()
                .rename(columns={"Value": f"Mean {measure_column}"})
            )
            time_series_title = (
                f"{granularity_label(time_granularity)} mean of "
                f"{measure_column} by {date_column}"
            )
    return {
        "profile": profile,
        "insights": insights,
        "analyzed_at": analyzed_at,
        "summary": summary_frame,
        "quality": quality_frame,
        "statistics": statistical_frame,
        "categorical": categorical_frame,
        "processed": processed_frame,
        "findings": findings_frame,
        "time_series": time_series,
        "time_series_title": time_series_title,
        "time_granularity": time_granularity,
    }


def create_excel_report(context):
    """Return a multi-sheet Excel report with formula-safe text values."""
    output = BytesIO()
    sheets = (
        ("Dataset Summary", context["summary"]),
        ("Data Quality", context["quality"]),
        ("Statistical Analysis", context["statistics"]),
        ("Categorical Analysis", context["categorical"]),
        ("Analytical Findings", context["findings"]),
        ("Processed Results", context["processed"]),
        ("Time Series", context["time_series"]),
    )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, frame in sheets:
            _excel_safe_frame(frame).to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_length = min(
                    max(len(str(cell.value or "")) for cell in column_cells) + 2,
                    60,
                )
                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = max(max_length, 12)
    output.seek(0)
    return output.getvalue()


class _DrawingFlowable(Flowable):
    def __init__(self, drawing):
        super().__init__()
        self.drawing = drawing
        self.width = drawing.width
        self.height = drawing.height

    def wrap(self, available_width, available_height):
        self.width = min(self.drawing.width, available_width)
        return self.width, self.drawing.height

    def draw(self):
        renderPDF.draw(self.drawing, self.canv, 0, 0)


def _report_chart(data, profile):
    numeric_columns = [
        item["name"]
        for item in profile["columns"]
        if item["detected_type"] == "numeric"
    ]
    category_columns = [
        item["name"]
        for item in profile["columns"]
        if item["detected_type"] in {"categorical", "boolean"}
    ]
    if not numeric_columns:
        return None

    measure = numeric_columns[0]
    if category_columns:
        category = category_columns[0]
        chart_data = data[[category, measure]].copy()
        chart_data[measure] = pd.to_numeric(
            chart_data[measure],
            errors="coerce",
        )
        grouped = (
            chart_data.dropna(subset=[measure])
            .groupby(category, dropna=False)[measure]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        labels = [
            "<missing>" if pd.isna(label) else str(label)[:18]
            for label in grouped.index
        ]
        values = grouped.astype(float).tolist()
        title = f"Mean {measure} by {category}"
    else:
        values = pd.to_numeric(data[measure], errors="coerce").dropna()
        if values.empty:
            return None
        counts, edges = np.histogram(values, bins=min(10, max(3, int(np.sqrt(len(values)))))
        )
        labels = [
            f"{edges[index]:.3g}–{edges[index + 1]:.3g}"
            for index in range(len(counts))
        ]
        values = counts.astype(float).tolist()
        title = f"{measure} distribution (counts)"

    if not values:
        return None
    drawing = Drawing(175 * mm, 75 * mm)
    chart = VerticalBarChart()
    chart.x = 12 * mm
    chart.y = 10 * mm
    chart.width = 150 * mm
    chart.height = 55 * mm
    chart.data = [values]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.angle = 25
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = min(0, min(values))
    chart.valueAxis.valueMax = max(values) * 1.15 if max(values) > 0 else 1
    chart.valueAxis.valueStep = max(
        (chart.valueAxis.valueMax - chart.valueAxis.valueMin) / 5,
        1,
    )
    chart.bars[0].fillColor = colors.HexColor("#4385ff")
    drawing.add(chart)
    return title, _DrawingFlowable(drawing)


def create_pdf_report(data, dataset_name, context):
    """Generate an evidence-based PDF report with a measured chart."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"RetailPulse AI report - {dataset_name}",
        author="RetailPulse AI",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_LEFT,
            textColor=colors.HexColor("#10213a"),
            fontSize=22,
            leading=27,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallReport",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
        )
    )
    story = [
        Paragraph("RetailPulse AI — Analytical Report", styles["ReportTitle"]),
        Paragraph(f"Dataset: {escape(str(dataset_name))}", styles["Heading2"]),
        Paragraph(
            f"Generated (UTC): {escape(context['analyzed_at'])}",
            styles["BodyText"],
        ),
        Spacer(1, 5 * mm),
        Paragraph("Executive summary", styles["Heading1"]),
    ]
    for finding in context["insights"]["findings"][:5]:
        story.append(Paragraph(f"• {escape(finding)}", styles["BodyText"]))
    summary = context["profile"]["summary"]
    story.extend(
        [
            Spacer(1, 4 * mm),
            Paragraph("Dataset overview", styles["Heading1"]),
            _pdf_table(
                [
                    ["Rows", f"{summary['rows']:,}"],
                    ["Columns", f"{summary['columns']:,}"],
                    ["Missing cells", f"{summary['missing_cells']:,}"],
                    ["Duplicate rows", f"{summary['duplicate_rows']:,}"],
                    ["Quality issues", f"{summary['issue_count']:,}"],
                    ["Memory (bytes)", f"{summary['memory_bytes']:,}"],
                ],
                header=False,
            ),
            Spacer(1, 4 * mm),
            Paragraph("Data quality assessment", styles["Heading1"]),
        ]
    )
    if context["quality"].empty:
        story.append(
            Paragraph(
                "No profiling issues were detected by the configured checks.",
                styles["BodyText"],
            )
        )
    else:
        quality_rows = [
            ["Issue", "Column", "Rows", "Severity", "Recommendation"]
        ]
        for row in context["quality"].to_dict("records"):
            quality_rows.append(
                [
                    str(row["Issue"]),
                    str(row["Column"]) or "—",
                    str(row["Affected rows"]),
                    str(row["Severity"]),
                    str(row["Recommendation"]),
                ]
            )
        story.append(_pdf_table(quality_rows, [24 * mm, 24 * mm, 14 * mm, 18 * mm, 90 * mm]))

    story.extend(
        [
            Spacer(1, 4 * mm),
            Paragraph("Statistical findings", styles["Heading1"]),
        ]
    )
    if context["statistics"].empty:
        story.append(
            Paragraph("No numeric columns were detected.", styles["BodyText"])
        )
    else:
        stats_rows = [
            ["Column", "Valid", "Missing", "Min", "Median", "Mean", "Max"]
        ]
        for row in context["statistics"].to_dict("records"):
            stats_rows.append(
                [
                    str(row["Column"]),
                    str(row["Valid values"]),
                    str(row["Missing values"]),
                    _display_number(row["Minimum"]),
                    _display_number(row["Median"]),
                    _display_number(row["Mean"]),
                    _display_number(row["Maximum"]),
                ]
            )
        story.append(_pdf_table(stats_rows))

    chart = _report_chart(data, context["profile"])
    if chart:
        story.extend(
            [
                Spacer(1, 4 * mm),
                Paragraph(escape(chart[0]), styles["Heading2"]),
                chart[1],
            ]
        )

    story.extend(
        [
            Spacer(1, 4 * mm),
            Paragraph("Time-series analysis", styles["Heading1"]),
        ]
    )
    time_series = context["time_series"]
    if time_series.empty:
        story.append(
            Paragraph(
                "No valid date and numeric column pair was available for "
                "daily or weekly time-series analysis.",
                styles["BodyText"],
            )
        )
    else:
        story.append(
            Paragraph(
                escape(context["time_series_title"]),
                styles["Heading2"],
            )
        )
        time_series_rows = [
            [str(column) for column in time_series.columns]
        ]
        time_series_rows.extend(
            [
                [
                    pd.Timestamp(row[0]).strftime("%Y-%m-%d"),
                    _display_number(row[1]),
                ]
                for row in time_series.head(60).itertuples(index=False, name=None)
            ]
        )
        story.append(_pdf_table(time_series_rows))
        if len(time_series) > 60:
            story.append(
                Paragraph(
                    f"PDF table limited to the first 60 of "
                    f"{len(time_series):,} periods; the Excel workbook "
                    "contains the complete time series.",
                    styles["SmallReport"],
                )
            )

    story.extend(
        [
            PageBreak(),
            Paragraph("Important insights", styles["Heading1"]),
        ]
    )
    for item in context["insights"]["findings"]:
        story.append(Paragraph(f"• {escape(item)}", styles["BodyText"]))
    story.extend(
        [
            Spacer(1, 4 * mm),
            Paragraph("Recommendations", styles["Heading1"]),
        ]
    )
    for item in context["insights"]["recommendations"]:
        story.append(Paragraph(f"• {escape(item)}", styles["BodyText"]))
    story.extend(
        [
            Spacer(1, 4 * mm),
            Paragraph("Methodology and limitations", styles["Heading1"]),
            Paragraph(
                "The report is generated from deterministic profile and "
                "summary calculations. Numeric statistics use valid finite "
                "values; potential outliers use the 1.5 × IQR rule. "
                "Categorical tables show the most frequent values. No source "
                "data is modified.",
                styles["BodyText"],
            ),
        ]
    )
    for item in context["insights"]["limitations"]:
        story.append(Paragraph(f"• {escape(item)}", styles["BodyText"]))
    story.append(
        Paragraph(
            "Charts and statistical associations are descriptive; they do "
            "not establish causation. Validate findings with domain context.",
            styles["BodyText"],
        )
    )
    document.build(story)
    output.seek(0)
    return output.getvalue()


def _display_number(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.4g}"


def _pdf_table(rows, widths=None, header=True):
    wrapped = []
    for row in rows:
        wrapped.append(
            [
                Paragraph(
                    escape(str(value)),
                    getSampleStyleSheet()["BodyText"],
                )
                for value in row
            ]
        )
    table = Table(
        wrapped,
        colWidths=widths,
        repeatRows=1 if header else 0,
    )
    style_commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style_commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#10213a")),
            ]
        )
    table.setStyle(
        TableStyle(style_commands)
    )
    return table
