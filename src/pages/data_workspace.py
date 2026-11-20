"""Upload and validate retail transaction datasets."""

import base64
import binascii
import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile
import zlib

import pandas as pd
from dash import dcc, html

from src.data_processing import (
    REQUIRED_COLUMNS,
    clean_transactions,
    create_quality_report,
    get_sales_transactions,
)


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_BASE64_LENGTH = 4 * ((MAX_UPLOAD_BYTES + 2) // 3)
MAX_DASHBOARD_PAYLOAD_BYTES = 8 * 1024 * 1024
MAX_DASHBOARD_JSON_BYTES = 64 * 1024 * 1024
MAX_DASHBOARD_SALES_ROWS = 100_000
MAX_WORKSPACE_COLUMNS = 200
MAX_WORKSPACE_CELLS = 2_000_000
MAX_XLSX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_XLSX_MEMBERS = 10_000
SUPPORTED_SUFFIXES = {".csv", ".xlsx"}
PREVIEW_ROW_LIMIT = 10
WORKSPACE_PREVIEW_COLUMN_LIMIT = 10
DASHBOARD_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "InvoiceDate",
    "Quantity",
    "UnitPrice",
    "Revenue",
    "Country",
    "CustomerID",
    "Product",
]


@dataclass(frozen=True)
class UploadAnalysis:
    """Validated and cleaned results for one uploaded retail dataset."""

    quality_report: dict
    cleaned_transactions: pd.DataFrame
    sales_rows: int


@dataclass(frozen=True)
class DatasetUpload:
    """A parsed dataset and its workbook sheet metadata."""

    data: pd.DataFrame
    sheet_names: tuple[str, ...]
    selected_sheet: str | None


def serialize_dashboard_sales(cleaned_transactions):
    """Compress validated sales for this browser's temporary Dash store."""
    sales_transactions = get_sales_transactions(cleaned_transactions).copy()
    if sales_transactions.empty:
        raise ValueError(
            "The file has no ordinary sales rows to analyze. "
            "The built-in dataset remains active."
        )
    if len(sales_transactions) > MAX_DASHBOARD_SALES_ROWS:
        raise ValueError(
            "The file contains more than 100,000 ordinary sales rows. "
            "Reduce the dataset before loading it into the dashboard."
        )

    sales_transactions["InvoiceDate"] = pd.to_datetime(
        sales_transactions["InvoiceDate"],
        errors="coerce",
    )
    sales_transactions["Product"] = (
        sales_transactions["StockCode"].astype(str)
        + " - "
        + sales_transactions["Description"].fillna("Unknown").astype(str)
    )
    encoded_json = sales_transactions[DASHBOARD_COLUMNS].to_json(
        orient="split",
        date_format="iso",
        double_precision=15,
    ).encode("utf-8")
    compressed = zlib.compress(encoded_json, level=6)
    if len(compressed) > MAX_DASHBOARD_PAYLOAD_BYTES:
        raise ValueError(
            "This dataset is too large for browser-session analytics. "
            "Upload fewer rows or columns and try again."
        )
    return base64.b64encode(compressed).decode("ascii")


def deserialize_dashboard_sales(encoded_payload):
    """Restore a bounded, validated sales frame from the browser store."""
    if not isinstance(encoded_payload, str):
        raise ValueError("The uploaded dashboard data is invalid.")

    try:
        compressed = base64.b64decode(encoded_payload, validate=True)
        if len(compressed) > MAX_DASHBOARD_PAYLOAD_BYTES:
            raise ValueError("The uploaded dashboard data is too large.")

        decompressor = zlib.decompressobj()
        json_bytes = decompressor.decompress(
            compressed,
            MAX_DASHBOARD_JSON_BYTES + 1,
        )
        if (
            len(json_bytes) > MAX_DASHBOARD_JSON_BYTES
            or decompressor.unconsumed_tail
        ):
            raise ValueError("The uploaded dashboard data is too large.")
        json_bytes += decompressor.flush()
    except (binascii.Error, zlib.error) as error:
        raise ValueError("The uploaded dashboard data is invalid.") from error

    if len(json_bytes) > MAX_DASHBOARD_JSON_BYTES or not decompressor.eof:
        raise ValueError("The uploaded dashboard data is invalid.")

    try:
        sales_transactions = pd.read_json(
            StringIO(json_bytes.decode("utf-8")),
            orient="split",
        )
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise ValueError("The uploaded dashboard data is invalid.") from error

    if (
        len(sales_transactions) > MAX_DASHBOARD_SALES_ROWS
        or list(sales_transactions.columns) != DASHBOARD_COLUMNS
    ):
        raise ValueError("The uploaded dashboard data is invalid.")

    sales_transactions["InvoiceDate"] = pd.to_datetime(
        sales_transactions["InvoiceDate"],
        errors="coerce",
    )
    sales_transactions = sales_transactions.dropna(
        subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "Revenue"]
    )
    return sales_transactions.reset_index(drop=True)


def _decode_upload(contents, filename):
    """Validate an upload envelope and return its suffix and file bytes."""
    if not filename:
        raise ValueError("Choose a CSV or XLSX file to upload.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Unsupported file type. Upload a .csv or .xlsx file.")

    if not contents or "," not in contents:
        raise ValueError("The uploaded file is empty or could not be decoded.")

    encoded = contents.split(",", 1)[1]
    if len(encoded) > MAX_UPLOAD_BASE64_LENGTH:
        raise ValueError("The file exceeds the 10 MB upload limit.")

    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("The uploaded file could not be decoded.") from error

    if not payload:
        raise ValueError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("The file exceeds the 10 MB upload limit.")
    return suffix, payload


def _validate_xlsx_payload(payload):
    """Reject XLSX archives with excessive declared expansion or members."""
    try:
        with ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
    except (BadZipFile, OSError) as error:
        raise ValueError(
            "The workbook could not be read. Upload a valid .xlsx file."
        ) from error

    if len(members) > MAX_XLSX_MEMBERS:
        raise ValueError(
            "The workbook contains too many internal files to process safely."
        )
    uncompressed_size = sum(member.file_size for member in members)
    if uncompressed_size > MAX_XLSX_UNCOMPRESSED_BYTES:
        raise ValueError(
            "The workbook expands beyond the supported 100 MB processing limit."
        )


def _read_csv_payload(payload):
    """Read delimited text using common encodings and detected separators."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = payload.decode(encoding)
        except UnicodeDecodeError:
            continue

        try:
            delimiter = csv.Sniffer().sniff(
                text[:8192],
                delimiters=",;\t|",
            ).delimiter
        except csv.Error:
            delimiter = ","

        try:
            csv_source = StringIO(text)
            header = pd.read_csv(
                csv_source,
                sep=delimiter,
                engine="python",
                on_bad_lines="error",
                nrows=0,
            )
            column_count = len(header.columns)
            if column_count > MAX_WORKSPACE_COLUMNS:
                raise ValueError(
                    f"The dataset exceeds the {MAX_WORKSPACE_COLUMNS}-column limit."
                )

            row_count = 0
            cell_count = column_count
            reader = csv.reader(
                StringIO(text, newline=""),
                delimiter=delimiter,
                strict=True,
            )
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                row_count += 1
                cell_count += len(row)
                if row_count > MAX_DASHBOARD_SALES_ROWS:
                    raise ValueError(
                        "The dataset exceeds the 100,000-row workspace limit. "
                        "Reduce the file size and try again."
                    )
                if cell_count > MAX_WORKSPACE_CELLS:
                    raise ValueError(
                        "The dataset exceeds the 2,000,000-cell workspace limit."
                    )

            return pd.read_csv(
                StringIO(text),
                sep=delimiter,
                engine="python",
                on_bad_lines="error",
                nrows=MAX_DASHBOARD_SALES_ROWS + 1,
            )
        except (
            csv.Error,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            ValueError,
        ) as error:
            if isinstance(error, ValueError) and str(error).startswith(
                "The dataset exceeds"
            ):
                raise
            raise ValueError(
                "The CSV could not be parsed. Check its encoding, delimiter, "
                "headers, quoted fields, and row structure."
            ) from error

    raise ValueError(
        "The CSV encoding is not supported. Save it as UTF-8 and try again."
    )


def _normalize_dataset_columns(data):
    """Trim headers and make blank or duplicate names unambiguous."""
    normalized = []
    seen = set()
    for index, column in enumerate(data.columns, start=1):
        name = str(column).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"Column {index}"

        base_name = name
        duplicate_number = 2
        while name in seen:
            name = f"{base_name} ({duplicate_number})"
            duplicate_number += 1
        seen.add(name)
        normalized.append(name)

    result = data.copy()
    result.columns = normalized
    return result


def list_uploaded_worksheet_names(contents, filename):
    """Return worksheet names for an uploaded XLSX workbook."""
    suffix, payload = _decode_upload(contents, filename)
    if suffix == ".csv":
        return ()
    _validate_xlsx_payload(payload)

    try:
        with pd.ExcelFile(BytesIO(payload), engine="openpyxl") as workbook:
            if not workbook.sheet_names:
                raise ValueError("The Excel workbook contains no worksheets.")
            return tuple(workbook.sheet_names)
    except (BadZipFile, OSError, ValueError) as error:
        raise ValueError(
            "The workbook could not be read. Upload a valid .xlsx file."
        ) from error


def process_uploaded_dataset(contents, filename, sheet_name=None):
    """Parse a generic CSV or XLSX dataset without imposing a retail schema."""
    suffix, payload = _decode_upload(contents, filename)
    sheet_names = ()
    selected_sheet = None
    if suffix == ".xlsx":
        _validate_xlsx_payload(payload)

    try:
        if suffix == ".csv":
            data = _read_csv_payload(payload)
        else:
            with pd.ExcelFile(BytesIO(payload), engine="openpyxl") as workbook:
                sheet_names = tuple(workbook.sheet_names)
                if not sheet_names:
                    raise ValueError(
                        "The Excel workbook contains no worksheets."
                    )
                selected_sheet = (
                    sheet_name if sheet_name in sheet_names else sheet_names[0]
                )
                worksheet = workbook.book[selected_sheet]
                if worksheet.max_row - 1 > MAX_DASHBOARD_SALES_ROWS:
                    raise ValueError(
                        "The selected worksheet exceeds the 100,000-row workspace limit."
                    )
                if worksheet.max_column > MAX_WORKSPACE_COLUMNS:
                    raise ValueError(
                        f"The selected worksheet exceeds the {MAX_WORKSPACE_COLUMNS}-column limit."
                    )
                if (
                    max(worksheet.max_row - 1, 0) * worksheet.max_column
                    > MAX_WORKSPACE_CELLS
                ):
                    raise ValueError(
                        "The selected worksheet exceeds the 2,000,000-cell workspace limit."
                    )
                data = pd.read_excel(
                    workbook,
                    sheet_name=selected_sheet,
                    nrows=MAX_DASHBOARD_SALES_ROWS + 1,
                )
    except (
        BadZipFile,
        OSError,
        UnicodeDecodeError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            (
                "The CSV could not be parsed",
                "The workbook contains",
                "The selected worksheet exceeds",
                "The dataset exceeds",
            )
        ):
            raise
        raise ValueError(
            "The file could not be read. Check that it is a valid CSV or XLSX "
            "with a header row."
        ) from error

    data = _normalize_dataset_columns(data)
    if data.shape[1] == 0:
        raise ValueError("The selected dataset contains no columns.")
    if len(data.columns) > MAX_WORKSPACE_COLUMNS:
        raise ValueError(
            f"The dataset exceeds the {MAX_WORKSPACE_COLUMNS}-column limit."
        )
    if len(data.index) * len(data.columns) > MAX_WORKSPACE_CELLS:
        raise ValueError(
            "The dataset exceeds the 2,000,000-cell workspace limit."
        )
    if len(data) > MAX_DASHBOARD_SALES_ROWS:
        raise ValueError(
            "The dataset exceeds the 100,000-row workspace limit. "
            "Reduce the file size and try again."
        )
    if data.empty:
        raise ValueError(
            "The selected dataset contains no data rows. Choose a non-empty "
            "worksheet or upload a file with records."
        )

    return DatasetUpload(
        data=data,
        sheet_names=sheet_names,
        selected_sheet=selected_sheet,
    )


def serialize_workspace_dataset(data):
    """Compress a bounded generic dataset for this browser's session store."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Workspace data must be a pandas DataFrame.")
    if data.empty or data.shape[1] == 0:
        raise ValueError("The dataset must contain rows and columns.")
    if len(data) > MAX_DASHBOARD_SALES_ROWS:
        raise ValueError(
            "The dataset exceeds the 100,000-row workspace limit. "
            "Reduce the file size and try again."
        )
    if len(data.columns) > MAX_WORKSPACE_COLUMNS:
        raise ValueError(
            f"The dataset exceeds the {MAX_WORKSPACE_COLUMNS}-column limit."
        )
    if len(data.index) * len(data.columns) > MAX_WORKSPACE_CELLS:
        raise ValueError(
            "The dataset exceeds the 2,000,000-cell workspace limit."
        )

    normalized = _normalize_dataset_columns(data)
    try:
        json_bytes = normalized.to_json(
            orient="split",
            date_format="iso",
            double_precision=15,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "The dataset contains values that cannot be stored for this "
            "browser session."
        ) from error
    if len(json_bytes) > MAX_DASHBOARD_JSON_BYTES:
        raise ValueError("The dataset is too large for browser-session analysis.")

    compressed = zlib.compress(json_bytes, level=6)
    if len(compressed) > MAX_DASHBOARD_PAYLOAD_BYTES:
        raise ValueError(
            "The dataset is too large for browser-session analysis. "
            "Upload fewer rows or columns and try again."
        )
    return base64.b64encode(compressed).decode("ascii")


def deserialize_workspace_dataset(encoded_payload):
    """Restore a bounded generic dataset from the browser session store."""
    if not isinstance(encoded_payload, str):
        raise ValueError("The uploaded workspace data is invalid.")

    try:
        compressed = base64.b64decode(encoded_payload, validate=True)
        if len(compressed) > MAX_DASHBOARD_PAYLOAD_BYTES:
            raise ValueError("The uploaded workspace data is too large.")

        decompressor = zlib.decompressobj()
        json_bytes = decompressor.decompress(
            compressed,
            MAX_DASHBOARD_JSON_BYTES + 1,
        )
        if (
            len(json_bytes) > MAX_DASHBOARD_JSON_BYTES
            or decompressor.unconsumed_tail
        ):
            raise ValueError("The uploaded workspace data is too large.")
        json_bytes += decompressor.flush()
    except (binascii.Error, zlib.error) as error:
        raise ValueError("The uploaded workspace data is invalid.") from error

    if len(json_bytes) > MAX_DASHBOARD_JSON_BYTES or not decompressor.eof:
        raise ValueError("The uploaded workspace data is invalid.")

    try:
        data = pd.read_json(
            StringIO(json_bytes.decode("utf-8")),
            orient="split",
        )
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise ValueError("The uploaded workspace data is invalid.") from error

    if (
        data.empty
        or len(data) > MAX_DASHBOARD_SALES_ROWS
        or len(data.columns) > MAX_WORKSPACE_COLUMNS
        or len(data.index) * len(data.columns) > MAX_WORKSPACE_CELLS
    ):
        raise ValueError("The uploaded workspace data is invalid.")
    return data


def process_retail_dataset(raw_transactions):
    """Apply the existing transaction cleaning to a retail-compatible frame."""
    missing_columns = REQUIRED_COLUMNS.difference(raw_transactions.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    quality_report = create_quality_report(raw_transactions)
    cleaned_transactions = clean_transactions(raw_transactions)
    sales_rows = len(get_sales_transactions(cleaned_transactions))

    return UploadAnalysis(
        quality_report=quality_report,
        cleaned_transactions=cleaned_transactions,
        sales_rows=sales_rows,
    )


def process_uploaded_data(contents, filename, sheet_name=None):
    """Decode, validate, and clean a retail-compatible CSV or XLSX dataset."""
    dataset = process_uploaded_dataset(contents, filename, sheet_name)
    return process_retail_dataset(dataset.data)


def build_data_workspace_page():
    """Build the browser-session data intake workspace."""
    return html.Div(
        className="main-content rp-data-workspace",
        children=[
            html.Section(
                className="rp-overview-hero",
                children=[
                    html.P("YOUR DATA  /  WORKSPACE", className="rp-eyebrow"),
                    html.H1("Bring a dataset into your workspace."),
                    html.P(
                        "Upload a CSV or Excel workbook to inspect its columns "
                        "and sample rows. Files stay in this browser session "
                        "and are not written to server disk. Retail-specific "
                        "dashboards are enabled when the required retail "
                        "columns are present.",
                        className="rp-overview-lede",
                    ),
                ],
            ),
            html.Section(
                className="rp-upload-section",
                **{"aria-labelledby": "upload-heading"},
                children=[
                    html.Div(
                        className="rp-section-heading",
                        children=[
                            html.Div(
                                children=[
                                    html.P("DATA INTAKE", className="rp-eyebrow"),
                                    html.H2(
                                        "Upload transactions",
                                        id="upload-heading",
                                    ),
                                ]
                            ),
                            html.P(
                                "CSV and .xlsx files · 10 MB maximum · Up to "
                                "100,000 rows, 200 columns, and 2,000,000 cells. "
                                "XLSX archives are limited to 100 MB uncompressed. "
                                "For GBP retail analytics, include InvoiceNo, "
                                "StockCode, Description, Quantity, InvoiceDate, "
                                "UnitPrice, CustomerID, and Country.",
                                className="rp-section-description",
                            ),
                        ],
                    ),
                    html.Label(
                        "Choose a CSV or Excel workbook",
                        htmlFor="retail-dataset-upload",
                        className="rp-upload-label",
                    ),
                    dcc.Upload(
                        id="retail-dataset-upload",
                        children=html.Div(
                            [
                                html.Strong("Choose a file"),
                                html.Span(" or drag it here"),
                                html.Small(
                                    "CSV and .xlsx · 10 MB maximum · "
                                    "processing limits apply"
                                ),
                            ]
                        ),
                        accept=".csv,.xlsx",
                        max_size=MAX_UPLOAD_BYTES,
                        multiple=False,
                        className="rp-upload-dropzone",
                    ),
                    html.Div(
                        id="retail-worksheet-field",
                        className="rp-worksheet-field rp-hidden",
                        children=[
                            html.Label(
                                "Worksheet",
                                htmlFor="retail-worksheet-selector",
                                className="rp-upload-label",
                            ),
                            dcc.Dropdown(
                                id="retail-worksheet-selector",
                                options=[],
                                value=None,
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        id="retail-upload-status",
                        className="rp-upload-status",
                        role="status",
                        **{"aria-live": "polite"},
                    ),
                    html.Div(id="retail-upload-results"),
                    html.Div(
                        className="rp-upload-actions",
                        children=[
                            html.Button(
                                "Download dataset as CSV",
                                id="retail-upload-download-button",
                                className="rp-primary-action",
                                disabled=True,
                            ),
                            html.Button(
                                "Use built-in dataset",
                                id="restore-default-data",
                                className="rp-secondary-action",
                            ),
                            dcc.Download(id="retail-upload-download"),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_upload_results(analysis):
    """Render a concise quality summary and bounded data preview."""
    report = analysis.quality_report
    cards = [
        ("Uploaded rows", report["rows"]),
        ("Duplicate rows", report["duplicate_rows"]),
        ("Cleaned rows", len(analysis.cleaned_transactions)),
        ("Ordinary sale rows", analysis.sales_rows),
        ("Missing customer IDs", report["missing_customer_ids"]),
    ]
    preview = analysis.cleaned_transactions.head(PREVIEW_ROW_LIMIT)
    headers = list(preview.columns)
    table = html.Div(
        className="rp-upload-table-wrap",
        children=html.Table(
            [
                html.Thead(html.Tr([html.Th(column) for column in headers])),
                html.Tbody(
                    [
                        html.Tr(
                            [html.Td("" if pd.isna(value) else str(value)) for value in row]
                        )
                        for row in preview.itertuples(index=False, name=None)
                    ]
                ),
            ]
        ),
    )
    return html.Div(
        className="rp-upload-results",
        children=[
            html.Div(
                className="rp-upload-metrics",
                children=[
                    html.Article(
                        className="rp-upload-metric",
                        children=[html.P(label), html.Strong(f"{value:,}")],
                    )
                    for label, value in cards
                ],
            ),
            html.Div(
                className="rp-upload-preview",
                children=[
                    html.H3("Cleaned data preview"),
                    html.P(
                        "Showing up to the first 10 cleaned rows. The download "
                        "contains all cleaned rows, including records marked "
                        "for review."
                    ),
                    table,
                ],
            ),
        ],
    )


def build_dataset_upload_results(data, sheet_name=None):
    """Render a generic dataset profile and a bounded sample preview."""
    row_count, column_count = data.shape
    missing_cells = int(data.isna().sum().sum())
    cards = [
        ("Rows", row_count),
        ("Columns", column_count),
        ("Missing cells", missing_cells),
    ]
    profile_rows = [
        html.Tr(
            [
                html.Td(str(column)),
                html.Td(str(data[column].dtype)),
                html.Td(f"{int(data[column].isna().sum()):,}"),
            ]
        )
        for column in data.columns
    ]
    profile = html.Div(
        className="rp-upload-table-wrap",
        children=html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Column"),
                            html.Th("Detected type"),
                            html.Th("Missing values"),
                        ]
                    )
                ),
                html.Tbody(profile_rows),
            ]
        ),
    )
    preview = data.iloc[:PREVIEW_ROW_LIMIT, :WORKSPACE_PREVIEW_COLUMN_LIMIT]
    preview_table = html.Div(
        className="rp-upload-table-wrap",
        children=html.Table(
            [
                html.Thead(
                    html.Tr([html.Th(str(column)) for column in preview.columns])
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [html.Td("" if pd.isna(value) else str(value)) for value in row]
                        )
                        for row in preview.itertuples(index=False, name=None)
                    ]
                ),
            ]
        ),
    )
    heading = "Dataset preview"
    if sheet_name:
        heading = f"Dataset preview: {sheet_name}"

    return html.Div(
        className="rp-upload-results",
        children=[
            html.Div(
                className="rp-upload-metrics",
                children=[
                    html.Article(
                        className="rp-upload-metric",
                        children=[html.P(label), html.Strong(f"{value:,}")],
                    )
                    for label, value in cards
                ],
            ),
            html.Div(
                className="rp-upload-preview",
                children=[
                    html.H3("Column profile"),
                    profile,
                    html.H3(heading, className="rp-dataset-preview-heading"),
                    html.P(
                        f"Showing the first {min(row_count, PREVIEW_ROW_LIMIT)} "
                        f"rows and up to {WORKSPACE_PREVIEW_COLUMN_LIMIT} columns."
                    ),
                    preview_table,
                ],
            ),
        ],
    )
