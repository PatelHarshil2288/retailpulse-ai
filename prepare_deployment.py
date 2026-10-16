
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from src.data_processing import (
    load_retail_data,
    clean_transactions,
    get_sales_transactions,
)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
)
ZIP_PATH = RAW_DIR / "online-retail.zip"
EXCEL_PATH = RAW_DIR / "Online Retail.xlsx"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not EXCEL_PATH.exists():
        print("Downloading the official UCI Online Retail dataset...")
        urlretrieve(DATA_URL, ZIP_PATH)

        print("Extracting the Excel dataset...")
        with ZipFile(ZIP_PATH, "r") as archive:
            excel_member = next(
                (
                    name
                    for name in archive.namelist()
                    if Path(name).name == "Online Retail.xlsx"
                ),
                None,
            )

            if excel_member is None:
                raise FileNotFoundError(
                    "Online Retail.xlsx was not found in the UCI ZIP archive."
                )

            with archive.open(excel_member) as source:
                with EXCEL_PATH.open("wb") as destination:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        destination.write(chunk)

        ZIP_PATH.unlink(missing_ok=True)

    print("Loading and cleaning transactions...")
    raw_df = load_retail_data(EXCEL_PATH)
    cleaned_df = clean_transactions(raw_df)
    sales_df = get_sales_transactions(cleaned_df)

    cleaned_df.to_csv(
        PROCESSED_DIR / "cleaned_transactions.csv", index=False
    )
    sales_df.to_csv(
        PROCESSED_DIR / "sales_transactions.csv", index=False
    )

    print(f"Cleaned transactions: {len(cleaned_df):,}")
    print(f"Ordinary sales transactions: {len(sales_df):,}")
    print("Deployment data preparation completed successfully.")


if __name__ == "__main__":
    main()
