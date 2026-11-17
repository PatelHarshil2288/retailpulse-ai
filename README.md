# RetailPulse AI 📊

**Interactive Retail Analytics Dashboard | Python · Plotly Dash · Pandas**

RetailPulse AI is an interactive retail analytics portfolio project built with Python, Pandas, Plotly, and Dash. Its Linux container is configured for Hostless as the primary host, with Render retained as a secondary target. Both use the same application and Dockerfile.

## 📸 Dashboard Preview

<!-- Add a dashboard screenshot here after capturing the running application. -->

*Interactive dashboard preview coming soon.*

## 🎯 Project Objectives

- Analyze historical retail sales and revenue trends.
- Monitor key performance indicators (KPIs) through an interactive dashboard.
- Compare product and country performance.
- Improve data quality through transaction validation and classification.
- Provide filtered transaction data through CSV export.
- Maintain a reproducible workflow with automated tests.

## ✨ Features

### Interactive Sales Dashboard

- **Gross Revenue:** Total revenue for the selected filters.
- **Unique Orders:** Number of distinct invoices.
- **Average Order Value:** Revenue divided by unique orders.
- **Monthly Revenue Growth:** Month-over-month revenue comparison.
- **Monthly Performance:** Explore monthly revenue, orders, or average order value.
- **Top Products:** Identify products with the highest revenue.
- **Country Performance:** Compare revenue across countries.
- **Revenue Contribution:** Explore each country's share of revenue.
- **Product Association Analysis:** Surface products frequently purchased together in the same invoice.
- **Top Customer Insights:** Rank the most valuable customer accounts by revenue, invoice count, and average order value.
- **Interactive Filters:** Filter results by country and date range.
- **Reset Filters:** Restore the default country and date selection.
- **CSV Export:** Download filtered sales transactions.
- **Your Data:** Upload a CSV or XLSX file (up to 10 MB) for a bounded column profile, missing-value summary, and sample preview. Excel workbooks support worksheet selection; datasets can be downloaded as CSV.
- **Data Quality Lab:** Review detected column types, missing values, exact duplicates, IQR-based outlier indicators, quality findings, and non-destructive cleaning recommendations for uploaded datasets.
- **Visual Explorer:** Open on a monthly revenue trend, compare retail dimensions with bar, donut, or line charts, or switch to type-aware universal analysis for uploaded tables, including grouped charts, numeric statistics, correlations, date trends, filters, previews, and CSV export.

Uploads are processed in memory and are not written to disk. The workspace accepts up to 100,000 rows and 200 columns. Uploads that include the eight retail transaction columns listed on the Your Data page can also be activated in Retail Intelligence, with prices interpreted in GBP. Other datasets remain available for inspection in Your Data and Universal Analytics; Retail Intelligence and the Data Quality Lab's retail-specific diagnostics continue to use the built-in UCI data when the upload is not retail-compatible. Refreshing the browser clears session uploads.

### Data Quality and Processing

The data-processing workflow supports schema validation, duplicate removal, date parsing, missing-value handling, revenue calculation, and transaction classification.

Transaction categories include ordinary sales, cancellations, returns or reversals, zero-price records, negative-price reviews, and accounting adjustments.

The dashboard analyzes ordinary sales transactions separately from records that require review.

The **Your Data** page accepts general CSV and XLSX datasets, detects common delimiters and encodings, normalizes blank or duplicate headers, and displays a data profile and bounded preview. Excel workbooks allow worksheet selection. A dataset becomes active in **Retail Intelligence** only when it has the complete RetailPulse transaction schema and can be processed for dashboard analytics; otherwise, the built-in UCI dataset remains active. Use **Use built-in dataset** to clear uploaded datasets and switch back to UCI Online Retail.

The **Data Quality Lab** profiles the uploaded dataset, including schema hints, column statistics, missingness, duplicate rows, potential numeric outliers, and date coverage when detectable. Findings include affected-record counts, severity, investigation guidance, and whether an automatic correction is considered safe; the source data is never silently modified. Retail-specific transaction classifications and cleaning-funnel details appear only for a compatible retail upload. The bundled UCI data is already preprocessed, so raw-file checks are not presented as though they were measured from its source.

The **Visual Explorer** retains its retail-focused view for the built-in UCI dataset. For any uploaded table, its universal mode detects column types, offers only compatible aggregations, filters and previews records, and exports grouped results. The same mode shows descriptive statistics for numeric columns, Pearson correlation for numeric pairs, monthly mean trends where date and numeric columns are available, and data-driven analysis suggestions. Categorical chart results are limited to the top 15 groups, and row previews to 10 rows. Correlation is descriptive and does not imply causation; trend aggregation is labeled explicitly.

## 🛠️ Technology Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Dashboard:** Plotly Dash
- **Visualization:** Plotly Express
- **Testing:** Pytest
- **Development Tools:** VS Code, Anaconda, Git, GitHub

## 📂 Dataset

**Source:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)

The dataset contains historical transactions from a UK-based online retail business.

The original dataset contains **541,909 transaction rows**. The current processing workflow removes exact duplicates and invalid records, classifies transactions, and separates ordinary sales from records requiring review.

The dataset is used for educational and analytical purposes. Please refer to the original UCI source for dataset details and usage information.

### Data files

Place the downloaded Excel file at:

`data/raw/Online Retail.xlsx`

Processed datasets are generated during the Docker image build by `prepare_deployment.py`. Raw and processed datasets remain excluded from Git and are created inside the container.

## 🚀 Getting Started

### Prerequisites

- Python compatible with the project's dependencies
- Git
- The UCI Online Retail dataset

### 1. Clone the repository

```bash
git clone https://github.com/PatelHarshil2288/retailpulse-ai.git
cd retailpulse-ai
```

### 2. Create and activate an environment

Using Anaconda:

```bash
conda create -n retailpulse python=3.12
conda activate retailpulse
```

The declared dependency ranges have been verified with Python 3.12 and 3.14. The Docker image uses Python 3.12.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

From the repository root, run the data-preparation script:

```bash
python prepare_deployment.py
```

The script downloads the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) if `data/raw/Online Retail.xlsx` is not present, then creates the processed CSV files required by the dashboard. The Docker image runs this same script during its build.

### 5. Run the dashboard locally

```bash
python -m src.app
```

Open the local address in your browser:

http://127.0.0.1:8050/

### 6. Build and run the Docker image locally

Install Docker Desktop, then from the repository root run:

```bash
docker build -t retailpulse-ai .
docker run --rm -p 7860:7860 retailpulse-ai
```

Open http://127.0.0.1:7860/. The image prepares the UCI dataset and processed CSVs during its build.

### 7. Deploy to Hostless

Create a Hostless App, connect the GitHub repository and `main` branch, and select **Docker** as the build system. The repository's `hostless.yaml` selects the Dockerfile, sets the HTTP health check, and caps runtime requests at the free pool's 0.5 vCPU and 1 GiB RAM for one replica. The container listens on Hostless's `PORT` environment variable.

The Hostless free allowance is shared across all apps, workers, and databases in the account. The app is configured to stay within that allowance by itself; other resources in the account also consume it. See [Hostless pricing](https://hostless.cloud/pricing) and the [Hostless first-app guide](https://docs.hostless.cloud/started).

### 8. Render backup

The repository includes a Render Blueprint configured to build the same `Dockerfile`. If Hostless has an availability or compatibility issue, create a Render web service from the repository and apply `render.yaml`. The container reads Render's `PORT` environment variable, so the application code does not need a hosting-specific rewrite.

## 🧪 Automated Testing

The project includes automated tests for data processing and dashboard behavior.

Run the complete test suite from the repository root:

```bash
python -m pytest -v
```

The latest verified test run reported **126 passed tests and one non-blocking warning**. Test results may change as the project evolves.

The tests cover data validation, transaction classification, CSV/XLSX upload handling, generic dataset profiling and quality findings, universal statistics, grouping, correlation, and trend analysis, upload-to-dashboard data selection, KPI calculations, country and date filtering, reset behavior, Visual Explorer aggregation/type checks, and raw/grouped CSV exports.

## 🗺️ Project Roadmap

Planned and implemented enhancements include:

- RFM-based customer segmentation. ✅
- Product association analysis and cross-selling insights. ✅
- Sales forecasting and model evaluation. ✅
- Additional customer-level analytics. ✅
- Top customer performance summaries and ranking. ✅
- Interactive Visual Explorer for the active retail dataset. ✅
- Universal analytics for uploaded tables: statistics, grouping, correlation, and trends. ✅
- Docker deployment to Hostless, with Render as a backup target. ⏳
- Dashboard screenshots and demonstration materials. ⏳

The project roadmap reflects implemented work once it is validated in the dashboard and test suite.

## 📌 Current Status

**Local Dash dashboard operational · Portable Docker deployment configured · Public deployment pending**

The full Python Dash application is preserved. Hostless is the primary target; Render is configured as a secondary target using the same Docker image.

## 👨‍💻 Author

**Harshil Patel**

Computer Science Engineering | Data Science & AI

- GitHub: [PatelHarshil2288](https://github.com/PatelHarshil2288)

## 📄 Responsible Use

This project is intended for educational and portfolio purposes. Analytical findings should be interpreted in the context of the underlying dataset. Future forecasts and customer segments will require appropriate validation before being presented as business recommendations.

## 📜 Acknowledgements

Dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail).