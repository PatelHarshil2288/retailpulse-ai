# RetailPulse AI 📊

**Interactive Retail Analytics Dashboard | Python · Plotly Dash · Pandas**

RetailPulse AI is an interactive retail analytics portfolio project that transforms historical transaction data into actionable business insights. Built with Python, Pandas, Plotly, and Dash, the application helps users explore sales performance, identify top-performing products and countries, and monitor key business metrics through an interactive dashboard.

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
- **Interactive Filters:** Filter results by country and date range.
- **Reset Filters:** Restore the default country and date selection.
- **CSV Export:** Download filtered sales transactions.

### Data Quality and Processing

The data-processing workflow supports schema validation, duplicate removal, date parsing, missing-value handling, revenue calculation, and transaction classification.

Transaction categories include ordinary sales, cancellations, returns or reversals, zero-price records, negative-price reviews, and accounting adjustments.

The dashboard analyzes ordinary sales transactions separately from records that require review.

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

Processed datasets are generated or stored under `data/processed/` according to the project's data-processing workflow. Raw and processed datasets should not be committed to GitHub unless there is a deliberate reason to publish them.

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

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) and place `Online Retail.xlsx` inside `data/raw/`.

Follow the project's data-processing workflow to create the processed files required by the dashboard.

### 5. Run the dashboard

```bash
python src/app.py
```

Open the local address in your browser:

http://127.0.0.1:8050/

## 🧪 Automated Testing

The project includes automated tests for data processing and dashboard behavior.

Run the complete test suite from the repository root:

```bash
python -m pytest -v
```

The latest verified test run reported **38 passed tests and one non-blocking warning**. Test results may change as the project evolves.

The tests cover data validation, transaction classification, KPI calculations, country and date filtering, reset behavior, callback outputs, and CSV export.

## 🗺️ Project Roadmap

Planned enhancements include:

- RFM-based customer segmentation.
- Product association analysis and cross-selling insights.
- Sales forecasting and model evaluation.
- Additional customer-level analytics.
- Deployment of a publicly accessible dashboard.
- Dashboard screenshots and demonstration materials.

These features will be documented as implemented only after they are developed and tested.

## 📌 Current Status

**Local dashboard operational · Automated tests passing · Public deployment pending**

The dashboard currently runs locally. A public deployment URL will be added after deployment and validation are complete.

## 👨‍💻 Author

**Harshil Patel**

Computer Science Engineering | Data Science & AI

- GitHub: [PatelHarshil2288](https://github.com/PatelHarshil2288)

## 📄 Responsible Use

This project is intended for educational and portfolio purposes. Analytical findings should be interpreted in the context of the underlying dataset. Future forecasts and customer segments will require appropriate validation before being presented as business recommendations.

## 📜 Acknowledgements

Dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail).