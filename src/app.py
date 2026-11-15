
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
from src.revenue_forecasting import (
    create_revenue_forecast,
    evaluate_forecasting_models,
)

from src.customer_segmentation import (
    build_rfm_features_from_csv,
    prepare_rfm_features,
    fit_customer_segments,
)
from src.product_association import (
    build_product_associations,
    summarize_top_product_pair,
)
from src.customer_insights import (
    build_customer_value_summary,
    summarize_top_customers,
)

# --------------------------------------------------
# RetailPulse AI - Real Retail Analytics Dashboard
# Dataset: UCI Online Retail
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SALES_FILE = BASE_DIR / "data" / "processed" / "sales_transactions.csv"

if not SALES_FILE.exists():
    raise FileNotFoundError(
        f"Sales dataset not found: {SALES_FILE}\n"
        "Run src/data_processing.py first."
    )

# Load only the columns needed for this dashboard.
sales = pd.read_csv(
    SALES_FILE,
    usecols=[
        "InvoiceNo",
        "StockCode",
        "Description",
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
        "Revenue",
        "Country",
        "CustomerID",
    ],
    low_memory=False,
)

# Parse the ISO-style dates written by data_processing.py.
sales["InvoiceDate"] = pd.to_datetime(
    sales["InvoiceDate"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce",
)

# Validate essential fields.
sales = sales.dropna(
    subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "Revenue"]
)

sales = sales[
    (sales["Quantity"] > 0)
    & (sales["UnitPrice"] > 0)
].copy()
sales["Product"] = (
    sales["StockCode"].astype(str)
    + " - "
    + sales["Description"].fillna("Unknown").astype(str)
)


# Load transaction classifications for the data-quality summary.
CLEANED_FILE = Path("data/processed/cleaned_transactions.csv")

transaction_types = pd.read_csv(
    CLEANED_FILE,
    usecols=["TransactionType"],
)

transaction_counts = transaction_types["TransactionType"].value_counts()

total_cleaned_transactions = len(transaction_types)


# The sales dataset contains ordinary sales only.
sales["Month"] = sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

monthly = (
    sales.groupby("Month", as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
    )
    .sort_values("Month")
)

monthly["Average Order Value"] = (monthly["Revenue"] / monthly["Orders"])
top_products = (
    sales.groupby("Product", as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        Quantity_Sold=("Quantity", "sum"),
    )
    .sort_values("Revenue", ascending=False)
    .head(10)
)


total_revenue = sales["Revenue"].sum()
total_orders = sales["InvoiceNo"].nunique()
average_order_value = (
    total_revenue / total_orders if total_orders else 0
)


# Calculate headline business insights.
country_revenue = (
    sales.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

top_country = country_revenue.index[0] if not country_revenue.empty else "N/A"
top_country_revenue = (
    country_revenue.iloc[0] if not country_revenue.empty else 0
)
top_country_share = (
    (top_country_revenue / total_revenue * 100) if total_revenue else 0
)

product_revenue = (
    sales.groupby("Product")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

top_product = product_revenue.index[0] if not product_revenue.empty else "N/A"
top_product_revenue = (
    product_revenue.iloc[0] if not product_revenue.empty else 0
)

top_product_share = (
    (top_product_revenue / total_revenue * 100) if total_revenue else 0
)

monthly_revenue = sales.groupby("Month")["Revenue"].sum().sort_index()

best_month = (
    monthly_revenue.idxmax().strftime("%B %Y")
    if not monthly_revenue.empty
    else "N/A"
)
best_month_revenue = (
    monthly_revenue.max() if not monthly_revenue.empty else 0
)

customer_revenue = (
    sales.dropna(subset=["CustomerID"])
    .groupby("CustomerID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)
top_customer = (
    int(customer_revenue.index[0]) if not customer_revenue.empty else "N/A"
)
top_customer_revenue = (
    customer_revenue.iloc[0] if not customer_revenue.empty else 0
)
top_customer_share = (
    (top_customer_revenue / total_revenue * 100) if total_revenue else 0
)

chart_template = "plotly_white"

def apply_chart_theme(figure):
    """Apply consistent styling to RetailPulse AI Plotly charts."""
    chart_font = '"Inter", "Segoe UI", Arial, sans-serif'

    figure.update_layout(
        template=chart_template,
        title={
            "x": 0.02,
            "xanchor": "left",
            "font": {
                "family": chart_font,
                "size": 17,
                "color": "#172033",
            },
        },
        font={
            "family": chart_font,
            "size": 12,
            "color": "#475569",
        },
        hoverlabel={
            "bgcolor": "#14213d",
            "bordercolor": "#203f72",
            "font": {
                "family": chart_font,
                "size": 12,
                "color": "#ffffff",
            },
            "align": "left",
        },
        legend={
            "font": {
                "family": chart_font,
                "size": 11,
                "color": "#475569",
            },
            "bgcolor": "rgba(255, 255, 255, 0)",
            "borderwidth": 0,
        },
        margin={"l": 55, "r": 24, "t": 70, "b": 45},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="closest",
    )

    figure.update_xaxes(
        showline=False,
        zeroline=False,
        tickfont={"family": chart_font, "size": 11, "color": "#64748b"},
        title_font={"family": chart_font, "size": 12, "color": "#475569"},
    )

    figure.update_yaxes(
        showline=False,
        zeroline=False,
        tickfont={"family": chart_font, "size": 11, "color": "#64748b"},
        title_font={"family": chart_font, "size": 12, "color": "#475569"},
    )

    figure.update_traces(
        hoverlabel={
            "bgcolor": "#14213d",
            "bordercolor": "#203f72",
            "font": {
                "family": chart_font,
                "size": 12,
                "color": "#ffffff",
            },
        }
    )

    return figure


# --------------------------------------------------
# Historical customer segmentation (RFM + K-Means)
# --------------------------------------------------

rfm_features = build_rfm_features_from_csv(SALES_FILE)
scaled_rfm, rfm_scaler = prepare_rfm_features(rfm_features)

# Use the previously validated two-cluster solution.
customer_assignments, _, customer_segmentation_model = (
    fit_customer_segments(scaled_rfm, n_clusters=2)
)

# Build interpretable segment profiles from original, unscaled RFM values.
customer_segment_data = rfm_features.join(customer_assignments["Cluster"])

customer_segment_profiles = (
    customer_segment_data.groupby("Cluster")
    .agg(
        CustomerCount=("Recency", "size"),
        MedianRecency=("Recency", "median"),
        MedianFrequency=("Frequency", "median"),
        MedianMonetary=("Monetary", "median"),
        MeanRecency=("Recency", "mean"),
        MeanFrequency=("Frequency", "mean"),
        MeanMonetary=("Monetary", "mean"),
    )
    .reset_index()
)

customer_segment_profiles["CustomerPercentage"] = (
    customer_segment_profiles["CustomerCount"]
    / customer_segment_profiles["CustomerCount"].sum()
    * 100
)


customer_segment_profile_components = []

for _, row in customer_segment_profiles.iterrows():
    cluster_id = int(row["Cluster"])

    if cluster_id == 0:
        segment_name = "Recent, frequent, higher-spending"
    else:
        segment_name = "Less recent, less frequent"

    customer_segment_profile_components.append(
        html.Div(
            className="saas-card",
            style={"marginBottom": "12px"},
            children=[
                html.H4(segment_name),
                html.P(
                    f"{int(row['CustomerCount']):,} customers "
                    f"({row['CustomerPercentage']:.1f}%)"
                ),
                html.P(
                    f"Median recency: {row['MedianRecency']:.0f} days | "
                    f"Median purchases: {row['MedianFrequency']:.0f} invoices"
                ),
                html.P(
                    f"Median monetary value: £{row['MedianMonetary']:,.2f}",
                    style={"color": "#64748b"},
                ),
            ],
        )
    )


app = Dash(
    __name__,
    assets_folder="../assets",
)
server = app.server
app.title = "RetailPulse AI"


card_style = {
    "background": "white",
    "padding": "24px",
    "borderRadius": "18px",
    "border": "1px solid #e6ebf3",
    "boxShadow": "0 6px 24px rgba(22, 39, 75, 0.06)",
    "transition": "box-shadow 180ms ease",
}


app.layout = html.Div(
    style={
        "fontFamily": '"Inter", "Segoe UI", Arial, sans-serif',
        "backgroundColor": "#f3f6fc",
        "minHeight": "100vh",
        "color": "#172033",
        "boxSizing": "border-box",
        "display": "flex",
        "alignItems": "stretch",
        "padding": "0",
    },
    children=[
        html.Div(
            className="sidebar",
            children=[
                html.H2(
                    "RetailPulse",
                    style={
                        "color": "#ffffff",
                        "fontSize": "25px",
                        "fontWeight": "800",
                        "marginBottom": "4px",
                        "letterSpacing": "-1px",
                    },
                ),
                html.P(
                    "AI ANALYTICS",
                    style={
                        "color": "#9fb4e8",
                        "fontSize": "10px",
                        "fontWeight": "700",
                        "letterSpacing": "2px",
                        "marginTop": "0",
                        "marginBottom": "38px",
                    },
                ),
                html.P(
                    "WORKSPACE",
                    style={
                        "color": "#8497bd",
                        "fontSize": "10px",
                        "fontWeight": "700",
                        "letterSpacing": "1.5px",
                    },
                ),
                html.Div(
                    "▦   Overview",
                    className="sidebar-item sidebar-item-active",
                ),
                html.P(
                    "DATASET",
                    style={
                        "color": "#8497bd",
                        "fontSize": "10px",
                        "fontWeight": "700",
                        "letterSpacing": "1.5px",
                        "marginTop": "28px",
                    },
                ),
                html.Div(
                    "◫   Retail Analytics",
                    className="sidebar-item",
                ),
                html.Div(
                    "◎   Data Quality",
                    className="sidebar-item",
                ),
                html.Div(
                    "↗   Business Insights",
                    className="sidebar-item",
                ),
                html.Div(style={"flex": "1"}),
                html.P(
                    "ONLINE RETAIL",
                    style={
                        "color": "#8497bd",
                        "fontSize": "10px",
                        "fontWeight": "700",
                        "letterSpacing": "1.5px",
                    },
                ),
                html.P(
                    "Portfolio Dashboard",
                    style={
                        "color": "#ffffff",
                        "fontSize": "13px",
                    },
                ),
            ],
        ),
        html.Div(
            className="main-content",
            children=[
                html.Div(
                    className="dashboard-hero",
                    children=[
                        html.Div(
                            className="dashboard-hero-copy",
                            children=[
                                html.P(
                                    "EXECUTIVE OVERVIEW",
                                    className="dashboard-hero-eyebrow",
                                ),
                                html.H1(
                                    "RetailPulse AI",
                                    className="dashboard-hero-title",
                                ),
                                html.P(
                                    "Retail performance, revenue trends, and "
                                    "customer intelligence in one view.",
                                    className="dashboard-hero-subtitle",
                                ),
                                html.P(
                                    "UCI Online Retail  |  "
                                    "December 2010 – November 2011  |  "
                                    "Currency: GBP (£)",
                                    className="dashboard-hero-meta",
                                ),
                                html.P(
                                    "Revenue represents gross sales from "
                                    "positive-quantity, positive-price "
                                    "transactions; returns are not deducted.",
                                    className="dashboard-hero-note",
                                ),
                            ],
                        ),
                    ],
                ),

                html.Div(
                    className="kpi-grid",
                    children=[
                        html.Div(
                            className="kpi-card kpi-card-revenue",
                            children=[
                                html.P(
                                    "Gross Sales Revenue",
                                    className="kpi-label",
                                ),
                                html.H2(
                                    f"£{total_revenue:,.2f}",
                                    id="gross-revenue-value",
                                    className="kpi-value",
                                ),
                                html.P(
                                    "Total gross sales",
                                    className="kpi-description",
                                ),
                            ],
                        ),
                        html.Div(
                            className="kpi-card",
                            children=[
                                html.P(
                                    "Unique Orders",
                                    className="kpi-label",
                                ),
                                html.H2(
                                    f"{total_orders:,}",
                                    id="unique-orders-value",
                                    className="kpi-value",
                                ),
                                html.P(
                                    "Distinct sales invoices",
                                    className="kpi-description",
                                ),
                            ],
                        ),
                        html.Div(
                            className="kpi-card",
                            children=[
                                html.P(
                                    "Units Sold",
                                        className="kpi-label",
                                    ),
                                    html.H2(
                                        f"{sales['Quantity'].sum():,.0f}",
                                        id="units-sold-value",
                                        className="kpi-value",
                                    ),
                                    html.P(
                                        "Total units sold",
                                        className="kpi-description",
                                    ),
                                ],
                            ),
                        
                            html.Div(
                                className="kpi-card",
                                children=[
                                    html.P(
                                        "Unique Customers",
                                        className="kpi-label",
                                    ),
                                    html.H2(
                                        f"{sales['CustomerID'].nunique():,}",
                                        id="unique-customers-value",
                                        className="kpi-value",
                                    ),
                                    html.P(
                                        "Identified customers",
                                        className="kpi-description",
                                    ),
                                ],
                            ),
                        html.Div(
                            className="kpi-card",
                            children=[
                                html.P(
                                    "Repeat Customers",
                                    className="kpi-label",
                                ),
                                html.H2(
                                    id="repeat-customers-value",
                                    children="0",
                                    className="kpi-value",
                                ),
                                html.P(
                                    "Customers with multiple invoices",
                                    className="kpi-description",
                                ),
                            ],
                        ),
                        html.Div(
                            className="kpi-card",
                            children=[
                                html.P(
                                    "Repeat Customer Rate",
                                    className="kpi-label",
                                ),
                                html.H2(
                                    id="repeat-customer-rate-value",
                                    children="0.0%",
                                    className="kpi-value",
                                ),
                                html.P(
                                    "Share of identified customers",
                                    className="kpi-description",
                                ),
                            ],
                        ),
                        html.Div(
                            className="kpi-card",
                            children=[
                                html.P(
                                    "Average Order Value",
                                    className="kpi-label",
                                ),
                                html.H2(
                                    f"£{average_order_value:,.2f}",
                                    id="average-order-value",
                                    className="kpi-value",
                                ),
                                html.P(
                                    "Revenue per unique order",
                                    className="kpi-description",
                                ),
                            ],
                        ),
                        html.Div(
                            className="kpi-card kpi-card-growth",
                            children=[
                                html.P(
                                    "Monthly Revenue Growth",
                                    className="kpi-label",
                                ),
                                html.H2(
                                    id="monthly-growth-value",
                                    children="N/A",
                                    className="kpi-value",
                                ),
                                html.P(
                                    "Compared with the previous month",
                                    className="kpi-description",
                                ),
                            ],
                        ),
                    ],
                ),

        html.Div(
            className="saas-card",
            style={"marginBottom": "24px"},
            children=[
                html.Label(
                    "Filter by country:",
                    htmlFor="country-selector",
                    style={"fontWeight": "bold", "marginTop": "12px"},
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                        "gap": "16px",
                        "marginTop": "10px",
                        "marginBottom": "20px",
                    },
                    children=[
                        html.Div(
                            [
                                html.Label("Start date"),
                                dcc.DatePickerSingle(
                                    id="start-date",
                                    min_date_allowed=sales["InvoiceDate"].min().date(),
                                    max_date_allowed=sales["InvoiceDate"].max().date(),
                                    date=sales["InvoiceDate"].min().date(),
                                    display_format="DD MMM YYYY",
                                    clearable=False,
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label("End date"),
                                dcc.DatePickerSingle(
                                    id="end-date",
                                    min_date_allowed=sales["InvoiceDate"].min().date(),
                                    max_date_allowed=sales["InvoiceDate"].max().date(),
                                    date=sales["InvoiceDate"].max().date(),
                                    display_format="DD MMM YYYY",
                                    clearable=False,
                                ),
                            ]
                        ),
                    ],
                ),



        # Dataset Overview
        html.Div(
            className="saas-card",
            style={"marginBottom": "24px"},
            children=[
                html.H3(
                    "Dataset Overview",
                    style={"marginTop": "0", "color": "#172033"},
                ),
                html.P(
                    "A quick summary of the retail dataset.",
                    style={"color": "#64748b"},
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                        "gap": "16px",
                    },
                    children=[
                        html.Div(
                            [
                                html.P("Sales Records"),
                                html.H3(f"{len(sales):,}"),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Unique Products"),
                                html.H3(f"{sales['StockCode'].nunique():,}"),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Countries"),
                                html.H3(f"{sales['Country'].nunique():,}"),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Customers Identified"),
                                html.H3(f"{sales['CustomerID'].nunique():,}"),
                            ],
                            style=card_style,
                        ),
                    ],
                ),
            ],
        ),

        # Transaction Quality Overview
        html.Div(
            className="saas-card",
            style={"marginBottom": "24px"},
            children=[
                html.H3(
                    "Transaction Quality Overview",
                    style={"marginTop": "0", "color": "#172033"},
                ),
                html.P(
                    "Transaction classifications from the cleaned dataset.",
                    style={"color": "#64748b"},
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                        "gap": "16px",
                    },
                    children=[
                        html.Div(
                            [
                                html.P("Ordinary Sales"),
                                html.H3(f"{transaction_counts.get('sale', 0):,}"),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Cancellations"),
                                html.H3(
                                    f"{transaction_counts.get('cancellation', 0):,}"
                                ),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Returns / Reversals"),
                                html.H3(
                                    f"{transaction_counts.get('return_or_reversal', 0):,}"
                                ),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Zero-Price Reviews"),
                                html.H3(
                                    f"{transaction_counts.get('zero_price_review', 0):,}"
                                ),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Negative-Price Reviews"),
                                html.H3(
                                    f"{transaction_counts.get('negative_price_review', 0):,}"
                                ),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Accounting Adjustments"),
                                html.H3(
                                    f"{transaction_counts.get('accounting_adjustment', 0):,}"
                                ),
                            ],
                            style=card_style,
                        ),
                    ],
                ),
                html.P(
                    f"Total cleaned records: {total_cleaned_transactions:,}",
                    style={
                        "color": "#64748b",
                        "marginTop": "16px",
                        "marginBottom": "0",
                    },
                ),
            ],
        ),

        html.Div(
            className="saas-card",
            style={"marginBottom": "24px"},
            children=[
                html.H3(
                    "Business Insights",
                    style={"marginTop": "0", "color": "#172033"},
                ),
                html.P(
                    "Highlights calculated from the full sales dataset.",
                    style={"color": "#64748b"},
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                        "gap": "16px",
                    },
                    children=[
                        html.Div(
                            [
                                html.P("Top Country by Revenue"),
                                html.H3(top_country, id="top-country-value"),
                                html.P(
                                    f"Revenue: £{top_country_revenue:,.2f}",
                                    id="top-country-revenue",
                                    style={"color": "#64748b"},
                                ),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Top Product by Revenue"),
                                html.H3(top_product, id="top-product-value", style={"fontSize": "16px"}),
                                html.P(
                                    f"Revenue: £{top_product_revenue:,.2f}",
                                    id="top-product-revenue",
                                    style={"color": "#64748b"},
                                ),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Best Month by Revenue"),
                                html.H3(best_month, id="best-month-value"),
                                html.P(
                                    f"Revenue: £{best_month_revenue:,.2f}",
                                    id="best-month-revenue",
                                    style={"color": "#64748b"},
                                ),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Top Customer by Revenue"),
                                html.H3(
                                    f"Customer {top_customer}" if top_customer != "N/A" else "N/A",
                                    id="top-customer-value",
                                ),
                                html.P(
                                    f"Revenue: £{top_customer_revenue:,.2f}",
                                    id="top-customer-revenue",
                                    style={"color": "#64748b"},
                                ),
                            ],
                            style=card_style,
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            className="saas-card",
            style={"marginBottom": "24px"},
            children=[
                html.H3(
                    "Calculated Business Observations",
                    style={"marginTop": "0", "color": "#172033"},
                ),
                html.P(
                    "These findings are calculated from the current dataset and should be interpreted as evidence, not recommendations.",
                    style={"color": "#64748b"},
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                        "gap": "16px",
                    },
                    children=[
                        html.Div(
                            [
                                html.P("Revenue concentration"),
                                html.H3(f"{top_country_share:.1f}% of gross sales", id="country-concentration-value"),
                                html.P(
                                    f"{top_country} contributes £{top_country_revenue:,.2f} of revenue.",
                                    style={"color": "#64748b"},
                                ),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Highest-selling product"),
                                html.H3(top_product, id="product-concentration-value", style={"fontSize": "16px"}),
                                html.P(
                                    f"{top_product_share:.1f}% of gross sales, or £{top_product_revenue:,.2f}.",
                                    style={"color": "#64748b"},
                                ),
                            ],
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.P("Top customer contribution"),
                                html.H3(
                                    f"Customer {top_customer}" if top_customer != "N/A" else "N/A",
                                    id="customer-contribution-value",
                                ),
                                html.P(
                                    f"{top_customer_share:.1f}% of revenue, or £{top_customer_revenue:,.2f}." if top_customer != "N/A" else "No identifiable customer data available.",
                                    style={"color": "#64748b"},
                                ),
                            ],
                            style=card_style,
                        ),
                    ],
                ),
                html.Div(
                    id="business-insight-summary",
                    style={
                        "marginTop": "18px",
                        "paddingTop": "18px",
                        "borderTop": "1px solid #e6ebf3",
                        "color": "#334155",
                        "fontSize": "14px",
                        "lineHeight": "1.6",
                    },
                    children=[
                        html.P(
                            "Revenue concentration, top product contribution, and repeat-customer activity are summarized here for the current filtered view.",
                            style={"margin": "0"},
                        )
                    ],
                ),
            ],
        ),

                dcc.Dropdown(
                    id="country-selector",
                    options=[
                        {"label": "All Countries", "value": "All"}
                    ] + [
                        {"label": country, "value": country}
                        for country in sorted(sales["Country"].unique())
                    ],
                    value="All",
                    clearable=False,
                    style={"marginTop": "10px", "marginBottom": "20px"},
                ),

                html.Button(
                    "Reset Filters",
                    id="reset-filters",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#2563eb",
                        "color": "white",
                        "border": "none",
                        "padding": "10px 18px",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "fontWeight": "bold",
                        "marginBottom": "20px",
                    },
                ),

                html.Button(
                    "Export Filtered Data (CSV)",
                    id="export-csv-button",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#0f766e",
                        "color": "white",
                        "border": "none",
                        "padding": "10px 18px",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "fontWeight": "bold",
                        "marginBottom": "20px",
                        "marginLeft": "10px",
                    },
                ),
                dcc.Download(id="download-filtered-csv"),

                html.Label(
                    "Select metric:",
                    htmlFor="view-selector",
                    style={"fontWeight": "bold"},
                ),
                dcc.RadioItems(
                    id="view-selector",
                    options=[
                        {"label": " Revenue (£)", "value": "Revenue"},
                        {"label": " Orders", "value": "Orders"},
                        {
                            "label": " Average Order Value (£)",
                            "value": "Average Order Value",
                        },
                    ],
                    value="Revenue",
                    inline=True,
                    style={
                        "display": "flex",
                        "gap": "18px",
                        "flexWrap": "wrap",
                        "marginTop": "12px",
                    },
                ),
            ],
        ),

        html.Div(
                className="saas-card",
                children=[
                    dcc.Graph(
                        id="monthly-chart",
                        config={"displayModeBar": True, "responsive": True},
                    ),
                    html.H3(
                        "Top 10 Products by Revenue",
                        className="section-title",
                        style={"marginTop": "30px"},
                        ),
                    dcc.Dropdown(
                        id="product-metric-selector",
                        options=[
                            {
                                "label": "Revenue (£)",
                                "value": "Revenue",
                            },
                            {
                                "label": "Units Sold",
                                "value": "Units Sold",
                            },
                            {
                                "label": "Average Selling Price (£)",
                                "value": "Average Selling Price",
                            },
                        ],
                        value="Revenue",
                        clearable=False,
                        searchable=False,
                        style={
                            "maxWidth": "360px",
                            "marginTop": "12px",
                            "marginBottom": "12px",
                        },
                    ),
                                            
                        dcc.Graph(
                            id="top-products-chart",
                            figure=px.bar(
                                top_products.sort_values("Revenue", ascending=True),
                                x="Revenue",
                                y="Product",
                                orientation="h",
                                title="Top 10 Products",
                                labels={
                                    "Revenue": "Revenue (£)",
                                    "Product": "Product",
                                },
                                template=chart_template,
                            ),
                            config={
                                "displayModeBar": True,
                                "responsive": True,
                            },
                        ),

                    html.H3(
                        "Top 10 Countries by Revenue",
                        className="section-title",
                        style={"marginTop": "30px"},
                    ),
                    dcc.Graph(
                        id="country-performance-chart",
                    ),

                    html.H3(
                        "Revenue Contribution by Country",
                        className="section-title",
                        style={"marginTop": "30px"},
                    ),
                    dcc.Graph(
                        id="country-contribution-chart",
                    ),
                ],
            ),

            html.Div(
                className="saas-card",
                style={"marginTop": "24px"},
                children=[
                    html.H3(
                        "Product Association Analysis",
                        className="section-title",
                    ),
                    html.P(
                        "Products commonly purchased together in the same invoice suggest natural cross-sell opportunities and basket patterns.",
                        style={"color": "#64748b"},
                    ),
                    html.Div(
                        id="cross-sell-summary",
                        style={
                            "marginBottom": "12px",
                            "color": "#475569",
                            "fontSize": "13px",
                        },
                    ),
                    dcc.Graph(
                        id="cross-sell-chart",
                        config={
                            "displayModeBar": True,
                            "responsive": True,
                        },
                    ),
                ],
            ),

            html.Div(
                className="saas-card",
                style={"marginTop": "24px"},
                children=[
                    html.H3(
                        "Top Customers by Revenue",
                        className="section-title",
                    ),
                    html.P(
                        "High-value customer accounts ranked by accumulated revenue and order count for the current selection.",
                        style={"color": "#64748b"},
                    ),
                    html.Div(
                        id="customer-insights-summary",
                        style={
                            "marginBottom": "12px",
                            "color": "#475569",
                            "fontSize": "13px",
                        },
                    ),
                    dcc.Graph(
                        id="customer-insights-chart",
                        config={
                            "displayModeBar": True,
                            "responsive": True,
                        },
                    ),
                ],
            ),

            # Experimental revenue forecasting
            html.Div(
                className="saas-card",
                style={"marginTop": "24px"},
                children=[
                    html.H3(
                        "Revenue Forecast",
                        className="section-title",
                    ),
                    html.P(
                        "Experimental three-month forecast selected by "
                        "chronological backtesting. Historical data is limited, "
                        "so forecasts should not be treated as reliable "
                        "business predictions.",
                        style={"color": "#64748b"},
                    ),
                    html.Div(
                        id="forecast-model-summary",
                        style={
                            "marginBottom": "12px",
                            "color": "#475569",
                            "fontSize": "13px",
                        },
                    ),
                    dcc.Graph(
                        id="revenue-forecast-chart",
                        config={
                            "displayModeBar": True,
                            "responsive": True,
                        },
                    ),

                ],
            ),
            # Historical customer segmentation
            html.Div(
                className="saas-card",
                style={"marginTop": "24px"},
                children=[
                    html.H3(
                        "Customer Segmentation",
                        className="section-title",
                    ),
                    html.P(
                        "Historical RFM segmentation using customer "
                        "purchase recency, invoice frequency, and total "
                        "monetary value. Based on transactions ending "
                        "30 November 2011; this is not a forecast.",
                        style={"color": "#64748b"},
                    ),
                    dcc.Graph(
                        id="customer-segment-chart",
                        figure=px.bar(
                            customer_segment_profiles.assign(
                                Segment=customer_segment_profiles[
                                    "Cluster"
                                ].map({
                                    0: "Recent, frequent, higher-spending",
                                    1: "Less recent, less frequent",
                                })
                            ),
                            x="Segment",
                            y="CustomerCount",
                            color="Segment",
                            text="CustomerCount",
                            title="Customers by Segment",
                            labels={
                                "Segment": "Customer segment",
                                "CustomerCount": "Number of customers",
                            },
                            template=chart_template,
                        ),
                        config={
                            "displayModeBar": True,
                            "responsive": True,
                        },
                    ),
                    dcc.Graph(
                        id="customer-orders-chart",
                        figure=px.bar(
                            title="One-time vs. Repeat Customers",
                            template=chart_template,
                        ),
                        config={
                            "displayModeBar": True,
                            "responsive": True,
                        },
                    ),
                    html.H3(
                        "Segment Profiles",
                        className="section-title",
                        style={"marginTop": "24px"},
                    ),
                    *customer_segment_profile_components,
                ],
            ),

                html.Footer(
                    "RetailPulse AI | Portfolio project | Python, Pandas, Dash & Plotly",
                    style={
                        "textAlign": "center",
                        "padding": "24px 0 0 0",
                        "color": "#586174",
                        "fontSize": "13px",
                    },
                ),
            ],
        ),
    ],
)



@app.callback(
    Output("monthly-chart", "figure"),
    Output("gross-revenue-value", "children"),
    Output("unique-orders-value", "children"),
    Output("average-order-value", "children"),
    Output("monthly-growth-value", "children"),
    Output("top-products-chart", "figure"),
    Output("country-performance-chart", "figure"),
    Output("country-contribution-chart", "figure"),
    Output("customer-orders-chart", "figure"),
    Output("repeat-customers-value", "children"),
    Output("repeat-customer-rate-value", "children"),
    Output("units-sold-value", "children"),
    Output("unique-customers-value", "children"),
    Output("top-country-value", "children"),
    Input("view-selector", "value"),
    Input("product-metric-selector", "value"),
    Input("country-selector", "value"),
    Input("start-date", "date"),
    Input("end-date", "date"),
)
def update_chart(
    selected_metric,
    selected_product_metric,
    selected_country,
    start_date,
    end_date,
):
    # Filter sales by country.
    if selected_country == "All":
        filtered_sales = sales
    else:
        filtered_sales = sales.loc[
            sales["Country"] == selected_country
        ]

    if start_date:
        filtered_sales = filtered_sales.loc[
            filtered_sales["InvoiceDate"] >= pd.Timestamp(start_date)
        ]

    if end_date:
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        filtered_sales = filtered_sales.loc[
            filtered_sales["InvoiceDate"] < end_exclusive
        ]

    revenue = filtered_sales["Revenue"].sum()
    orders = filtered_sales["InvoiceNo"].nunique()
    average_value = revenue / orders if orders else 0
    units_sold = filtered_sales["Quantity"].sum()

    customer_id_column = filtered_sales.get("CustomerID")
    if customer_id_column is None:
        unique_customers = 0
        repeat_customers = 0
        repeat_customer_rate = 0.0
        customer_orders = pd.Series(dtype=float)
        one_time_customers = 0
    else:
        customer_orders = (
            filtered_sales.dropna(subset=["CustomerID"])
            .groupby("CustomerID")["InvoiceNo"]
            .nunique()
        )
        unique_customers = int(customer_id_column.nunique())
        repeat_customers = int((customer_orders > 1).sum())
        one_time_customers = int((customer_orders == 1).sum())
        repeat_customer_rate = (
            repeat_customers / len(customer_orders) * 100
            if len(customer_orders) > 0
            else 0
        )

    customer_orders_fig = px.bar(
        x=["One-time customers", "Repeat customers"],
        y=[one_time_customers, repeat_customers],
        labels={
            "x": "Customer type",
            "y": "Number of customers",
        },
        title="One-time vs. Repeat Customers",
        text=[one_time_customers, repeat_customers],
        template=chart_template,
    )
    customer_orders_fig.update_traces(
        textposition="outside",
        hovertemplate="%{x}<br>Customers: %{y:,}<extra></extra>",
    )
    customer_orders_fig.update_yaxes(rangemode="tozero", gridcolor="#e9edf5")
    customer_orders_fig = apply_chart_theme(customer_orders_fig)

    country_revenue = (
        filtered_sales.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    top_country = (
        country_revenue.index[0] if not country_revenue.empty else "N/A"
    )
    top_country_revenue = (
        country_revenue.iloc[0] if not country_revenue.empty else 0
    )

    product_revenue = (
        filtered_sales.groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    top_product = (
        product_revenue.index[0] if not product_revenue.empty else "N/A"
    )
    top_product_revenue = (
        product_revenue.iloc[0] if not product_revenue.empty else 0
    )

    monthly_revenue = (
        filtered_sales.groupby(
            filtered_sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
        )["Revenue"]
        .sum()
        .sort_index()
    )
    best_month = (
        monthly_revenue.idxmax().strftime("%B %Y")
        if not monthly_revenue.empty
        else "N/A"
    )
    best_month_revenue = (
        monthly_revenue.max() if not monthly_revenue.empty else 0
    )

    monthly_filtered = (
        filtered_sales.groupby(
            filtered_sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
        )
        .agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"))
        .rename_axis("Month")
        .reset_index()
        .sort_values("Month")
    )
    monthly_filtered["Average Order Value"] = (
        monthly_filtered["Revenue"].div(monthly_filtered["Orders"])
    )

    monthly_growth = "N/A"
    if not monthly_filtered.empty:
        monthly_calendar = monthly_filtered.set_index("Month")["Revenue"]
        monthly_calendar = monthly_calendar.asfreq("MS", fill_value=0)
        if len(monthly_calendar) >= 2:
            previous_revenue = monthly_calendar.iloc[-2]
            current_revenue = monthly_calendar.iloc[-1]
            if previous_revenue > 0:
                growth_percent = (
                    (current_revenue - previous_revenue) / previous_revenue
                ) * 100
                monthly_growth = f"{growth_percent:+.2f}%"
            elif current_revenue > 0:
                monthly_growth = "New sales"
            else:
                monthly_growth = "0.00%"

    labels = {
        "Revenue": "Gross sales revenue (£)",
        "Orders": "Unique orders",
        "Average Order Value": "Average order value (£)",
    }

    if monthly_filtered.empty:
        fig = px.bar(title=f"No monthly data available — {selected_country}", template=chart_template)
        fig.add_annotation(
            text="Try selecting a different country or date range.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#64748b"},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    else:
        fig = px.bar(
            monthly_filtered,
            x="Month",
            y=selected_metric,
            title=f"Monthly {selected_metric} — {selected_country}",
            template=chart_template,
            labels={"Month": "Month", selected_metric: labels[selected_metric]},
            custom_data=["Revenue", "Orders", "Average Order Value"],
        )
        fig.update_traces(
            hovertemplate=(
                "<b>Month: %{x|%B %Y}</b><br>"
                "Gross sales revenue (£): %{customdata[0]:,.2f}<br>"
                "Orders: %{customdata[1]:,.0f}<br>"
                "Average Order Value (£): %{customdata[2]:,.2f}"
                "<extra></extra>"
            ),
            hoverlabel={
                "bgcolor": "#14213d",
                "bordercolor": "#203f72",
                "font": {"family": "Arial, sans-serif", "size": 13, "color": "#ffffff"},
            },
        )
        fig.update_xaxes(showgrid=False, tickformat="%b %Y", dtick="M1", ticklabelmode="period")
        fig.update_yaxes(gridcolor="#e9edf5", rangemode="tozero")
    fig.update_layout(
        title_x=0.02,
        margin={"l": 45, "r": 25, "t": 70, "b": 45},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033"},
    )

    products_filtered = (
        filtered_sales.groupby("Product", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Quantity_Sold=("Quantity", "sum"))
    )
    products_filtered["Average_Selling_Price"] = (
        products_filtered["Revenue"]
        / products_filtered["Quantity_Sold"].where(products_filtered["Quantity_Sold"] != 0)
    )

    product_metric_config = {
        "Revenue": {"column": "Revenue", "axis_label": "Revenue (£)", "title": "Revenue"},
        "Units Sold": {"column": "Quantity_Sold", "axis_label": "Units sold", "title": "Units Sold"},
        "Average Selling Price": {
            "column": "Average_Selling_Price",
            "axis_label": "Average selling price (£)",
            "title": "Average Selling Price",
        },
    }
    product_config = product_metric_config.get(selected_product_metric, product_metric_config["Revenue"])
    product_column = product_config["column"]
    product_axis_label = product_config["axis_label"]
    product_title = product_config["title"]

    valid_products = products_filtered.dropna(subset=[product_column])
    products_filtered = (
        valid_products.sort_values(product_column, ascending=False)
        .head(10)
        .sort_values(product_column, ascending=True)
    )

    if products_filtered.empty or products_filtered[product_column].sum() <= 0:
        product_fig = px.bar(title=f"No product data available — {selected_country}", template=chart_template)
        product_fig.add_annotation(
            text="Try selecting a different country or date range.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#64748b"},
        )
        product_fig.update_xaxes(visible=False)
        product_fig.update_yaxes(visible=False)
    else:
        product_fig = px.bar(
            products_filtered,
            x=product_column,
            y="Product",
            orientation="h",
            title=f"Top 10 Products by {product_title} — {selected_country}",
            labels={
                product_column: product_axis_label,
                "Product": "Product",
                "Revenue": "Revenue (£)",
                "Quantity_Sold": "Quantity sold",
                "Average_Selling_Price": "Average selling price (£)",
            },
            template=chart_template,
            hover_data={"Revenue": ":.2f", "Quantity_Sold": True},
        )
        product_fig.update_xaxes(gridcolor="#e9edf5", rangemode="tozero")
    product_fig.update_layout(
        title_x=0.02,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033"},
        yaxis={"categoryorder": "total ascending"},
    )

    country_sales = sales
    if start_date:
        country_sales = country_sales.loc[
            country_sales["InvoiceDate"] >= pd.Timestamp(start_date)
        ]
    if end_date:
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        country_sales = country_sales.loc[
            country_sales["InvoiceDate"] < end_exclusive
        ]

    country_performance = (
        country_sales.groupby("Country", as_index=False)
        .agg(Revenue=("Revenue", "sum"))
        .sort_values("Revenue", ascending=False)
        .head(10)
        .sort_values("Revenue", ascending=True)
    )
    if country_performance.empty or country_performance["Revenue"].sum() <= 0:
        country_fig = px.bar(title="No country revenue available for this date range", template=chart_template)
        country_fig.add_annotation(
            text="Try selecting a different date range.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#64748b"},
        )
        country_fig.update_xaxes(visible=False)
        country_fig.update_yaxes(visible=False)
    else:
        country_fig = px.bar(
            country_performance,
            x="Revenue",
            y="Country",
            orientation="h",
            title="Top 10 Countries by Revenue",
            labels={"Revenue": "Revenue (£)", "Country": "Country"},
            template=chart_template,
        )
    country_fig.update_layout(
        title_x=0.02,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033"},
        yaxis={"categoryorder": "total ascending"},
    )
    country_fig.update_xaxes(gridcolor="#e9edf5", rangemode="tozero")

    country_contribution = (
        country_sales.groupby("Country", as_index=False)
        .agg(Revenue=("Revenue", "sum"))
    )
    total_country_revenue = country_contribution["Revenue"].sum()
    if not country_contribution.empty and total_country_revenue > 0:
        country_contribution["Revenue Share (%)"] = (
            country_contribution["Revenue"] / total_country_revenue * 100
        )
        country_contribution = country_contribution.sort_values(
            "Revenue Share (%)", ascending=False
        ).head(10)
        contribution_fig = px.pie(
            country_contribution,
            names="Country",
            values="Revenue Share (%)",
            title="Revenue Contribution by Country (Top 10)",
            template=chart_template,
            hole=0.45,
        )
        contribution_fig.update_traces(
            textposition="outside",
            textinfo="none",
            customdata=country_contribution[["Revenue"]].to_numpy(),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Revenue share: %{percent}<br>"
                "Revenue: £%{customdata[0]:,.2f}"
                "<extra></extra>"
            ),
        )
    else:
        contribution_fig = px.pie(
            title="No revenue contribution available for this date range",
            template=chart_template,
            hole=0.45,
        )
        contribution_fig.add_annotation(
            text="Try selecting a different date range.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#64748b"},
        )
        contribution_fig.update_xaxes(visible=False)
        contribution_fig.update_yaxes(visible=False)

    contribution_fig.update_layout(
        title_x=0.02,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033"},
        showlegend=True,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )

    fig = apply_chart_theme(fig)
    product_fig = apply_chart_theme(product_fig)
    country_fig = apply_chart_theme(country_fig)
    contribution_fig = apply_chart_theme(contribution_fig)

    return (
        fig,
        f"£{revenue:,.2f}",
        f"{orders:,}",
        f"£{average_value:,.2f}",
        monthly_growth,
        product_fig,
        country_fig,
        contribution_fig,
        customer_orders_fig,
        f"{repeat_customers:,}",
        f"{repeat_customer_rate:.1f}%",
        f"{units_sold:,.0f}",
        f"{unique_customers:,}",
        top_country,
    )


@app.callback(
    Output("top-product-value", "children"),
    Output("best-month-value", "children"),
    Output("top-country-revenue", "children"),
    Output("top-product-revenue", "children"),
    Output("best-month-revenue", "children"),
    Output("top-customer-value", "children"),
    Output("top-customer-revenue", "children"),
    Output("country-concentration-value", "children"),
    Output("customer-contribution-value", "children"),
    Output("product-concentration-value", "children"),
    Output("business-insight-summary", "children"),
    Input("view-selector", "value"),
    Input("product-metric-selector", "value"),
    Input("country-selector", "value"),
    Input("start-date", "date"),
    Input("end-date", "date"),
)
def update_summary_cards(
    selected_metric,
    selected_product_metric,
    selected_country,
    start_date,
    end_date,
):
    if selected_country == "All":
        filtered_sales = sales
    else:
        filtered_sales = sales.loc[sales["Country"] == selected_country]

    if start_date:
        filtered_sales = filtered_sales.loc[
            filtered_sales["InvoiceDate"] >= pd.Timestamp(start_date)
        ]
    if end_date:
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        filtered_sales = filtered_sales.loc[
            filtered_sales["InvoiceDate"] < end_exclusive
        ]

    country_revenue = filtered_sales.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    product_revenue = filtered_sales.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
    customer_revenue = (
        filtered_sales.dropna(subset=["CustomerID"])
        .groupby("CustomerID")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    monthly_revenue = (
        filtered_sales.groupby(filtered_sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp())["Revenue"]
        .sum()
        .sort_index()
    )

    total_revenue = filtered_sales["Revenue"].sum()
    top_country = country_revenue.index[0] if not country_revenue.empty else "N/A"
    top_product = product_revenue.index[0] if not product_revenue.empty else "N/A"
    best_month = monthly_revenue.idxmax().strftime("%B %Y") if not monthly_revenue.empty else "N/A"
    top_customer = int(customer_revenue.index[0]) if not customer_revenue.empty else "N/A"
    top_customer_revenue = customer_revenue.iloc[0] if not customer_revenue.empty else 0
    top_country_revenue = country_revenue.iloc[0] if not country_revenue.empty else 0
    top_product_revenue = product_revenue.iloc[0] if not product_revenue.empty else 0

    country_share = (top_country_revenue / total_revenue * 100) if total_revenue else 0
    customer_share = (top_customer_revenue / total_revenue * 100) if total_revenue else 0
    product_share = (top_product_revenue / total_revenue * 100) if total_revenue else 0

    customer_counts = (
        filtered_sales.dropna(subset=["CustomerID"])
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
    )
    repeat_customers = int((customer_counts > 1).sum()) if not customer_counts.empty else 0
    identified_customers = int(customer_counts.shape[0]) if not customer_counts.empty else 0
    repeat_rate = (
        repeat_customers / identified_customers * 100 if identified_customers else 0
    )

    monthly_growth = "N/A"
    if not monthly_revenue.empty:
        monthly_calendar = monthly_revenue.asfreq("MS", fill_value=0)
        if len(monthly_calendar) >= 2:
            previous_revenue = monthly_calendar.iloc[-2]
            current_revenue = monthly_calendar.iloc[-1]
            if previous_revenue > 0:
                growth_percent = ((current_revenue - previous_revenue) / previous_revenue) * 100
                monthly_growth = f"{growth_percent:+.2f}%"
            elif current_revenue > 0:
                monthly_growth = "New sales"
            else:
                monthly_growth = "0.00%"

    narrative = [
        html.P(
            f"{top_country} accounts for {country_share:.1f}% of current revenue, contributing £{top_country_revenue:,.2f} to the filtered view.",
            style={"margin": "0 0 8px 0"},
        )
    ]
    if top_product != "N/A":
        narrative.append(
            html.P(
                f"{top_product} is the leading product by revenue, contributing {product_share:.1f}% of gross sales (£{top_product_revenue:,.2f}).",
                style={"margin": "0 0 8px 0"},
            )
        )
    if identified_customers:
        narrative.append(
            html.P(
                f"Repeat-customer activity is {repeat_rate:.1f}% ({repeat_customers} of {identified_customers} identified customers), indicating {repeat_rate:.1f}% of customers purchased more than once in the current selection.",
                style={"margin": "0 0 8px 0"},
            )
        )
    narrative.append(
        html.P(
            f"Revenue momentum: the latest month is {monthly_growth} versus the prior month, with the strongest month at {best_month}.",
            style={"margin": "0"},
        )
    )

    return (
        top_product,
        best_month,
        f"Revenue: £{top_country_revenue:,.2f}",
        f"Revenue: £{top_product_revenue:,.2f}",
        f"Revenue: £{monthly_revenue.max() if not monthly_revenue.empty else 0:,.2f}",
        f"Customer {top_customer}" if top_customer != "N/A" else "N/A",
        f"Revenue: £{top_customer_revenue:,.2f}" if top_customer != "N/A" else "Revenue: £0.00",
        f"{country_share:.1f}% of gross sales",
        f"{customer_share:.1f}% of gross sales" if top_customer != "N/A" else "No identifiable customer data",
        f"{product_share:.1f}% of gross sales",
        narrative,
    )


@app.callback(
    Output("country-selector", "value"),
    Output("start-date", "date"),
    Output("end-date", "date"),
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(n_clicks):
    if not n_clicks:
        return "All", None, None

    return (
        "All",
        sales["InvoiceDate"].min().date(),
        sales["InvoiceDate"].max().date(),
    )



@app.callback(
    Output("download-filtered-csv", "data"),
    Input("export-csv-button", "n_clicks"),
    State("country-selector", "value"),
    State("start-date", "date"),
    State("end-date", "date"),
    prevent_initial_call=True,
)
def export_filtered_csv(n_clicks, selected_country, start_date, end_date):
    if not n_clicks:
        return None

    filtered_sales = sales.copy()

    if selected_country != "All":
        filtered_sales = filtered_sales[
            filtered_sales["Country"] == selected_country
        ]

    if start_date:
        filtered_sales = filtered_sales[
            filtered_sales["InvoiceDate"] >= pd.Timestamp(start_date)
        ]

    if end_date:
        filtered_sales = filtered_sales[
            filtered_sales["InvoiceDate"]
            < pd.Timestamp(end_date) + pd.Timedelta(days=1)
        ]

    export_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
        "Revenue",
        "Country",
    ]

    export_columns = [
        column for column in export_columns
        if column in filtered_sales.columns
    ]

    return dcc.send_data_frame(
        filtered_sales[export_columns].to_csv,
        "retailpulse_filtered_sales.csv",
        index=False,
        encoding="utf-8-sig",
    )



@app.callback(
    Output("cross-sell-chart", "figure"),
    Output("cross-sell-summary", "children"),
    Input("country-selector", "value"),
    Input("start-date", "date"),
    Input("end-date", "date"),
)
def update_cross_sell(selected_country, start_date, end_date):
    """Calculate basket co-purchase patterns for the current filter selection."""
    mask = pd.Series(True, index=sales.index)

    if selected_country != "All":
        mask &= sales["Country"].eq(selected_country)

    if start_date:
        mask &= sales["InvoiceDate"] >= pd.Timestamp(start_date)

    if end_date:
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        mask &= sales["InvoiceDate"] < end_exclusive

    filtered_sales = sales.loc[mask].copy()
    pair_data = build_product_associations(filtered_sales, max_pairs=8)

    if pair_data.empty:
        fig = px.bar(
            title="No co-purchase patterns available for this selection",
            template=chart_template,
        )
        fig.add_annotation(
            text="Try selecting a different country or date range.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#64748b"},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        summary_content = html.P(
            "No repeated product pairs were observed in the current filtered view.",
            style={"margin": "0"},
        )
        return fig, summary_content

    top_pair = summarize_top_product_pair(filtered_sales, max_pairs=8)
    pair_chart = pair_data.head(8).sort_values("CoPurchaseCount", ascending=True)
    fig = px.bar(
        pair_chart,
        x="CoPurchaseCount",
        y="ProductPair",
        orientation="h",
        title="Most Common Product Pairs",
        labels={
            "ProductPair": "Product pair",
            "CoPurchaseCount": "Co-purchase count",
        },
        template=chart_template,
        color="CoPurchaseCount",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        title_x=0.02,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033"},
        yaxis={"categoryorder": "total ascending"},
    )
    fig.update_xaxes(gridcolor="#e9edf5", rangemode="tozero")

    summary_content = [
        html.P(
            f"Strongest pair: {top_pair['pair_name']} ({top_pair['count']} co-purchases, {top_pair['support']:.1f}% support)",
            style={"margin": "0 0 8px 0"},
        ),
        html.P(
            "This view tracks recurring item bundles purchased within the same invoice, which can indicate natural cross-sell opportunities.",
            style={"margin": "0"},
        ),
    ]
    return fig, summary_content


@app.callback(
    Output("customer-insights-chart", "figure"),
    Output("customer-insights-summary", "children"),
    Input("country-selector", "value"),
    Input("start-date", "date"),
    Input("end-date", "date"),
)
def update_customer_insights(selected_country, start_date, end_date):
    """Show top customer accounts for the current filtered data slice."""
    mask = pd.Series(True, index=sales.index)

    if selected_country != "All":
        mask &= sales["Country"].eq(selected_country)

    if start_date:
        mask &= sales["InvoiceDate"] >= pd.Timestamp(start_date)

    if end_date:
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        mask &= sales["InvoiceDate"] < end_exclusive

    filtered_sales = sales.loc[mask].copy()
    customer_summary = build_customer_value_summary(filtered_sales, top_n=10)

    if customer_summary.empty:
        fig = px.bar(
            title="No customer revenue data available for this selection",
            template=chart_template,
        )
        fig.add_annotation(
            text="Try selecting a different country or date range.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#64748b"},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        summary_content = html.P(
            "No identifiable customer data is available in the current filtered view.",
            style={"margin": "0"},
        )
        return fig, summary_content

    summary = summarize_top_customers(filtered_sales, top_n=5)
    chart_data = customer_summary.head(10).sort_values("TotalRevenue", ascending=True)
    fig = px.bar(
        chart_data,
        x="TotalRevenue",
        y="CustomerID",
        orientation="h",
        title="Top Customers by Revenue",
        labels={
            "CustomerID": "Customer ID",
            "TotalRevenue": "Revenue (£)",
        },
        template=chart_template,
        text="TotalRevenue",
        color="TotalRevenue",
        color_continuous_scale="Teal",
    )
    fig.update_traces(
        texttemplate="£%{text:,.0f}",
        textposition="outside",
        hovertemplate=(
            "<b>Customer %{y}</b><br>"
            "Revenue: £%{x:,.2f}<br>"
            "Invoices: %{customdata[0]:,.0f}<br>"
            "Average order value: £%{customdata[1]:,.2f}"
            "<extra></extra>"
        ),
        customdata=chart_data[["InvoiceCount", "AverageOrderValue"]].to_numpy(),
    )
    fig.update_layout(
        title_x=0.02,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033"},
        yaxis={"categoryorder": "total ascending"},
    )
    fig.update_xaxes(gridcolor="#e9edf5", rangemode="tozero")

    summary_content = [
        html.P(
            f"Top customer: Customer {summary['top_customer']} with £{summary['top_revenue']:,.2f} in revenue and £{summary['avg_order_value']:,.2f} average order value.",
            style={"margin": "0 0 8px 0"},
        ),
        html.P(
            f"The top {summary['customer_count']} customer accounts shown here account for the strongest value concentration in the current selection.",
            style={"margin": "0"},
        ),
    ]
    return fig, summary_content


@app.callback(
    Output("revenue-forecast-chart", "figure"),
    Output("forecast-model-summary", "children"),
    Input("country-selector", "value"),
    Input("start-date", "date"),
    Input("end-date", "date"),
)
def update_revenue_forecast(selected_country, start_date, end_date):
    """Update the experimental forecast for the selected data filters."""



    # Build filters without modifying the shared sales DataFrame.
    mask = pd.Series(True, index=sales.index)

    if selected_country != "All":
        mask &= sales["Country"].eq(selected_country)

    if start_date:
        mask &= sales["InvoiceDate"] >= pd.Timestamp(start_date)

    if end_date:
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        mask &= sales["InvoiceDate"] < end_exclusive

    # Keep only the two columns required for forecasting.
    forecast_data = sales.loc[mask, ["InvoiceDate", "Revenue"]]

    # Aggregate revenue by month without adding a column to sales.
    monthly_revenue = (
        forecast_data.groupby(
            forecast_data["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
        )["Revenue"]
        .sum()
        .sort_index()
    )


    # Require sufficient, consecutive monthly history.
    if len(monthly_revenue) < 6:
        fig = px.line(
            title="Insufficient monthly history for forecasting",
            template=chart_template,
        )
        fig.add_annotation(
            text=(
                "Select a wider date range or another country "
                "to include at least six consecutive months."
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 13, "color": "#64748b"},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return apply_chart_theme(fig), "Forecast unavailable for this selection."

    # Add missing calendar months as zero-revenue months.
    monthly_revenue = monthly_revenue.asfreq("MS", fill_value=0)

    try:
        forecast, evaluation = create_revenue_forecast(
            monthly_revenue,
            horizon=3,
        )
    except (TypeError, ValueError) as error:
        fig = px.line(
            title="Forecast unavailable for this selection",
            template=chart_template,
        )
        fig.add_annotation(
            text=str(error),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 13, "color": "#64748b"},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return apply_chart_theme(fig), "The selected data could not be forecast."

    history = monthly_revenue.rename("Revenue").reset_index()
    history.columns = ["Month", "Revenue"]

    fig = px.line(
        history,
        x="Month",
        y="Revenue",
        markers=True,
        title="Historical and Forecast Revenue",
        labels={
            "Month": "Month",
            "Revenue": "Revenue (£)",
        },
        template=chart_template,
    )

    fig.update_traces(
        name="Historical revenue",
        showlegend=True,
        hovertemplate=(
            "%{x|%b %Y}<br>Revenue: £%{y:,.2f}<extra></extra>"
        ),
    )

    fig.add_scatter(
        x=forecast["Month"],
        y=forecast["Forecast Revenue"],
        mode="lines+markers",
        name="Forecast revenue",
        line={"dash": "dash", "width": 3},
        hovertemplate=(
            "%{x|%b %Y}<br>Forecast: £%{y:,.2f}<extra></extra>"
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        tickformat="%b %Y",
    )
    fig.update_yaxes(
        title="Revenue (£)",
        gridcolor="#e9edf5",
        rangemode="tozero",
    )

    fig = apply_chart_theme(fig)

    selected_model = evaluation.iloc[0]["Model"]
    selected_mae = evaluation.iloc[0]["MAE"]

    summary = (
        f"Selected model: {selected_model} | "
        f"Backtesting MAE: £{selected_mae:,.2f} | "
        "Experimental forecast — limited historical data."
    )

    return fig, summary




if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
