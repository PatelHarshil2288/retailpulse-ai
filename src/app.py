
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context


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
    parse_dates=["InvoiceDate"],
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

product_revenue = (
    sales.groupby("Product")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

top_product = product_revenue.index[0] if not product_revenue.empty else "N/A"
top_product_revenue = (
    product_revenue.iloc[0] if not product_revenue.empty else 0
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


chart_template = "plotly_white"

app = Dash(__name__)
app.title = "RetailPulse AI"

card_style = {
    "background": "white",
    "padding": "22px",
    "borderRadius": "14px",
    "boxShadow": "0 3px 12px #15264b12",
}

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f3f6fb",
        "minHeight": "100vh",
        "padding": "28px",
        "color": "#172033",
    },
    children=[
        html.Div(
            style={
                "background": "linear-gradient(120deg, #14213d, #2457a7)",
                "padding": "28px",
                "borderRadius": "18px",
                "color": "white",
                "marginBottom": "24px",
            },
            children=[
                html.H1("RetailPulse AI - Retail Analytics"),
                html.P(
                    "Interactive Retail Analytics & Customer Intelligence",
                    style={"fontSize": "17px"},
                ),
                html.P(
                    "Powered by the UCI Online Retail dataset | "
                    "December 2010 - December 2011 | Currency: GBP (£)",
                    style={"fontSize": "13px"},
                ),
                html.P(
                    "Revenue represents gross sales from positive-quantity, "
                    "positive-price transactions; returns are not deducted.",
                    style={"fontSize": "12px"},
                ),
            ],
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(190px, 1fr))",
                "gap": "16px",
                "marginBottom": "24px",
            },
            children=[
                html.Div(
                    [
                        html.P("Gross Sales Revenue"),
                        html.H2(
                                 f"£{total_revenue:,.2f}",
                                 id="gross-revenue-value",
                                ),
                    ],
                    style=card_style,
                ),
                html.Div(
                    [
                        html.P("Unique Orders"),
                        html.H2(f"{total_orders:,}",
                        id="unique-orders-value",
                        ),
                    ],
                    style=card_style,
                ),
                html.Div(
                    [
                        html.P("Average Order Value"),
                        html.H2(f"£{average_order_value:,.2f}",
                                id="average-order-value",
                            ),
                    ],
                    style=card_style,
                ),
                
                html.Div(
                    [
                        html.P(
                            "Monthly Revenue Growth",
                            style={"fontSize": "14px", "color": "#64748b"},
                        ),
                        html.H3(
                            id="monthly-growth-value",
                            children="N/A",
                            style={"margin": "0", "color": "#172033"},
                        ),
                        html.P(
                            "Compared with the previous month",
                            style={"fontSize": "12px", "color": "#64748b"},
                        ),
                    ],
                    style={
                        "backgroundColor": "white",
                        "padding": "20px",
                        "borderRadius": "12px",
                        "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                    },
                ),

            ],
        ),

        
        html.Div(
            style={
                "background": "white",
                "padding": "22px",
                "borderRadius": "14px",
                "marginBottom": "24px",
            },
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
            style={
                "background": "white",
                "padding": "22px",
                "borderRadius": "14px",
                "marginBottom": "24px",
            },
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
            style={
                "background": "white",
                "padding": "22px",
                "borderRadius": "14px",
                "marginBottom": "24px",
            },
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
            style={
                "background": "white",
                "padding": "22px",
                "borderRadius": "14px",
                "marginBottom": "24px",
            },
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
            style={
                "background": "white",
                "padding": "18px",
                "borderRadius": "14px",
            },
            children=[
                dcc.Graph(
                    id="monthly-chart",
                    config={"displayModeBar": True, "responsive": True},
                ),
                html.H3(
                      "Top 10 Products by Revenue",
                       style={"marginTop": "30px"},
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
                ),
                html.H3(
                    "Top 10 Countries by Revenue",
                    style={"marginTop": "30px"},
                ),
                dcc.Graph(
                    id="country-performance-chart",
                ),
                
                html.H3(
                    "Revenue Contribution by Country",
                    style={"marginTop": "30px"},
                ),
                dcc.Graph(
                    id="country-contribution-chart",
                ),


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
    Output("top-country-value", "children"),
    Output("top-product-value", "children"),
    Output("best-month-value", "children"),
    Output("top-country-revenue", "children"),
    Output("top-product-revenue", "children"),
    Output("best-month-revenue", "children"),
    Input("view-selector", "value"),
    Input("country-selector", "value"),
    Input("start-date", "date"),
    Input("end-date", "date"),
)
def update_chart(
    selected_metric,
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

    # Apply the date range, including the entire end date.
    if start_date:
        filtered_sales = filtered_sales.loc[
            filtered_sales["InvoiceDate"] >= pd.Timestamp(start_date)
        ]

    if end_date:
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        filtered_sales = filtered_sales.loc[
            filtered_sales["InvoiceDate"] < end_exclusive
        ]

    # Calculate KPIs for the selected filters.
    revenue = filtered_sales["Revenue"].sum()
    orders = filtered_sales["InvoiceNo"].nunique()
    average_value = revenue / orders if orders else 0

    
    # Calculate Business Insights for the selected filters.
    country_revenue = (
        filtered_sales.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    top_country = (
        country_revenue.index[0]
        if not country_revenue.empty
        else "N/A"
    )
    top_country_revenue = (
        country_revenue.iloc[0]
        if not country_revenue.empty
        else 0
    )

    product_revenue = (
        filtered_sales.groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    top_product = (
        product_revenue.index[0]
        if not product_revenue.empty
        else "N/A"
    )

    top_product_revenue = (
        product_revenue.iloc[0]
        if not product_revenue.empty
        else 0
    )

    monthly_revenue = (
        filtered_sales.assign(
            Month=filtered_sales["InvoiceDate"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        .groupby("Month")["Revenue"]
        .sum()
        .sort_index()
    )

    best_month = (
        monthly_revenue.idxmax().strftime("%B %Y")
        if not monthly_revenue.empty
        else "N/A"
    )

    best_month_revenue = (
        monthly_revenue.max()
        if not monthly_revenue.empty
        else 0
    )


    # Aggregate monthly results.
    monthly_filtered = (
        filtered_sales.assign(
            Month=filtered_sales["InvoiceDate"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        .groupby("Month", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .sort_values("Month")
    )

    monthly_filtered["Average Order Value"] = monthly_filtered["Revenue"].div(monthly_filtered["Orders"])


    # Calculate growth between consecutive calendar months.
    monthly_growth = "N/A"

    if not monthly_filtered.empty:
        monthly_calendar = monthly_filtered.set_index("Month")["Revenue"]

        # Include missing calendar months with zero revenue.
        monthly_calendar = monthly_calendar.asfreq("MS", fill_value=0)

        if len(monthly_calendar) >= 2:
            previous_revenue = monthly_calendar.iloc[-2]
            current_revenue = monthly_calendar.iloc[-1]

            if previous_revenue > 0:
                growth_percent = (
                    (current_revenue - previous_revenue)
                    / previous_revenue
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

    # Monthly chart.
    if monthly_filtered.empty:
        fig = px.bar(
            title=f"No monthly data available — {selected_country}",
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
    else:
        fig = px.bar(
            monthly_filtered,
            x="Month",
            y=selected_metric,
            title=f"Monthly {selected_metric} — {selected_country}",
            template=chart_template,
            labels={
                "Month": "Month",
                selected_metric: labels[selected_metric],
            },
            hover_data={
                "Month": "|%B %Y",
                "Revenue": ":,.2f",
                "Orders": ":,",
                "Average Order Value": ":.2f",
            },
        )

        fig.update_xaxes(
            showgrid=False,
            tickformat="%b %Y",
            dtick="M1",
            ticklabelmode="period",
        )
        fig.update_yaxes(
            gridcolor="#e9edf5",
            rangemode="tozero",
        )

    fig.update_layout(
        title_x=0.02,
        margin={"l": 45, "r": 25, "t": 70, "b": 45},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033"},
    )

    
    # Prepare top 10 products for the selected country and date range.
    products_filtered = (
        filtered_sales.groupby("Product", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity_Sold=("Quantity", "sum"),
        )
        .sort_values("Revenue", ascending=False)
        .head(10)
        .sort_values("Revenue", ascending=True)
    )

   
    # Top 10 products chart with an empty state.
    if products_filtered.empty or products_filtered["Revenue"].sum() <= 0:
        product_fig = px.bar(
            title=f"No product revenue available — {selected_country}",
            template=chart_template,
        )
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
            x="Revenue",
            y="Product",
            orientation="h",
            title=f"Top 10 Products by Revenue — {selected_country}",
            labels={
                "Revenue": "Revenue (£)",
                "Product": "Product",
                "Quantity_Sold": "Quantity sold",
            },
            template=chart_template,
            hover_data={"Quantity_Sold": True},
        )
        product_fig.update_xaxes(
            gridcolor="#e9edf5",
            rangemode="tozero",
        )

    product_fig.update_layout(
        title_x=0.02,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033"},
        yaxis={"categoryorder": "total ascending"},
    )

    
    # Compare countries using the selected date range.
    # Ignore the country selector here so all countries remain comparable.
    country_sales = sales.copy()

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
        country_fig = px.bar(
            title="No country revenue available for this date range",
            template=chart_template,
        )
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
            labels={
                "Revenue": "Revenue (£)",
                "Country": "Country",
            },
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

    country_fig.update_xaxes(
        gridcolor="#e9edf5",
        rangemode="tozero",
    )
    
    # Calculate each country's share of total revenue
    # for the selected date range.
    
    country_contribution = (
        country_sales.groupby("Country", as_index=False)
        .agg(Revenue=("Revenue", "sum"))
    )

    total_country_revenue = country_contribution["Revenue"].sum()

    if not country_contribution.empty and total_country_revenue > 0:
        country_contribution["Revenue Share (%)"] = (
            country_contribution["Revenue"]
            / total_country_revenue
            * 100
        )

        country_contribution = (
            country_contribution.sort_values(
                "Revenue Share (%)",
                ascending=False,
            )
            .head(10)
        )

        contribution_fig = px.pie(
            country_contribution,
            names="Country",
            values="Revenue Share (%)",
            title="Revenue Contribution by Country (Top 10)",
            template=chart_template,
            hole=0.45,
        )

        contribution_fig.update_traces(
            textposition="inside",
            textinfo="percent",
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
    )



    
    
    return (
        fig,
        f"£{revenue:,.2f}",
        f"{orders:,}",
        f"£{average_value:,.2f}",
        monthly_growth,
        product_fig,
        country_fig,
        contribution_fig,
        top_country,
        top_product,
        best_month,
        f"Revenue: £{top_country_revenue:,.2f}",
        f"Revenue: £{top_product_revenue:,.2f}",
        f"Revenue: £{best_month_revenue:,.2f}",
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




if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
