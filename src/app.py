
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# RetailPulse AI — Interactive Retail Analytics
# Initial dashboard using clearly labelled demo data
# --------------------------------------------------

app = Dash(__name__)
app.title = "RetailPulse AI"

# Demonstration dataset. We will replace this with
# the public UCI Online Retail dataset in the next phase.
sales = pd.DataFrame({
    "Month": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ],
    "Revenue": [
        42000, 46500, 51000, 48000, 57000, 61500,
        59000, 67000, 72000, 69500, 83000, 95000
    ],
    "Orders": [
        820, 875, 940, 900, 1020, 1100,
        1060, 1180, 1250, 1210, 1400, 1580
    ],
})

sales["Average Order Value"] = (
    sales["Revenue"] / sales["Orders"]
)

monthly_revenue = sales["Revenue"].sum()
total_orders = sales["Orders"].sum()
average_order_value = sales["Revenue"].sum() / sales["Orders"].sum()

# Consistent visual theme for Plotly charts
chart_template = "plotly_white"

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
                html.H1(
                    "RetailPulse AI 📊",
                    style={"margin": "0 0 8px 0"},
                ),
                html.P(
                    "Interactive Retail Analytics & Customer Intelligence",
                    style={"margin": "0", "fontSize": "17px"},
                ),
                html.P(
                    "Prototype dashboard • Demonstration data, not actual sales",
                    style={"margin": "12px 0 0 0", "fontSize": "13px"},
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
                        html.P("Annual Revenue", style={"margin": "0"}),
                        html.H2(
                            f"${monthly_revenue:,.0f}",
                            style={"margin": "12px 0 0 0"},
                        ),
                    ],
                    style={
                        "background": "white",
                        "padding": "22px",
                        "borderRadius": "14px",
                        "boxShadow": "0 3px 12px #15264b12",
                    },
                ),
                html.Div(
                    [
                        html.P("Total Orders", style={"margin": "0"}),
                        html.H2(
                            f"{total_orders:,}",
                            style={"margin": "12px 0 0 0"},
                        ),
                    ],
                    style={
                        "background": "white",
                        "padding": "22px",
                        "borderRadius": "14px",
                        "boxShadow": "0 3px 12px #15264b12",
                    },
                ),
                html.Div(
                    [
                        html.P("Average Order Value", style={"margin": "0"}),
                        html.H2(
                            f"${average_order_value:,.2f}",
                            style={"margin": "12px 0 0 0"},
                        ),
                    ],
                    style={
                        "background": "white",
                        "padding": "22px",
                        "borderRadius": "14px",
                        "boxShadow": "0 3px 12px #15264b12",
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
                    "Select the revenue view:",
                    htmlFor="view-selector",
                    style={"fontWeight": "bold"},
                ),
                dcc.RadioItems(
                    id="view-selector",
                    options=[
                        {"label": " Revenue", "value": "Revenue"},
                        {"label": " Orders", "value": "Orders"},
                        {
                            "label": " Average Order Value",
                            "value": "Average Order Value",
                        },
                    ],
                    value="Revenue",
                    inline=True,
                    style={"display": "flex", "gap": "18px", "flexWrap": "wrap",
                           "marginTop": "12px"},
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
                )
            ],
        ),

        html.Footer(
            "RetailPulse AI | Portfolio prototype | Built with Python, Dash and Plotly",
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
    Input("view-selector", "value"),
)
def update_chart(selected_metric):
    fig = px.bar(
        sales,
        x="Month",
        y=selected_metric,
        title=f"Monthly {selected_metric}",
        template=chart_template,
        color="Revenue",
        color_continuous_scale="Blues",
        labels={
            "Month": "Month",
            "Revenue": "Revenue ($)",
            "Orders": "Number of orders",
            "Average Order Value": "Average order value ($)",
        },
        hover_data={
            "Revenue": ":,.0f",
            "Orders": ":,",
            "Average Order Value": ":.2f",
        },
    )

    fig.update_layout(
        title_x=0.02,
        margin={"l": 35, "r": 25, "t": 70, "b": 35},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033"},
        coloraxis_showscale=False,
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e9edf5", rangemode="tozero")

    return fig


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
