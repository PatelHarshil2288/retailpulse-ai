import numpy as np
import pandas as pd
import pytest

from src.data_profiling import profile_dataset
from src.universal_analytics import (
    ROW_COUNT_MEASURE,
    aggregate_dataset,
    analysis_control_options,
    build_analysis_recommendations,
    build_correlation_figure,
    build_trend_figure,
    build_universal_figure,
    filter_dataset,
)


@pytest.fixture
def generic_data():
    return pd.DataFrame(
        {
            "Group": ["A", "A", "B", "B", "B"],
            "Amount": [10, 12, 20, 22, 100],
            "Other": [2, 4, 6, 8, 10],
            "EventDate": [
                "2024-01-02",
                "2024-01-20",
                "2024-02-02",
                "2024-02-18",
                "2024-02-25",
            ],
        }
    )


def test_universal_controls_use_detected_columns(generic_data):
    controls = analysis_control_options(profile_dataset(generic_data))

    assert controls["dimension"] == "EventDate"
    assert controls["measure"] == "Amount"
    assert "Amount" in controls["numeric_columns"]
    assert controls["date_columns"] == ["EventDate"]
    assert {option["value"] for option in controls["filter_options"]} >= {
        "Group",
        "Amount",
        "EventDate",
    }


def test_groups_generic_numeric_measures_by_category(generic_data):
    profile = profile_dataset(generic_data)
    controls = analysis_control_options(profile)

    grouped = aggregate_dataset(
        generic_data,
        "Group",
        "Amount",
        "mean",
        numeric_columns=controls["numeric_columns"],
    )

    assert grouped.to_dict("records") == [
        {"Category": "B", "Value": pytest.approx(142 / 3)},
        {"Category": "A", "Value": 11.0},
    ]


def test_generic_date_grouping_uses_months_and_supports_line_chart(generic_data):
    controls = analysis_control_options(profile_dataset(generic_data))
    grouped = aggregate_dataset(
        generic_data,
        "EventDate",
        "Amount",
        "sum",
        numeric_columns=controls["numeric_columns"],
        date_columns=controls["date_columns"],
    )
    figure = build_universal_figure(
        grouped,
        "EventDate",
        "Amount",
        "sum",
        "line",
        date_columns=controls["date_columns"],
    )

    assert grouped.to_dict("records") == [
        {"Category": "2024-01", "Value": 22},
        {"Category": "2024-02", "Value": 142},
    ]
    assert figure.data[0].type == "scatter"
    assert figure.layout.title.text == "Sum Amount by EventDate"


def test_row_count_aggregation_counts_all_records(generic_data):
    grouped = aggregate_dataset(
        generic_data,
        "Group",
        ROW_COUNT_MEASURE,
        "count",
    )

    assert grouped.to_dict("records") == [
        {"Category": "B", "Value": 3},
        {"Category": "A", "Value": 2},
    ]


def test_generic_filter_is_literal_and_case_insensitive(generic_data):
    filtered = filter_dataset(generic_data, "Group", "b")

    assert len(filtered) == 3
    assert filtered["Group"].tolist() == ["B", "B", "B"]


def test_numeric_aggregation_rejects_categorical_measure(generic_data):
    with pytest.raises(ValueError, match="requires a numeric measure"):
        aggregate_dataset(
            generic_data,
            "Group",
            "Group",
            "sum",
        )


def test_correlation_and_trend_figures_use_detected_data(generic_data):
    profile = profile_dataset(generic_data)

    correlation = build_correlation_figure(generic_data, profile)
    trend = build_trend_figure(generic_data, profile)

    assert correlation.data[0].type == "heatmap"
    assert trend.data[0].type == "scatter"
    assert "Monthly mean" in trend.layout.title.text
    recommendations = build_analysis_recommendations(profile)
    assert any("correlation" in item.lower() for item in recommendations)
    assert any("trend" in item.lower() for item in recommendations)


def test_analytics_excludes_infinite_numeric_values(generic_data):
    data = generic_data.copy()
    data["Amount"] = data["Amount"].astype(float)
    data.loc[4, "Amount"] = np.inf
    profile = profile_dataset(data)
    controls = analysis_control_options(profile)

    grouped = aggregate_dataset(
        data,
        "Group",
        "Amount",
        "sum",
        numeric_columns=controls["numeric_columns"],
    )
    correlation = build_correlation_figure(data, profile)
    trend = build_trend_figure(data, profile)

    assert grouped.loc[grouped["Category"] == "B", "Value"].item() == 42
    assert np.isfinite(np.asarray(correlation.data[0].z, dtype=float)).all()
    assert np.isfinite(np.asarray(trend.data[0].y, dtype=float)).all()
