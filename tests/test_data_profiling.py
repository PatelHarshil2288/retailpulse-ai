import pandas as pd

from src.data_profiling import profile_dataset


def test_profile_reports_schema_missing_values_outliers_and_dates():
    data = pd.DataFrame(
        {
            "Amount": [10, 11, 12, 13, 100, None],
            "Segment": ["A", "B", "A", None, "B", "A"],
            "EventDate": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "unknown",
            ],
        }
    )
    original = data.copy(deep=True)

    profile = profile_dataset(data)

    assert profile["summary"]["dimensions"] == "6 rows x 3 columns"
    assert profile["summary"]["missing_cells"] == 2
    assert profile["summary"]["rows_with_missing"] == 2
    assert profile["summary"]["date_coverage"]["earliest"].startswith(
        "2024-01-01"
    )
    columns = {column["name"]: column for column in profile["columns"]}
    assert columns["Amount"]["detected_type"] == "numeric"
    assert columns["Amount"]["outlier_count"] == 1
    assert columns["Segment"]["detected_type"] == "categorical"
    assert columns["EventDate"]["detected_type"] == "date"
    assert columns["EventDate"]["invalid_date_count"] == 1
    assert any(issue["type"] == "missing_values" for issue in profile["issues"])
    assert any(
        issue["type"] == "potential_outliers" for issue in profile["issues"]
    )
    pd.testing.assert_frame_equal(data, original)


def test_profile_reports_duplicate_constant_and_high_cardinality_findings():
    data = pd.DataFrame(
        {
            "Identifier": [f"record-{index}" for index in range(60)],
            "Constant": ["same"] * 60,
            "Value": list(range(59)) + [58],
        }
    )
    data.loc[59] = data.loc[58]

    profile = profile_dataset(data)
    issue_types = {issue["type"] for issue in profile["issues"]}

    assert profile["summary"]["duplicate_rows"] == 1
    assert {"duplicate_rows", "constant_column", "high_cardinality"} <= issue_types
    assert all(
        issue["automatic_correction_safe"] is False
        for issue in profile["issues"]
    )
    assert any(
        "not been changed" in recommendation
        for recommendation in profile["recommendations"]
    )


def test_profile_reports_invalid_values_in_numeric_like_column():
    data = pd.DataFrame(
        {"Amount": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "bad"]}
    )

    profile = profile_dataset(data)
    column = profile["columns"][0]

    assert column["detected_type"] == "numeric"
    assert column["invalid_numeric_count"] == 1
    assert any(
        issue["type"] == "invalid_numeric_values"
        for issue in profile["issues"]
    )
