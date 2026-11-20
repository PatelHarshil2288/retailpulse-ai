import pandas as pd

from src.ai_insights import generate_deterministic_ai_insights
from src.data_profiling import profile_dataset


def test_generates_evidence_based_findings_and_recommendations():
    data = pd.DataFrame(
        {
            "Group": ["A", "A", "B", "B", "B", None],
            "Amount": [10, 12, 20, 22, 300, None],
            "EventDate": [
                "2024-01-01",
                "2024-01-01",
                "2024-02-01",
                "2024-02-15",
                "2024-02-20",
                None,
            ],
        }
    )
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    profile = profile_dataset(data)

    report = generate_deterministic_ai_insights(profile, "sample.csv")
    findings = " ".join(report["findings"])
    recommendations = " ".join(report["recommendations"])
    limitations = " ".join(report["limitations"])

    assert "sample.csv" in findings
    assert "quality issue" in findings
    assert "duplicate row" in findings
    assert "missing cell" in findings
    assert "date coverage spans" in findings
    assert "missing-value treatment" in recommendations
    assert "trend views" in recommendations
    assert "do not infer causation" in limitations


def test_marks_modeling_limits_without_numeric_or_date_columns():
    data = pd.DataFrame({"Category": ["A", "B", "C"]})
    profile = profile_dataset(data)

    report = generate_deterministic_ai_insights(profile, "labels.csv")
    limitations = " ".join(report["limitations"])

    assert "sample size is limited" in limitations
    assert "No numeric columns were detected" in limitations
    assert "No usable date column was detected" in limitations
