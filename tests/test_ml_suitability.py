import pandas as pd

from src.data_profiling import profile_dataset
from src import ml_suitability
from src.ml_suitability import assess_ml_suitability


def test_ml_suitability_detects_supervised_candidates_and_workflows():
    data = pd.DataFrame(
        {
            "FeatureA": [float(index % 17) for index in range(300)],
            "FeatureB": [float(index * 0.5) for index in range(300)],
            "Segment": [
                "A" if index % 3 else "B" for index in range(300)
            ],
            "TargetValue": [index * 1.2 + 5 for index in range(300)],
            "EventDate": pd.date_range(
                "2024-01-01", periods=300, freq="D"
            ).astype("string"),
        }
    )
    profile = profile_dataset(data)

    report = assess_ml_suitability(data, profile)
    checks = {item["name"]: item for item in report["checks"]}
    workflows = " ".join(report["workflows"])

    assert report["requires_target_confirmation"] is True
    assert "TargetValue" in report["regression_targets"]
    assert "Segment" in report["classification_targets"]
    assert checks["Dataset size"]["status"] in {"pass", "review"}
    assert "Regression baseline" in workflows
    assert "Classification baseline" in workflows


def test_ml_suitability_blocks_tiny_dataset():
    data = pd.DataFrame({"Category": ["A", "B", "C"], "Value": [1, 2, 3]})
    profile = profile_dataset(data)

    report = assess_ml_suitability(data, profile)
    checks = {item["name"]: item for item in report["checks"]}
    workflows = " ".join(report["workflows"])

    assert checks["Dataset size"]["status"] == "block"
    assert report["requires_target_confirmation"] is False
    assert "No workflow is ready for automatic training yet" in workflows


def test_ml_suitability_flags_datasets_over_model_row_limit(monkeypatch):
    monkeypatch.setattr(ml_suitability, "MAX_MODEL_ROWS", 3)
    data = pd.DataFrame(
        {
            "Feature": [1, 2, 3, 4],
            "Target": [10, 20, 30, 40],
        }
    )

    report = assess_ml_suitability(data, profile_dataset(data))
    size_check = next(
        item for item in report["checks"] if item["name"] == "Dataset size"
    )

    assert size_check["status"] == "review"
    assert "4 rows exceeds the 3-row limit" in size_check["evidence"]
