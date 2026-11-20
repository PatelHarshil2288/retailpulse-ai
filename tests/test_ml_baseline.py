import pandas as pd
import pytest

from src.data_profiling import profile_dataset
from src import ml_baseline
from src.ml_baseline import run_baseline_model_evaluation


def test_runs_regression_baseline_with_metrics():
    rows = 240
    data = pd.DataFrame(
        {
            "FeatureA": [float(index % 19) for index in range(rows)],
            "FeatureB": [float(index) * 0.3 for index in range(rows)],
            "Region": ["A" if index % 2 == 0 else "B" for index in range(rows)],
            "Target": [50 + index * 1.8 for index in range(rows)],
        }
    )
    profile = profile_dataset(data)

    report = run_baseline_model_evaluation(
        data,
        profile,
        "Target",
        "regression",
    )

    assert report["task_type"] == "regression"
    assert report["rows_used"] == rows
    assert report["feature_count"] >= 2
    assert len(report["models"]) == 2
    assert {"MAE", "RMSE", "R2"} <= set(report["models"][0].keys())
    assert report["best_model"] in {
        "Linear Regression",
        "Random Forest Regressor",
    }


def test_runs_classification_baseline_with_confusion_matrix():
    rows = 240
    data = pd.DataFrame(
        {
            "FeatureA": [float(index % 13) for index in range(rows)],
            "FeatureB": [float(index % 7) for index in range(rows)],
            "Channel": [
                "Retail" if index % 3 else "Online"
                for index in range(rows)
            ],
            "TargetClass": [
                "High" if index % 4 in {0, 1} else "Low"
                for index in range(rows)
            ],
        }
    )
    profile = profile_dataset(data)

    report = run_baseline_model_evaluation(
        data,
        profile,
        "TargetClass",
        "classification",
    )

    assert report["task_type"] == "classification"
    assert report["rows_used"] == rows
    assert report["feature_count"] >= 2
    assert len(report["models"]) == 2
    assert {"Accuracy", "Precision", "Recall", "F1"} <= set(
        report["models"][0].keys()
    )
    assert report["best_model"] in {
        "Logistic Regression",
        "Random Forest Classifier",
    }
    assert report["confusion_matrix"] is not None
    assert len(report["confusion_matrix"]["labels"]) == 2


def test_baseline_rejects_datasets_above_the_model_row_limit(monkeypatch):
    monkeypatch.setattr(ml_baseline, "MAX_MODEL_ROWS", 3)
    data = pd.DataFrame({"Target": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="limited to 3 rows"):
        run_baseline_model_evaluation(data, {}, "Target")
