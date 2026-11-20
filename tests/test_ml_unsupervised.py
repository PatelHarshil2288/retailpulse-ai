import numpy as np
import pandas as pd
import pytest

from src.data_profiling import profile_dataset
from src import ml_unsupervised
from src.ml_unsupervised import (
    available_unsupervised_features,
    run_unsupervised_analysis,
)


@pytest.fixture
def cluster_data():
    rng = np.random.default_rng(42)
    first = rng.normal(loc=0, scale=0.25, size=(60, 2))
    second = rng.normal(loc=5, scale=0.25, size=(60, 2))
    return pd.DataFrame(
        {
            "MeasureA": np.concatenate([first[:, 0], second[:, 0]]),
            "MeasureB": np.concatenate([first[:, 1], second[:, 1]]),
            "Segment": ["A"] * 60 + ["B"] * 60,
            "Constant": [1] * 120,
        }
    )


def test_available_features_exclude_constant_columns(cluster_data):
    profile = profile_dataset(cluster_data)

    options, selected = available_unsupervised_features(profile)

    assert "Constant" not in {option["value"] for option in options}
    assert selected == ["MeasureA", "MeasureB"]


def test_kmeans_returns_cluster_summaries_and_silhouette(cluster_data):
    profile = profile_dataset(cluster_data)

    report = run_unsupervised_analysis(
        cluster_data,
        profile,
        ["MeasureA", "MeasureB", "Segment", "Constant"],
        "clustering",
        2,
    )

    assert report["workflow"] == "clustering"
    assert report["cluster_count"] == 2
    assert report["silhouette_score"] is not None
    assert sum(item["rows"] for item in report["clusters"]) == 120
    assert report["dropped_features"] == ["Constant"]


def test_isolation_forest_returns_ranked_row_candidates():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(100, 2))
    outliers = np.array([[10.0, 10.0], [-10.0, -10.0]])
    data = pd.DataFrame(
        np.vstack([normal, outliers]),
        columns=["MetricA", "MetricB"],
    )
    profile = profile_dataset(data)

    report = run_unsupervised_analysis(
        data,
        profile,
        ["MetricA", "MetricB"],
        "anomaly",
    )

    assert report["workflow"] == "anomaly"
    assert report["anomaly_count"] > 0
    assert report["top_candidates"]
    assert report["top_candidates"][0]["row_number"] in {101, 102}
    assert report["top_candidates"][0]["anomaly_score"] > 0


def test_unsupervised_analysis_rejects_small_dataset(cluster_data):
    data = cluster_data.head(10)

    with pytest.raises(ValueError, match="At least 30 rows"):
        run_unsupervised_analysis(
            data,
            profile_dataset(data),
            ["MeasureA", "MeasureB"],
            "clustering",
            2,
        )


def test_unsupervised_analysis_rejects_high_cardinality_category():
    data = pd.DataFrame(
        {
            "Value": np.arange(40, dtype=float),
            "Category": [f"Category {index}" for index in range(40)],
        }
    )

    with pytest.raises(ValueError, match="more than 20 categories"):
        run_unsupervised_analysis(
            data,
            profile_dataset(data),
            ["Value", "Category"],
            "clustering",
            2,
        )


def test_unsupervised_analysis_rejects_datasets_above_the_model_row_limit(
    monkeypatch,
):
    monkeypatch.setattr(ml_unsupervised, "MAX_MODEL_ROWS", 30)
    data = pd.DataFrame({"Value": np.arange(31, dtype=float)})

    with pytest.raises(ValueError, match="limited to 30 rows"):
        run_unsupervised_analysis(data, {}, ["Value"], "anomaly")
