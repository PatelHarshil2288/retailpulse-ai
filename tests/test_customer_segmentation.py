
import numpy as np
import pandas as pd
import pytest

from src.customer_segmentation import (
    build_rfm_features,
    build_rfm_features_from_csv,
    prepare_rfm_features,
    evaluate_cluster_counts,
    fit_customer_segments,
)


@pytest.fixture
def sample_sales():
    """Provide small sales data with several distinct customer profiles."""
    return pd.DataFrame(
        {
            "CustomerID": [
                101, 101, 102, 102, 103, 104, 105, 106,
            ],
            "InvoiceNo": [
                "A1", "A2", "B1", "B2", "C1", "D1", "E1", "F1",
            ],
            "InvoiceDate": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-10",
                    "2021-01-05",
                    "2021-01-08",
                    "2021-01-15",
                    "2021-01-20",
                    "2021-01-25",
                    "2021-01-30",
                ]
            ),
            "Revenue": [100, 150, 50, 60, 500, 20, 25, 900],
        }
    )


def test_build_rfm_features_creates_one_row_per_customer(sample_sales):
    rfm = build_rfm_features(
        sample_sales,
        reference_date="2021-02-01",
    )

    assert len(rfm) == 6
    assert rfm.index.is_unique
    assert list(rfm.columns) == ["Recency", "Frequency", "Monetary"]


def test_build_rfm_features_calculates_expected_values(sample_sales):
    rfm = build_rfm_features(
        sample_sales,
        reference_date="2021-02-01",
    )

    assert rfm.loc[101, "Frequency"] == 2
    assert rfm.loc[101, "Monetary"] == 250
    assert rfm.loc[101, "Recency"] == 22


def test_build_rfm_features_excludes_missing_customer_ids(sample_sales):
    sample_sales.loc[0, "CustomerID"] = np.nan

    rfm = build_rfm_features(
        sample_sales,
        reference_date="2021-02-01",
    )

    assert np.nan not in rfm.index
    assert len(rfm) == 6


def test_build_rfm_features_rejects_negative_revenue(sample_sales):
    sample_sales.loc[0, "Revenue"] = -10

    with pytest.raises(ValueError, match="Negative revenue"):
        build_rfm_features(sample_sales)


def test_prepare_rfm_features_standardizes_data(sample_sales):
    rfm = build_rfm_features(
        sample_sales,
        reference_date="2021-02-01",
    )

    scaled, scaler = prepare_rfm_features(rfm)

    assert scaled.shape == rfm.shape
    assert np.isfinite(scaled.to_numpy()).all()
    assert scaler is not None
    assert np.allclose(scaled.mean().to_numpy(), 0, atol=1e-10)


def test_prepare_rfm_features_rejects_invalid_values():
    invalid_rfm = pd.DataFrame(
        {
            "Recency": [1, np.nan],
            "Frequency": [2, 3],
            "Monetary": [100, 200],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        prepare_rfm_features(invalid_rfm)


def test_evaluate_cluster_counts_returns_valid_scores(sample_sales):
    rfm = build_rfm_features(
        sample_sales,
        reference_date="2021-02-01",
    )
    scaled, _ = prepare_rfm_features(rfm)

    results = evaluate_cluster_counts(
        scaled,
        min_clusters=2,
        max_clusters=3,
    )

    assert not results.empty
    assert {"Clusters", "Silhouette Score", "Inertia"} <= set(results.columns)
    assert results["Silhouette Score"].between(-1, 1).all()


def test_fit_customer_segments_assigns_every_customer(sample_sales):
    rfm = build_rfm_features(
        sample_sales,
        reference_date="2021-02-01",
    )
    scaled, _ = prepare_rfm_features(rfm)

    assignments, profiles, model = fit_customer_segments(
        scaled,
        n_clusters=2,
    )

    assert len(assignments) == len(rfm)
    assert assignments.index.is_unique
    assert assignments["Cluster"].nunique() == 2
    assert profiles["CustomerCount"].sum() == len(rfm)
    assert model.n_clusters == 2


def test_fit_customer_segments_is_reproducible(sample_sales):
    rfm = build_rfm_features(
        sample_sales,
        reference_date="2021-02-01",
    )
    scaled, _ = prepare_rfm_features(rfm)

    first, _, _ = fit_customer_segments(scaled, n_clusters=2)
    second, _, _ = fit_customer_segments(scaled, n_clusters=2)

    pd.testing.assert_series_equal(
        first["Cluster"],
        second["Cluster"],
    )


def test_fit_customer_segments_rejects_too_few_customers():
    scaled = pd.DataFrame(
        {
            "Recency": [0.0, 1.0, 2.0],
            "Frequency": [0.0, 1.0, 2.0],
            "Monetary": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="At least 4 customers"):
        fit_customer_segments(scaled, n_clusters=2)



def test_build_rfm_features_from_csv_matches_in_memory(
    sample_sales, tmp_path
):
    csv_path = tmp_path / "sales.csv"
    sample_sales.to_csv(csv_path, index=False)

    expected = build_rfm_features(
        sample_sales,
        reference_date="2021-02-01",
    )
    actual = build_rfm_features_from_csv(
        csv_path,
        reference_date="2021-02-01",
        chunksize=3,
    )

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_exact=False,
        rtol=1e-9,
        atol=1e-6,
    )


def test_build_rfm_features_from_csv_counts_invoices_across_chunks(
    tmp_path,
):
    csv_path = tmp_path / "sales.csv"
    sales = pd.DataFrame(
        {
            "CustomerID": [101, 101, 101],
            "InvoiceNo": ["A1", "A1", "A2"],
            "InvoiceDate": pd.to_datetime(
                ["2021-01-01", "2021-01-01", "2021-01-10"]
            ),
            "Revenue": [100.0, 50.0, 150.0],
        }
    )
    sales.to_csv(csv_path, index=False)

    rfm = build_rfm_features_from_csv(
        csv_path,
        reference_date="2021-02-01",
        chunksize=1,
    )

    assert rfm.loc[101, "Frequency"] == 2
    assert rfm.loc[101, "Monetary"] == pytest.approx(300.0)
    assert rfm.loc[101, "Recency"] == 22


def test_build_rfm_features_from_csv_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        build_rfm_features_from_csv(missing_path)


def test_build_rfm_features_from_csv_rejects_missing_columns(
    tmp_path,
):
    csv_path = tmp_path / "sales.csv"
    pd.DataFrame({"CustomerID": [101], "Revenue": [100.0]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        build_rfm_features_from_csv(csv_path)


@pytest.mark.parametrize("chunksize", [0, -1, 1.5, True])
def test_build_rfm_features_from_csv_rejects_invalid_chunksize(
    tmp_path, chunksize
):
    csv_path = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "CustomerID": [101],
            "InvoiceNo": ["A1"],
            "InvoiceDate": ["2021-01-01"],
            "Revenue": [100.0],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="chunksize must be a positive integer"):
        build_rfm_features_from_csv(csv_path, chunksize=chunksize)
