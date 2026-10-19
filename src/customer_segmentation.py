
"""Customer segmentation utilities for RetailPulse AI.

Build customer-level RFM features and fit reproducible K-Means clusters.
Segments describe historical purchasing behavior in the available dataset.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path


RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]
DEFAULT_RANDOM_STATE = 42
MINIMUM_CUSTOMERS = 4


def build_rfm_features(sales, reference_date=None):
    """Build one Recency, Frequency, Monetary row per identified customer."""
    required_columns = {
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "Revenue",
    }
    missing_columns = required_columns - set(sales.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if sales.empty:
        raise ValueError("Sales data cannot be empty.")

    data = sales[list(required_columns)].copy()
    data["CustomerID"] = pd.to_numeric(
        data["CustomerID"], errors="coerce"
    )
    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )
    data["Revenue"] = pd.to_numeric(data["Revenue"], errors="coerce")

    data = data.dropna(
        subset=["CustomerID", "InvoiceNo", "InvoiceDate", "Revenue"]
    ).copy()

    if data.empty:
        raise ValueError(
            "No valid transactions with identified customers remain."
        )

    if not np.isfinite(data["Revenue"].to_numpy(dtype=float)).all():
        raise ValueError("Revenue must contain only finite values.")

    # Customer IDs are identifiers, not quantities. Keep only valid IDs.
    data = data.loc[data["CustomerID"] > 0].copy()

    if data.empty:
        raise ValueError("No valid positive customer IDs remain.")

    if (data["Revenue"] < 0).any():
        raise ValueError(
            "Negative revenue found. Supply ordinary positive-sales "
            "transactions rather than returns or accounting adjustments."
        )

    if reference_date is None:
        reference_date = data["InvoiceDate"].max().normalize()
    else:
        reference_date = pd.Timestamp(reference_date).normalize()

    if reference_date < data["InvoiceDate"].max().normalize():
        raise ValueError(
            "reference_date cannot be earlier than the latest transaction."
        )

    customer_features = (
        data.groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
    )

    customer_features["Recency"] = (
        reference_date - customer_features["LastPurchase"].dt.normalize()
    ).dt.days

    customer_features = customer_features[RFM_COLUMNS].astype(float)

    if not np.isfinite(customer_features.to_numpy()).all():
        raise ValueError("RFM features must contain only finite values.")

    if (customer_features["Frequency"] < 1).any():
        raise ValueError("Customer frequency must be at least one.")

    if (customer_features["Monetary"] <= 0).any():
        raise ValueError("Customer monetary value must be positive.")

    customer_features.index.name = "CustomerID"
    return customer_features.sort_index()


def build_rfm_features_from_csv(
    csv_path,
    reference_date=None,
    chunksize=10000,
):
    """Build customer RFM features from CSV chunks with disk-backed invoice counting."""
    import sqlite3
    import tempfile
    from contextlib import closing

    csv_path = Path(csv_path)

    required_columns = {
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "Revenue",
    }

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if (
        not isinstance(chunksize, int)
        or isinstance(chunksize, bool)
        or chunksize < 1
    ):
        raise ValueError("chunksize must be a positive integer.")

    header = pd.read_csv(csv_path, nrows=0)
    missing_columns = required_columns - set(header.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    customer_parts = []
    latest_transaction = None

    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = Path(temp_dir) / "customer_invoices.sqlite"

        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE customer_invoices (
                    CustomerID REAL NOT NULL,
                    InvoiceNo TEXT NOT NULL,
                    PRIMARY KEY (CustomerID, InvoiceNo)
                ) WITHOUT ROWID
                """
            )

            for chunk in pd.read_csv(
                csv_path,
                usecols=list(required_columns),
                chunksize=chunksize,
            ):
                chunk["CustomerID"] = pd.to_numeric(
                    chunk["CustomerID"], errors="coerce"
                )
                chunk["InvoiceDate"] = pd.to_datetime(
                    chunk["InvoiceDate"], errors="coerce"
                )
                chunk["Revenue"] = pd.to_numeric(
                    chunk["Revenue"], errors="coerce"
                )

                chunk = chunk.dropna(
                    subset=[
                        "CustomerID",
                        "InvoiceNo",
                        "InvoiceDate",
                        "Revenue",
                    ]
                ).copy()

                if chunk.empty:
                    continue

                if not np.isfinite(
                    chunk["Revenue"].to_numpy(dtype=float)
                ).all():
                    raise ValueError(
                        "Revenue must contain only finite values."
                    )

                chunk = chunk.loc[chunk["CustomerID"] > 0].copy()

                if chunk.empty:
                    continue

                if (chunk["Revenue"] < 0).any():
                    raise ValueError(
                        "Negative revenue found. Supply ordinary positive-sales "
                        "transactions rather than returns or accounting adjustments."
                    )

                chunk_latest = chunk["InvoiceDate"].max()

                if (
                    latest_transaction is None
                    or chunk_latest > latest_transaction
                ):
                    latest_transaction = chunk_latest

                grouped = chunk.groupby("CustomerID").agg(
                    LastPurchase=("InvoiceDate", "max"),
                    Monetary=("Revenue", "sum"),
                )
                customer_parts.append(grouped)

                # Deduplicate invoice pairs on disk, including pairs spanning chunks.
                pairs = chunk[["CustomerID", "InvoiceNo"]].drop_duplicates()

                connection.executemany(
                    """
                    INSERT OR IGNORE INTO customer_invoices
                    (CustomerID, InvoiceNo)
                    VALUES (?, ?)
                    """,
                    (
                        (float(customer_id), str(invoice_no))
                        for customer_id, invoice_no in pairs.itertuples(
                            index=False, name=None
                        )
                    ),
                )

            if not customer_parts:
                raise ValueError(
                    "No valid transactions with identified customers remain."
                )

            customer_features = (
                pd.concat(customer_parts)
                .groupby(level=0)
                .agg(
                    LastPurchase=("LastPurchase", "max"),
                    Monetary=("Monetary", "sum"),
                )
            )

            if reference_date is None:
                reference_date = latest_transaction.normalize()
            else:
                reference_date = pd.Timestamp(reference_date).normalize()

            if reference_date < latest_transaction.normalize():
                raise ValueError(
                    "reference_date cannot be earlier than the latest transaction."
                )
            connection.commit()

            frequency = pd.read_sql_query(
                """
                SELECT CustomerID, COUNT(*) AS Frequency
                FROM customer_invoices
                GROUP BY CustomerID
                """,
                connection,
                index_col="CustomerID",
            )["Frequency"]

    customer_features["Frequency"] = frequency.reindex(
        customer_features.index
    ).fillna(0)

    customer_features["Recency"] = (
        reference_date
        - customer_features["LastPurchase"].dt.normalize()
    ).dt.days

    customer_features = customer_features[RFM_COLUMNS].astype(float)

    if not np.isfinite(customer_features.to_numpy()).all():
        raise ValueError("RFM features must contain only finite values.")

    if (customer_features["Frequency"] < 1).any():
        raise ValueError("Customer frequency must be at least one.")

    if (customer_features["Monetary"] <= 0).any():
        raise ValueError("Customer monetary value must be positive.")

    customer_features.index.name = "CustomerID"
    return customer_features.sort_index()

def prepare_rfm_features(rfm):
    """Validate RFM data and return standardized clustering features."""
    if not isinstance(rfm, pd.DataFrame):
        raise TypeError("rfm must be a pandas DataFrame.")

    missing_columns = set(RFM_COLUMNS) - set(rfm.columns)
    if missing_columns:
        raise ValueError(
            f"Missing RFM columns: {sorted(missing_columns)}"
        )

    if rfm.empty:
        raise ValueError("RFM data cannot be empty.")

    features = rfm[RFM_COLUMNS].copy()

    for column in RFM_COLUMNS:
        features[column] = pd.to_numeric(
            features[column], errors="raise"
        )

    values = features.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("RFM features must contain finite values.")

    if (features["Recency"] < 0).any():
        raise ValueError("Recency cannot be negative.")

    if (features["Frequency"] < 1).any():
        raise ValueError("Frequency must be at least one.")

    if (features["Monetary"] <= 0).any():
        raise ValueError("Monetary value must be positive.")

    # RFM features are positive and generally right-skewed.
    transformed = np.log1p(features)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(transformed)

    scaled = pd.DataFrame(
        scaled_values,
        index=features.index,
        columns=RFM_COLUMNS,
    )

    return scaled, scaler


def evaluate_cluster_counts(scaled_rfm, min_clusters=2, max_clusters=6):
    """Evaluate feasible K-Means cluster counts using silhouette scores."""
    if not isinstance(scaled_rfm, pd.DataFrame):
        raise TypeError("scaled_rfm must be a pandas DataFrame.")

    if scaled_rfm.empty:
        raise ValueError("Scaled RFM data cannot be empty.")

    values = scaled_rfm.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Scaled RFM data must contain finite values.")

    customer_count = len(scaled_rfm)

    if customer_count < MINIMUM_CUSTOMERS:
        raise ValueError(
            f"At least {MINIMUM_CUSTOMERS} customers are required."
        )

    if (
        not isinstance(min_clusters, int)
        or not isinstance(max_clusters, int)
        or min_clusters < 2
        or max_clusters < min_clusters
    ):
        raise ValueError("Invalid cluster-count range.")

    if np.unique(values, axis=0).shape[0] < 2:
        raise ValueError("At least two distinct customer profiles are required.")

    rows = []

    for cluster_count in range(
        min_clusters, min(max_clusters, customer_count - 1) + 1
    ):
        model = KMeans(
            n_clusters=cluster_count,
            random_state=DEFAULT_RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(values)
        distinct_labels = np.unique(labels)

        # Silhouette is undefined when all customers form one cluster
        # or each customer becomes its own cluster.
        if len(distinct_labels) < 2 or len(distinct_labels) >= customer_count:
            continue

        score = silhouette_score(
            values,
            labels,
            sample_size=min(300, len(values)),
            random_state=DEFAULT_RANDOM_STATE,
        )

        rows.append(
            {
                "Clusters": cluster_count,
                "Silhouette Score": float(score),
                "Inertia": float(model.inertia_),
                "Smallest Cluster": int(
                    pd.Series(labels).value_counts().min()
                ),
            }
        )

    if not rows:
        raise ValueError(
            "No valid cluster count could be evaluated for these customers."
        )

    return pd.DataFrame(rows).sort_values(
        "Clusters", ignore_index=True
    )


def fit_customer_segments(scaled_rfm, n_clusters=None):
    """Fit K-Means and return customer assignments and segment profiles."""
    if not isinstance(scaled_rfm, pd.DataFrame):
        raise TypeError("scaled_rfm must be a pandas DataFrame.")

    if scaled_rfm.empty:
        raise ValueError("Scaled RFM data cannot be empty.")

    values = scaled_rfm.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Scaled RFM data must contain finite values.")

    customer_count = len(scaled_rfm)

    if customer_count < MINIMUM_CUSTOMERS:
        raise ValueError(
            f"At least {MINIMUM_CUSTOMERS} customers are required."
        )

    if n_clusters is None:
        evaluation = evaluate_cluster_counts(scaled_rfm)
        n_clusters = int(
            evaluation.loc[
                evaluation["Silhouette Score"].idxmax(), "Clusters"
            ]
        )

    if (
        not isinstance(n_clusters, int)
        or isinstance(n_clusters, bool)
        or not 2 <= n_clusters < customer_count
    ):
        raise ValueError(
            "n_clusters must be an integer between 2 and customer_count - 1."
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=DEFAULT_RANDOM_STATE,
        n_init=10,
    )
    labels = model.fit_predict(values)

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "K-Means produced fewer distinct clusters than requested."
        )

    assignments = pd.DataFrame(
        {
            "CustomerID": scaled_rfm.index,
            "Cluster": labels,
        }
    ).set_index("CustomerID")

    profiles = (
        scaled_rfm.assign(Cluster=labels)
        .groupby("Cluster")
        .agg(
            CustomerCount=("Recency", "size"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
        )
        .reset_index()
    )

    return assignments, profiles, model
