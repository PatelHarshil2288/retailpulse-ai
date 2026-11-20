"""Unsupervised clustering and anomaly analysis for uploaded datasets."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.ml_limits import MAX_MODEL_ROWS


MIN_ROWS = 30
MAX_CATEGORY_LEVELS = 20
MAX_SILHOUETTE_SAMPLE = 1_000


def available_unsupervised_features(profile):
    """Return bounded numeric and categorical features for model controls."""
    options = []
    selected = []
    for column in profile["columns"]:
        detected_type = column["detected_type"]
        supported = detected_type in {"numeric", "boolean"} or (
            detected_type == "categorical"
            and column["unique_count"] <= MAX_CATEGORY_LEVELS
        )
        if not supported or column["constant"]:
            continue
        name = column["name"]
        label = f"{name} ({detected_type})"
        options.append({"label": label, "value": name})
        normalized_name = name.lower().replace("_", " ")
        looks_like_identifier = column["high_cardinality"] or any(
            token in normalized_name
            for token in (" id", "id ", "uuid", "invoice", "transaction")
        )
        if detected_type == "numeric" and not looks_like_identifier:
            selected.append(name)
    if not selected:
        selected = [option["value"] for option in options[:5]]
    return options, selected


def _prepare_features(data, profile, feature_columns):
    profile_by_name = {
        item["name"]: item for item in profile["columns"]
    }
    if not feature_columns:
        raise ValueError("Select at least one feature column.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Feature columns must be unique.")

    encoded_parts = []
    display_data = pd.DataFrame(index=data.index)
    dropped = []
    for name in feature_columns:
        if name not in data.columns or name not in profile_by_name:
            raise ValueError(f"Unknown feature column: {name}")
        column_profile = profile_by_name[name]
        if column_profile["constant"]:
            dropped.append(name)
            continue
        detected_type = column_profile["detected_type"]
        series = data[name]
        if detected_type == "numeric":
            values = pd.to_numeric(series, errors="coerce")
            values = values.where(np.isfinite(values), np.nan)
            encoded_parts.append(values.rename(name).to_frame())
            display_data[name] = values
        elif detected_type == "boolean":
            normalized = series.astype("string").str.strip().str.lower()
            values = normalized.map(
                {
                    "true": 1.0,
                    "false": 0.0,
                    "yes": 1.0,
                    "no": 0.0,
                    "y": 1.0,
                    "n": 0.0,
                    "1": 1.0,
                    "0": 0.0,
                }
            )
            encoded_parts.append(
                pd.to_numeric(values, errors="coerce").rename(name).to_frame()
            )
            display_data[name] = series
        elif detected_type == "categorical":
            if column_profile["unique_count"] > MAX_CATEGORY_LEVELS:
                raise ValueError(
                    f"{name} has more than {MAX_CATEGORY_LEVELS} categories; "
                    "choose a lower-cardinality feature."
                )
            categorical = series.astype("string").fillna("<missing>")
            one_hot = pd.get_dummies(
                categorical,
                prefix=name,
                dtype=float,
            )
            encoded_parts.append(one_hot)
            display_data[name] = categorical
        else:
            raise ValueError(
                f"{name} has type {detected_type}; choose numeric, boolean, "
                "or categorical features."
            )

    if not encoded_parts:
        raise ValueError("All selected features are constant or unusable.")

    encoded = pd.concat(encoded_parts, axis=1)
    encoded = encoded.dropna(axis=1, how="all")
    if encoded.empty:
        raise ValueError("No usable feature values remain after preprocessing.")
    encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0)
    if encoded.shape[1] > 100:
        raise ValueError(
            "Selected features expand to more than 100 model inputs; "
            "reduce feature selection."
        )
    return encoded, display_data, dropped


def _cluster_summaries(labels, display_data, numeric_features):
    frame = display_data.copy()
    frame["_cluster"] = labels
    summaries = []
    for cluster in sorted(pd.unique(labels)):
        rows = frame.loc[frame["_cluster"] == cluster]
        summary = {
            "cluster": int(cluster),
            "rows": int(len(rows)),
        }
        for name in numeric_features:
            summary[f"{name} mean"] = float(rows[name].mean())
        summaries.append(summary)
    return summaries


def run_unsupervised_analysis(
    data,
    profile,
    feature_columns,
    workflow,
    n_clusters=3,
):
    """Run K-means clustering or Isolation Forest with measured summaries."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Unsupervised analysis requires a pandas DataFrame.")
    if workflow not in {"clustering", "anomaly"}:
        raise ValueError(f"Unsupported unsupervised workflow: {workflow}")
    if len(data) > MAX_MODEL_ROWS:
        raise ValueError(
            f"Unsupervised model runs are limited to {MAX_MODEL_ROWS:,} "
            "rows. Upload a smaller dataset before analysis."
        )
    if len(data) < MIN_ROWS:
        raise ValueError(
            f"At least {MIN_ROWS} rows are required for unsupervised analysis."
        )
    if workflow == "clustering" and (
        not isinstance(n_clusters, int) or not 2 <= n_clusters <= 10
    ):
        raise ValueError("Cluster count must be an integer from 2 to 10.")

    encoded, display_data, dropped = _prepare_features(
        data,
        profile,
        feature_columns,
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(encoded)

    if workflow == "clustering":
        if len(data) <= n_clusters:
            raise ValueError(
                "The dataset must contain more rows than the requested clusters."
            )
        labels = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        ).fit_predict(scaled)
        unique_labels = np.unique(labels)
        silhouette = None
        if 2 <= len(unique_labels) < len(labels):
            sample_indices = np.random.default_rng(42).choice(
                len(labels),
                size=min(len(labels), MAX_SILHOUETTE_SAMPLE),
                replace=False,
            )
            sample_labels = labels[sample_indices]
            if 2 <= len(np.unique(sample_labels)) < len(sample_labels):
                silhouette = float(
                    silhouette_score(
                        scaled[sample_indices],
                        sample_labels,
                    )
                )
        profile_by_name = {
            item["name"]: item for item in profile["columns"]
        }
        numeric_features = [
            name
            for name in feature_columns
            if profile_by_name[name]["detected_type"] == "numeric"
            and name in display_data.columns
        ]
        return {
            "workflow": workflow,
            "rows_used": int(len(data)),
            "feature_count": int(encoded.shape[1]),
            "selected_features": list(feature_columns),
            "dropped_features": dropped,
            "cluster_count": int(len(unique_labels)),
            "silhouette_score": silhouette,
            "clusters": _cluster_summaries(
                labels,
                display_data,
                numeric_features,
            ),
            "notes": [
                "K-means uses standardized inputs and a fixed random seed.",
                "Cluster labels are arbitrary; summaries require domain interpretation.",
                "Silhouette score is descriptive and is omitted when only one cluster is produced.",
            ],
        }

    model = IsolationForest(
        contamination="auto",
        random_state=42,
        n_estimators=200,
    )
    labels = model.fit_predict(scaled)
    anomaly_scores = -model.score_samples(scaled)
    anomaly_positions = np.flatnonzero(labels == -1)
    top_positions = sorted(
        anomaly_positions,
        key=lambda position: anomaly_scores[position],
        reverse=True,
    )[:10]
    return {
        "workflow": workflow,
        "rows_used": int(len(data)),
        "feature_count": int(encoded.shape[1]),
        "selected_features": list(feature_columns),
        "dropped_features": dropped,
        "anomaly_count": int(len(anomaly_positions)),
        "anomaly_percent": float(100 * len(anomaly_positions) / len(data)),
        "top_candidates": [
            {
                "row_number": int(position + 1),
                "anomaly_score": float(anomaly_scores[position]),
            }
            for position in top_positions
        ],
        "notes": [
            "Isolation Forest anomaly scores are relative ranking signals, not proof of errors.",
            "Review flagged records in business context before taking action.",
        ],
    }
