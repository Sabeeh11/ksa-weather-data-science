"""K-Means clustering for weather regime discovery."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PATTERN_NAMES = {
    0: "Humid Stable Air Mass",
    1: "Cold Dry Continental Air",
    2: "Warm Dry Transitional Air Mass",
    3: "Low-Ceiling Stratiform Cloud Regime",
}


def scale_features(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale features and return a DataFrame with the original index."""
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
    return X_scaled, scaler


def run_kmeans(
    df: pd.DataFrame,
    X: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Fit K-Means and attach cluster labels to the DataFrame."""
    df = df.copy()
    X_scaled, scaler = scale_features(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)

    df["WEATHER_PATTERN"] = kmeans.labels_
    df["WEATHER_PATTERN_NAME"] = df["WEATHER_PATTERN"].map(PATTERN_NAMES)

    return df, kmeans, scaler


def get_cluster_centers(kmeans: KMeans, feature_names: list[str]) -> pd.DataFrame:
    """Return cluster centers as a DataFrame."""
    return pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)
