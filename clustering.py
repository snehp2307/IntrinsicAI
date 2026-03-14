"""
clustering.py
=============
Implements unsupervised learning for company financial health grouping:
  - K-Means Clustering: segments companies into financial health clusters
  - PCA (2D): reduces feature space for scatter-plot visualisation

Cluster labelling uses post-hoc analysis of cluster centroids to assign
meaningful names (e.g. "Financially Healthy", "Distressed").
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


N_CLUSTERS = 4   # Empirically chosen for interpretable financial segmentation

CLUSTER_LABELS = {
    0: "Financially Stable",
    1: "Moderate Stress",
    2: "High Distress",
    3: "Leveraged / Growth",
}


def run_kmeans(X_scaled: np.ndarray, n_clusters: int = N_CLUSTERS) -> np.ndarray:
    """
    Fit K-Means and return cluster assignments (0-indexed integers).
    """
    km = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        max_iter=300,
        random_state=42,
    )
    labels = km.fit_predict(X_scaled)
    return labels, km


def run_pca(X_scaled: np.ndarray) -> tuple:
    """
    Reduce feature matrix to 2 principal components for visualisation.
    Returns (pca_coords [n×2], explained_variance [2]).
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    variance = pca.explained_variance_ratio_.tolist()
    return coords, variance


def label_clusters(df: pd.DataFrame, km: KMeans, feature_cols: list) -> dict:
    """
    Analyse cluster centroids (in original standardised space) to assign
    human-readable labels.  Returns {cluster_id: label_string}.
    """
    centroids = km.cluster_centers_  # shape: (n_clusters, n_features)

    # Map feature positions for key indicators
    try:
        cr_idx  = feature_cols.index("current_ratio")
        dte_idx = feature_cols.index("debt_to_equity")
        pm_idx  = feature_cols.index("profit_margin")
        ic_idx  = feature_cols.index("interest_coverage")
    except ValueError:
        return {i: f"Cluster {i}" for i in range(len(centroids))}

    cluster_scores = []
    for i, c in enumerate(centroids):
        # Heuristic health score from centroid values (higher = healthier)
        health = (
            c[cr_idx]  * 1.5   # liquidity
            - c[dte_idx] * 1.2  # penalise leverage
            + c[pm_idx]  * 1.5  # profitability
            + c[ic_idx]  * 1.0  # debt service
        )
        cluster_scores.append((i, health))

    cluster_scores.sort(key=lambda x: -x[1])  # best → worst

    preset = [
        "Financially Stable",
        "Moderate Stress",
        "High Distress",
        "Leveraged / Growth",
    ]
    label_map = {}
    for rank, (cluster_id, _) in enumerate(cluster_scores):
        label_map[cluster_id] = preset[rank % len(preset)]

    return label_map


def get_cluster_summary(df: pd.DataFrame, feature_cols: list) -> list:
    """
    Return per-cluster summary statistics for the frontend.
    """
    summary = []
    for cid in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cid]
        entry = {
            "cluster_id":    int(cid),
            "cluster_label": subset["cluster_label"].iloc[0],
            "count":         int(len(subset)),
            "avg_risk_score": round(float(subset["risk_score"].mean()), 1),
            "high_risk_pct": round(
                float((subset["risk_category"] == "High Risk").mean() * 100), 1
            ),
        }
        summary.append(entry)
    return summary
