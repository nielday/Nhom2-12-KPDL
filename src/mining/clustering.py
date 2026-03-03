"""
Clustering module.

Implements KMeans and DBSCAN clustering with cluster
profiling and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# Default behavioral features for clustering
DEFAULT_FEATURES = [
    "lead_time", "total_stays", "total_guests", "adr",
    "booking_changes", "previous_cancellations",
    "days_in_waiting_list", "total_of_special_requests",
]


def prepare_clustering_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Select and standardize features for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with engineered features.
    feature_cols : list of str, optional
        Columns to use. Defaults to DEFAULT_FEATURES.

    Returns
    -------
    tuple of (X_scaled, scaler, used_cols)
    """
    if feature_cols is None:
        feature_cols = [
            c for c in DEFAULT_FEATURES if c in df.columns
        ]

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(
        f"[Clustering] Prepared {X_scaled.shape[0]:,} samples "
        f"× {X_scaled.shape[1]} features"
    )
    return X_scaled, scaler, feature_cols


def find_optimal_k(
    X: np.ndarray,
    k_range: Optional[List[int]] = None,
    seed: int = 42,
) -> Dict[str, List]:
    """
    Find optimal K using Elbow and Silhouette methods.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    k_range : list of int
        Range of K values to try.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys 'k_values', 'inertias', 'silhouettes'
    """
    if k_range is None:
        k_range = list(range(2, 9))

    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(
            n_clusters=k, random_state=seed, n_init=10,
        )
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
        print(
            f"  K={k}: inertia={km.inertia_:.0f}, "
            f"silhouette={sil:.4f}"
        )

    best_k = k_range[np.argmax(silhouettes)]
    print(
        f"[Clustering] Best K by silhouette: {best_k} "
        f"(score={max(silhouettes):.4f})"
    )

    return {
        "k_values": k_range,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
    }


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    """
    Run KMeans clustering.

    Returns
    -------
    tuple of (labels, fitted_model)
    """
    km = KMeans(
        n_clusters=n_clusters, random_state=seed, n_init=10,
    )
    labels = km.fit_predict(X)
    print(
        f"[Clustering] KMeans K={n_clusters}: "
        f"inertia={km.inertia_:.0f}"
    )
    return labels, km


def run_dbscan(
    X: np.ndarray,
    eps: Optional[float] = None,
    min_samples: int = 5,
) -> Tuple[np.ndarray, DBSCAN]:
    """
    Run DBSCAN clustering.

    If eps is None, it is estimated from k-distance graph.

    Returns
    -------
    tuple of (labels, fitted_model)
    """
    if eps is None:
        # Estimate eps from k-distance graph
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        k_distances = np.sort(distances[:, -1])
        # Use knee point heuristic (90th percentile)
        eps = float(np.percentile(k_distances, 90))
        print(f"[Clustering] DBSCAN auto eps={eps:.4f}")

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(
        f"[Clustering] DBSCAN: {n_clusters} clusters, "
        f"{n_noise} noise points"
    )
    return labels, db


def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    target_col: str = "is_canceled",
) -> pd.DataFrame:
    """
    Create cluster profiles with statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    labels : np.ndarray
        Cluster labels for each row.
    feature_cols : list of str
        Feature columns to profile.
    target_col : str
        Target column for cancel rate.

    Returns
    -------
    pd.DataFrame
        Cluster profile table.
    """
    df_temp = df.copy()
    df_temp["cluster"] = labels

    # Calculate profiles
    profiles = df_temp.groupby("cluster")[feature_cols].mean()
    profiles = profiles.round(2)

    # Add cancel rate
    if target_col in df_temp.columns:
        cancel_rate = df_temp.groupby("cluster")[
            target_col
        ].mean()
        profiles["cancel_rate"] = cancel_rate.round(4)

    # Add cluster sizes
    sizes = df_temp["cluster"].value_counts().sort_index()
    profiles["size"] = sizes
    profiles["pct"] = (sizes / len(df_temp) * 100).round(1)

    print("[Clustering] Cluster profiles:")
    print(profiles.to_string())
    return profiles


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate clustering quality.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    labels : np.ndarray
        Cluster labels.

    Returns
    -------
    dict with silhouette, dbi, calinski_harabasz scores.
    """
    # Filter out noise points for evaluation
    mask = labels != -1
    if mask.sum() < 2:
        return {
            "silhouette": float("nan"),
            "dbi": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    X_eval = X[mask]
    labels_eval = labels[mask]

    n_unique = len(set(labels_eval))
    if n_unique < 2:
        return {
            "silhouette": float("nan"),
            "dbi": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    sil = silhouette_score(X_eval, labels_eval)
    dbi = davies_bouldin_score(X_eval, labels_eval)
    ch = calinski_harabasz_score(X_eval, labels_eval)

    metrics = {
        "silhouette": round(sil, 4),
        "dbi": round(dbi, 4),
        "calinski_harabasz": round(ch, 2),
    }
    print(f"[Clustering] Evaluation: {metrics}")
    return metrics


def reduce_to_2d(X: np.ndarray) -> np.ndarray:
    """Reduce features to 2D using PCA for visualization."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    explained = sum(pca.explained_variance_ratio_) * 100
    print(
        f"[Clustering] PCA 2D: "
        f"{explained:.1f}% variance explained"
    )
    return X_2d
