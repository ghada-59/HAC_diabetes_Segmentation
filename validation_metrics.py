"""
Clustering validation metrics for HAC analysis.
"""

import logging
from typing import Dict, List

import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


def comprehensive_cluster_validation(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> Dict[str, float]:
    """Compute internal validation metrics for one clustering solution."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of samples.")
    if k < 2:
        raise ValueError("k must be at least 2.")

    n_clusters = len(np.unique(labels))
    if n_clusters != k:
        raise ValueError(f"Expected {k} clusters, got {n_clusters}.")
    if k >= X.shape[0]:
        raise ValueError("k must be smaller than the number of samples.")

    return {
        "silhouette_score": float(silhouette_score(X, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz_index": float(calinski_harabasz_score(X, labels)),
    }


def evaluate_k_range(
    X: np.ndarray,
    linkage_matrix: np.ndarray,
    k_range: List[int],
) -> Dict[int, Dict[str, float]]:
    """Evaluate clustering quality for each requested k."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")
    if linkage_matrix.ndim != 2 or linkage_matrix.shape[1] != 4:
        raise ValueError("linkage_matrix must have shape (n_samples - 1, 4).")
    if not k_range:
        raise ValueError("k_range must not be empty.")

    results: Dict[int, Dict[str, float]] = {}
    for k in k_range:
        if k < 2 or k >= X.shape[0]:
            logger.warning("Skipping invalid k=%s.", k)
            continue

        labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
        actual_k = len(np.unique(labels))
        if actual_k != k:
            logger.warning("Skipping k=%s because fcluster produced %s clusters.", k, actual_k)
            continue

        results[k] = comprehensive_cluster_validation(X, labels, k)

    if not results:
        raise ValueError("No valid clustering solutions were evaluated.")

    return results
