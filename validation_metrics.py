"""
Comprehensive clustering validation metrics for HAC analysis.

Provides multiple internal validation indices:
- Silhouette Score (cohesion vs. separation)
- Davies-Bouldin Index (cluster compactness)
- Calinski-Harabasz Index (ratio of between to within cluster variance)
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score,
)
import logging

logger = logging.getLogger(__name__)


def comprehensive_cluster_validation(
    X: np.ndarray,
    labels: np.ndarray,
    k: int
) -> Dict[str, float]:
    """
    Compute all validation metrics for clustering solution.
    
    Args:
        X: Feature matrix (samples × features)
        labels: Cluster assignments
        k: Number of clusters
        
    Returns:
        Dictionary of validation metrics
        
    Raises:
        ValueError: If invalid inputs
    """
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have same number of samples")
    
    if len(np.unique(labels)) != k:
        raise ValueError(f"Expected {k} clusters, got {len(np.unique(labels))}")
    
    metrics = {}
    
    # Silhouette Score [-1, 1]: Higher is better (≥ 0.5 is good)
    try:
        silhouette = silhouette_score(X, labels)
        metrics['silhouette_score'] = float(silhouette)
        logger.info(f"Silhouette Score: {silhouette:.4f}")
    except Exception as e:
        logger.warning(f"Silhouette score failed: {e}")
        metrics['silhouette_score'] = np.nan
    
    # Davies-Bouldin Index [0, ∞): Lower is better (< 1.5 is good)
    try:
        db_index = davies_bouldin_score(X, labels)
        metrics['davies_bouldin_index'] = float(db_index)
        logger.info(f"Davies-Bouldin Index: {db_index:.4f}")
    except Exception as e:
        logger.warning(f"Davies-Bouldin failed: {e}")
        metrics['davies_bouldin_index'] = np.nan
    
    # Calinski-Harabasz Index (0, ∞): Higher is better (> 30 is good)
    try:
        ch_index = calinski_harabasz_score(X, labels)
        metrics['calinski_harabasz_index'] = float(ch_index)
        logger.info(f"Calinski-Harabasz Index: {ch_index:.4f}")
    except Exception as e:
        logger.warning(f"Calinski-Harabasz failed: {e}")
        metrics['calinski_harabasz_index'] = np.nan
    
    return metrics


def evaluate_k_range(
    X: np.ndarray,
    linkage_matrix: np.ndarray,
    k_range: List[int]
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate clustering quality for range of k values.
    
    Args:
        X: Feature matrix
        linkage_matrix: Hierarchical clustering linkage matrix
        k_range: List of k values to evaluate
        
    Returns:
        Dictionary mapping k to metrics
    """
    from scipy.cluster.hierarchy import fcluster
    
    results = {}
    
    for k in k_range:
        try:
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            metrics = comprehensive_cluster_validation(X, labels, k)
            results[k] = metrics
            logger.info(f"k={k}: Silhouette={metrics.get('silhouette_score', np.nan):.4f}")
        except Exception as e:
            logger.error(f"Evaluation failed for k={k}: {e}")
            results[k] = {}
    
    return results