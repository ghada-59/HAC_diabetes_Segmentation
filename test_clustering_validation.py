"""
Unit tests for clustering validation metrics.
"""

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from sklearn.datasets import make_blobs

from validation_metrics import comprehensive_cluster_validation, evaluate_k_range


@pytest.fixture
def synthetic_data():
    """Generate reproducible synthetic clustering data."""
    X, y = make_blobs(n_samples=300, centers=3, n_features=4, random_state=42)
    return X, y


def test_validation_metrics_perfect_clustering(synthetic_data):
    X, y = synthetic_data
    metrics = comprehensive_cluster_validation(X, y, k=3)

    assert set(metrics) == {
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_index",
    }
    assert metrics["silhouette_score"] > 0.3


def test_validation_metrics_poor_clustering(synthetic_data):
    X, _ = synthetic_data
    rng = np.random.default_rng(42)
    random_labels = rng.integers(0, 3, size=X.shape[0])

    # Random labels may occasionally omit a class; retry with a fixed
    # permutation-based construction to guarantee exactly three clusters.
    random_labels = np.tile(np.arange(3), X.shape[0] // 3 + 1)[: X.shape[0]]
    rng.shuffle(random_labels)

    metrics = comprehensive_cluster_validation(X, random_labels, k=3)
    assert metrics["silhouette_score"] < 0.3


def test_shape_mismatch_raises_error(synthetic_data):
    X, y = synthetic_data
    with pytest.raises(ValueError, match="same number of samples"):
        comprehensive_cluster_validation(X, y[:-10], k=3)


def test_k_mismatch_raises_error(synthetic_data):
    X, y = synthetic_data
    with pytest.raises(ValueError, match="Expected 5 clusters"):
        comprehensive_cluster_validation(X, y, k=5)


def test_invalid_dimensions_raise_error(synthetic_data):
    X, y = synthetic_data
    with pytest.raises(ValueError, match="2D feature matrix"):
        comprehensive_cluster_validation(X[0], y, k=3)


def test_evaluate_k_range(synthetic_data):
    X, _ = synthetic_data
    Z = linkage(X, method="ward")
    results = evaluate_k_range(X, Z, k_range=[2, 3, 4, 5])

    assert set(results) == {2, 3, 4, 5}
    assert all(
        set(metrics) == {
            "silhouette_score",
            "davies_bouldin_index",
            "calinski_harabasz_index",
        }
        for metrics in results.values()
    )


def test_invalid_k_range_raises_error(synthetic_data):
    X, _ = synthetic_data
    Z = linkage(X, method="ward")

    with pytest.raises(ValueError, match="No valid clustering solutions"):
        evaluate_k_range(X, Z, k_range=[0, 1]);
