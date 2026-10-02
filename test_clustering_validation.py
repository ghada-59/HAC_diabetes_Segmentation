"""
Unit tests for clustering validation metrics.
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from validation_metrics import comprehensive_cluster_validation, evaluate_k_range


class TestClusteringValidation:
    """Test cluster validation metrics."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic clustering data."""
        X, y = make_blobs(n_samples=300, centers=3, n_features=4, random_state=42)
        return X, y
    
    def test_validation_metrics_perfect_clustering(self, synthetic_data):
        """Test metrics on perfect clustering."""
        X, y = synthetic_data
        
        metrics = comprehensive_cluster_validation(X, y, k=3)
        
        assert 'silhouette_score' in metrics
        assert 'davies_bouldin_index' in metrics
        assert 'calinski_harabasz_index' in metrics
        
        # Good clustering should have high silhouette
        assert metrics['silhouette_score'] > 0.3
    
    def test_validation_metrics_poor_clustering(self, synthetic_data):
        """Test metrics on poor (random) clustering."""
        X, y = synthetic_data
        
        # Random labels
        rng = np.random.default_rng(42)
        random_labels = rng.integers(0, 3, size=X.shape[0])
        
        metrics = comprehensive_cluster_validation(X, random_labels, k=3)
        
        # Random clustering should have lower metrics
        assert metrics['silhouette_score'] < 0.3
    
    def test_shape_mismatch_raises_error(self, synthetic_data):
        """Test that shape mismatch raises error."""
        X, y = synthetic_data
        
        with pytest.raises(ValueError):
            comprehensive_cluster_validation(X, y[:-10], k=3)
    
    def test_k_mismatch_raises_error(self, synthetic_data):
        """Test that k mismatch raises error."""
        X, y = synthetic_data
        
        with pytest.raises(ValueError):
            comprehensive_cluster_validation(X, y, k=5)  # y has 3 clusters
    
    def test_evaluate_k_range(self, synthetic_data):
        """Test evaluation across k range."""
        X, y = synthetic_data
        
        Z = linkage(X, method='ward')
        results = evaluate_k_range(X, Z, k_range=[2, 3, 4, 5])
        
        assert len(results) == 4
        assert all(k in results for k in [2, 3, 4, 5])
        assert 'silhouette_score' in results[3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])