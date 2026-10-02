"""
Clustering validation metrics for HAC analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "pima_diabetes.csv"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
COLUMNS = FEATURES + ["Outcome"]
INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


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
            logger.warning(
                "Skipping k=%s because fcluster produced %s clusters.",
                k,
                actual_k,
            )
            continue

        results[k] = comprehensive_cluster_validation(X, labels, k)

    if not results:
        raise ValueError("No valid clustering solutions were evaluated.")

    return results


def load_feature_matrix(data_path: Path = DATA_PATH) -> np.ndarray:
    """Load and standardize the eight clustering features."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path, header=None, names=COLUMNS)
    for column in INVALID_ZERO_COLUMNS:
        df[column] = df[column].replace(0, np.nan)
        df[column] = df[column].fillna(df[column].median())

    return StandardScaler().fit_transform(df[FEATURES])


def main() -> None:
    """Print internal validation metrics for k=2..5."""
    X = load_feature_matrix()
    linkage_matrix = linkage(X, method="ward", metric="euclidean")
    results = evaluate_k_range(X, linkage_matrix, list(range(2, 6)))

    print("k | Silhouette | Davies-Bouldin | Calinski-Harabasz")
    print("-" * 52)
    for k, metrics in results.items():
        print(
            f"{k} | {metrics['silhouette_score']:.3f}"
            f" | {metrics['davies_bouldin_index']:.3f}"
            f" | {metrics['calinski_harabasz_index']:.3f}"
        )

    best_k = max(results, key=lambda k: results[k]["silhouette_score"])
    print(f"\nSilhouette-selected k: {best_k}")


if __name__ == "__main__":
    main()
