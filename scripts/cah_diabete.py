from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
COLUMNS = FEATURES + ["Outcome"]
INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    """Load the headerless Pima dataset and replace invalid zero measurements."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path, header=None, names=COLUMNS)

    for column in INVALID_ZERO_COLUMNS:
        df[column] = df[column].replace(0, np.nan)
        df[column] = df[column].fillna(df[column].median())

    return df


def select_cluster_count(X_scaled: np.ndarray, linkage_matrix: np.ndarray) -> tuple[int, float]:
    """Select k in [2, 5] using the silhouette score."""
    best_k = None
    best_score = float("-inf")

    for k in range(2, 6):
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
        n_clusters = len(np.unique(labels))

        if n_clusters != k:
            continue

        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        raise RuntimeError("No valid cluster count was produced for k in [2, 5].")

    return best_k, best_score


def cluster_cut_height(linkage_matrix: np.ndarray, k: int) -> float:
    """Return a threshold between the merges that produce k and k-1 clusters."""
    if k < 2 or k > len(linkage_matrix):
        raise ValueError(f"Invalid cluster count: {k}")

    lower = linkage_matrix[-k, 2]
    upper = linkage_matrix[-k + 1, 2]
    return float((lower + upper) / 2)


def main() -> None:
    data_path = DATA_DIR / "pima_diabetes.csv"
    df = load_and_prepare_data(data_path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    linkage_matrix = linkage(X_scaled, method="ward", metric="euclidean")
    best_k, best_score = select_cluster_count(X_scaled, linkage_matrix)
    df["Cluster"] = fcluster(linkage_matrix, t=best_k, criterion="maxclust")

    cut_height = cluster_cut_height(linkage_matrix, best_k)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=25,
        color_threshold=cut_height,
        above_threshold_color="gray",
        ax=axes[0, 0],
    )
    axes[0, 0].axhline(
        y=cut_height,
        linestyle="--",
        label=f"Cut threshold (k={best_k})",
    )
    axes[0, 0].set_title("1. CAH Dendrogram (Truncated)")
    axes[0, 0].set_ylabel("Linkage Distance")
    axes[0, 0].legend(loc="upper right")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df["Cluster"],
        style=df["Outcome"],
        markers=["o", "X"],
        s=50,
        alpha=0.8,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title(f"2. PCA Projection (Variance: {sum(var_exp) * 100:.1f}%)")
    axes[0, 1].set_xlabel(f"PC1 ({var_exp[0] * 100:.1f}%)")
    axes[0, 1].set_ylabel(f"PC2 ({var_exp[1] * 100:.1f}%)")

    df_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
    df_scaled["Cluster"] = df["Cluster"]
    df_melted = df_scaled.melt(id_vars=["Cluster"], value_vars=FEATURES)

    sns.barplot(
        data=df_melted,
        x="variable",
        y="value",
        hue="Cluster",
        errorbar=None,
        ax=axes[1, 0],
    )
    axes[1, 0].axhline(0, linestyle="--", linewidth=0.8)
    axes[1, 0].set_title("3. Cluster Profiles (Z-Scores)")
    axes[1, 0].set_ylabel("Deviation from Mean (Std Dev)")
    axes[1, 0].tick_params(axis="x", rotation=45)
    plt.setp(axes[1, 0].get_xticklabels(), ha="right")

    df_prop = (
        df.groupby("Cluster")["Outcome"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("Percentage")
        .reset_index()
    )
    sns.barplot(
        data=df_prop,
        x="Cluster",
        y="Percentage",
        hue="Outcome",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("4. Observed Outcome Distribution by Cluster")
    axes[1, 1].set_ylabel("Percentage (%)")
    axes[1, 1].set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "dashboard_complet.png", dpi=300)
    plt.close(fig)

    individual_figures = [
        ("1_dendrogramme.png", "CAH Dendrogram", lambda ax: dendrogram(
            linkage_matrix, truncate_mode="lastp", p=25,
            color_threshold=cut_height, above_threshold_color="gray", ax=ax
        )),
        ("2_projection_acp.png", "2D PCA Projection", lambda ax: sns.scatterplot(
            x=X_pca[:, 0], y=X_pca[:, 1], hue=df["Cluster"], style=df["Outcome"], ax=ax
        )),
        ("3_profils_clusters.png", "Cluster Profiles", lambda ax: sns.barplot(
            data=df_melted, x="variable", y="value", hue="Cluster", errorbar=None, ax=ax
        )),
        ("4_repartition_diabete.png", "Outcome Distribution by Cluster", lambda ax: sns.barplot(
            data=df_prop, x="Cluster", y="Percentage", hue="Outcome", ax=ax
        )),
    ]

    for filename, title, plotter in individual_figures:
        figure, axis = plt.subplots(figsize=(8, 5))
        plotter(axis)
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(REPORTS_DIR / filename, dpi=300)
        plt.close(figure)

    output_path = PROCESSED_DIR / "pima_diabetes_segmented.csv"
    df.to_csv(output_path, index=False, sep=";")

    print("=" * 70)
    print(f"Successful clustering: {best_k} clusters.")
    print(f"Silhouette score: {best_score:.3f}")
    print(f"Processed dataset saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
