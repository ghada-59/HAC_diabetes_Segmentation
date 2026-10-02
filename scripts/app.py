from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "pima_diabetes.csv"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
COLUMNS = FEATURES + ["Outcome"]
INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

st.set_page_config(page_title="HAC Dashboard - Diabetes", layout="wide")
st.title("🩸 Diabetes Patient Segmentation — HAC")
st.markdown(
    "Interactive educational exploration of **Hierarchical Agglomerative Clustering (HAC)** "
    "on the *Pima Indians Diabetes* dataset. The clustering does not use the Outcome column."
)


@st.cache_data
def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

    for column in INVALID_ZERO_COLUMNS:
        df[column] = df[column].replace(0, np.nan)
        df[column] = df[column].fillna(df[column].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    return df, X_scaled


def get_cut_height(linkage_matrix, k):
    if k < 2 or k > len(linkage_matrix):
        raise ValueError(f"Invalid cluster count: {k}")
    return float((linkage_matrix[-k, 2] + linkage_matrix[-k + 1, 2]) / 2)


df, X_scaled = load_data()
Z = linkage(X_scaled, method="ward", metric="euclidean")

st.sidebar.header("Clustering Parameters")
selected_k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=5, value=4)

cluster_labels = fcluster(Z, t=selected_k, criterion="maxclust")
actual_k = len(np.unique(cluster_labels))
if actual_k != selected_k:
    st.error(f"The requested k={selected_k} produced {actual_k} clusters. Choose another value.")
    st.stop()

df["Cluster"] = cluster_labels
silhouette = silhouette_score(X_scaled, cluster_labels)
st.sidebar.metric("Silhouette score", f"{silhouette:.3f}")

cut_height = get_cut_height(Z, selected_k)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. HAC Dendrogram")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    dendrogram(
        Z, truncate_mode="lastp", p=20,
        color_threshold=cut_height, above_threshold_color="gray", ax=ax1,
    )
    ax1.axhline(y=cut_height, linestyle="--", label=f"Cut threshold (k={selected_k})")
    ax1.set_ylabel("Linkage Distance")
    ax1.legend(loc="upper right")
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.subheader("2. PCA Projection (2D)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    variance = pca.explained_variance_ratio_

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=df["Cluster"], style=df["Outcome"],
        markers=["o", "X"], alpha=0.8, ax=ax2,
    )
    ax2.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
    st.pyplot(fig2)
    plt.close(fig2)

col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Cluster Profiles (Z-Scores)")
    scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)
    scaled_df["Cluster"] = df["Cluster"]
    melted = scaled_df.melt(id_vars=["Cluster"], value_vars=FEATURES)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=melted, x="variable", y="value",
        hue="Cluster", errorbar=None, ax=ax3,
    )
    ax3.axhline(0, linestyle="--", linewidth=0.8)
    ax3.set_ylabel("Deviation from Mean (Std Dev)")
    ax3.tick_params(axis="x", rotation=45)
    plt.setp(ax3.get_xticklabels(), ha="right")
    st.pyplot(fig3)
    plt.close(fig3)

with col4:
    st.subheader("4. Observed Outcome Distribution by Cluster")
    proportions = (
        df.groupby("Cluster")["Outcome"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("Percentage")
        .reset_index()
    )

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=proportions, x="Cluster", y="Percentage",
        hue="Outcome", ax=ax4,
    )
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("Percentage (%)")
    st.pyplot(fig4)
    plt.close(fig4)

st.success(f"Clustering executed with k={selected_k}.")