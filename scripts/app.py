import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Streamlit page configuration
st.set_page_config(page_title="HAC Dashboard - Diabetes", layout="wide")
sns.set_theme(style="whitegrid", palette="muted")

st.title("🩸 Intelligent Segmentation of Diabetic Patients (HAC)")
st.markdown("This interactive application applies **Hierarchical Agglomerative Clustering (HAC)** to the *Pima Indians Diabetes* medical dataset.")

# ============================================================================
# LOADING AND PREPROCESSING
# ============================================================================
@st.cache_data
def load_data():
    data_path = "data/pima_diabetes.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(data_path, names=columns)
    
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Correction of aberrant zeros
    zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_invalid:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return df, X_scaled, features

df, X_scaled, features = load_data()

# Interactive sidebar
st.sidebar.header("Model Parameters")
selected_k = st.sidebar.slider("Number of clusters (k):", min_value=2, max_value=5, value=4)

# HAC execution
Z = linkage(X_scaled, method='ward', metric='euclidean')
df['Cluster'] = fcluster(Z, t=selected_k, criterion='maxclust')

# ============================================================================
# DISPLAY CHARTS
# ============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. HAC Dendrogram")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    dendrogram(Z, truncate_mode='lastp', p=20, color_threshold=Z[-selected_k+1, 2], above_threshold_color='gray', ax=ax1)
    ax1.axhline(y=Z[-selected_k+1, 2], color='r', linestyle='--', label=f'Threshold (k={selected_k})')
    ax1.set_ylabel("Inter-class Inertia")
    ax1.legend(loc='upper right')
    st.pyplot(fig1)

with col2:
    st.subheader("2. PCA Projection (2D)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='Set1', style=df['Outcome'], markers=['o', 'X'], alpha=0.8, ax=ax2)
    ax2.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Clinical Profiles (Z-Scores)")
    df_scaled_df = pd.DataFrame(X_scaled, columns=features)
    df_scaled_df['Cluster'] = df['Cluster']
    df_melted_scaled = df_scaled_df.melt(id_vars=['Cluster'], value_vars=features)
    
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_melted_scaled, x='variable', y='value', hue='Cluster', palette='Set1', errorbar=None, ax=ax3)
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax3.tick_params(axis='x', rotation=45, labelsize=9)
    plt.setp(ax3.get_xticklabels(), ha="right")
    st.pyplot(fig3)

with col4:
    st.subheader("4. Actual Diabetes Prevalence (%)")
    df_prop = df.groupby('Cluster')['Outcome'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
    
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_prop, x='Cluster', y='Percentage', hue='Outcome', palette='Set2', ax=ax4)
    ax4.set_ylim(0, 100)
    ax4.legend(title='Outcome', labels=['Healthy (0)', 'Diabetic (1)'])
    st.pyplot(fig4)

st.success(f"✅ Model successfully executed for {selected_k} clusters!")