import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Automatically create the export directory
os.makedirs("reports", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ============================================================================
# 1. LOADING AND CLEANING THE PIMA DATASET
# ============================================================================
data_path = os.path.join("data", "pima_diabetes.csv")
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Load data
df = pd.read_csv(data_path, names=columns)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Correct aberrant 0 values (replaced by the median)
zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_invalid:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Z-Score Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ============================================================================
# 2. HIERARCHICAL CLUSTERING (WARD'S METHOD)
# ============================================================================
Z = linkage(X_scaled, method='ward', metric='euclidean')

# Evaluation of optimal k (searching k between 2 and 5)
best_k, best_score = 2, -1
for k in range(2, 6):
    labels = fcluster(Z, t=k, criterion='maxclust')
    score = silhouette_score(X_scaled, labels)
    if score > best_score:
        best_score, best_k = score, k

df['Cluster'] = fcluster(Z, t=best_k, criterion='maxclust')

# ============================================================================
# 3. GENERATION OF THE GLOBAL FIGURE (4 VISUALIZATIONS)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Chart 1: CAH Dendrogram
dendrogram(
    Z, 
    truncate_mode='lastp', 
    p=25, 
    color_threshold=Z[-best_k+1, 2], 
    above_threshold_color='gray',
    ax=axes[0, 0]
)
axes[0, 0].axhline(y=Z[-best_k+1, 2], color='r', linestyle='--', label=f'Optimal cut threshold (k={best_k})')
axes[0, 0].set_title("1. CAH Dendrogram (Truncated)", fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel("Linkage Distance")
axes[0, 0].legend(loc='upper right')

# Chart 2: PCA Projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1], 
    hue=df['Cluster'], palette='Set1', 
    style=df['Outcome'], markers=['o', 'X'],
    s=50, alpha=0.8, ax=axes[0, 1]
)
axes[0, 1].set_title(f"2. PCA Projection (Variance: {sum(var_exp)*100:.1f}%)", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
axes[0, 1].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")

# Chart 3: Cluster Profiles (Normalized) - OPTIMIZED FOR READABILITY
df_scaled_df = pd.DataFrame(X_scaled, columns=features)
df_scaled_df['Cluster'] = df['Cluster']
df_melted_scaled = df_scaled_df.melt(id_vars=['Cluster'], value_vars=features)

sns.barplot(data=df_melted_scaled, x='variable', y='value', hue='Cluster', palette='Set1', errorbar=None, ax=axes[1, 0])
axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[1, 0].set_title("3. Clinical Profiles (Z-Scores)", fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel("Deviation from Mean (Std Dev)")
axes[1, 0].tick_params(axis='x', rotation=45, labelsize=9)
plt.setp(axes[1, 0].get_xticklabels(), ha="right")

# Chart 4: Diabetes Distribution (Outcome) by Cluster
df_prop = df.groupby('Cluster')['Outcome'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
sns.barplot(data=df_prop, x='Cluster', y='Percentage', hue='Outcome', palette='Set2', ax=axes[1, 1])
axes[1, 1].set_title("4. Diabetes Prevalence (%)", fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel("Percentage (%)")
axes[1, 1].set_ylim(0, 100)
axes[1, 1].legend(title='Outcome', labels=['Healthy (0)', 'Diabetic (1)'])

plt.tight_layout()

# Save individual and global images
plt.savefig("reports/dashboard_complet.png", dpi=300)

# Save individual figures for reports
fig1, ax1 = plt.subplots(figsize=(8, 5))
dendrogram(Z, truncate_mode='lastp', p=25, color_threshold=Z[-best_k+1, 2], above_threshold_color='gray', ax=ax1)
ax1.set_title("CAH Dendrogram")
fig1.savefig("reports/1_dendrogramme.png", dpi=300)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='Set1', style=df['Outcome'], ax=ax2)
ax2.set_title("2D PCA Projection")
fig2.savefig("reports/2_projection_acp.png", dpi=300)
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(9, 5))
sns.barplot(data=df_melted_scaled, x='variable', y='value', hue='Cluster', palette='Set1', errorbar=None, ax=ax3)
ax3.tick_params(axis='x', rotation=45, labelsize=9)
plt.setp(ax3.get_xticklabels(), ha="right")
ax3.set_title("Clinical Profiles by Cluster")
fig3.savefig("reports/3_profils_clusters.png", dpi=300)
plt.close(fig3)

fig4, ax4 = plt.subplots(figsize=(7, 5))
sns.barplot(data=df_prop, x='Cluster', y='Percentage', hue='Outcome', palette='Set2', ax=ax4)
ax4.set_title("Diabetes Prevalence by Cluster (%)")
fig4.savefig("reports/4_repartition_diabete.png", dpi=300)
plt.close(fig4)

# Final display of the complete dashboard
plt.show()

# ============================================================================
# 4. RESULTS EXPORT
# ============================================================================
df.to_csv("data/pima_diabetes_segmented.csv", index=False, sep=";")
print("=" * 70)
print(f"✅ SUCCESSFUL TEST! Ideal partition: {best_k} clusters.")
print("✅ The 'reports/' directory has been updated with clear, readable plots.")
print("=" * 70)