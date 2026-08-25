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

# Création automatique du dossier d'export
os.makedirs("reports", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ============================================================================
# 1. CHARGEMENT ET NETTOYAGE DU DATASET PIMA
# ============================================================================
data_path = os.path.join("data", "pima_diabetes.csv")
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Chargement des données
df = pd.read_csv(data_path, names=columns)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Correction des 0 aberrants (remplacés par la médiane)
zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_invalid:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Standardisation Z-Score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ============================================================================
# 2. ALGORITHME CAH (CRITÈRE DE WARD)
# ============================================================================
Z = linkage(X_scaled, method='ward', metric='euclidean')

# Évaluation du k optimal (recherche de k entre 2 et 5)
best_k, best_score = 2, -1
for k in range(2, 6):
    labels = fcluster(Z, t=k, criterion='maxclust')
    score = silhouette_score(X_scaled, labels)
    if score > best_score:
        best_score, best_k = score, k

df['Cluster'] = fcluster(Z, t=best_k, criterion='maxclust')

# ============================================================================
# 3. GÉNÉRATION DE LA FIGURE GLOBALE (4 VISUALISATIONS)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Graphique 1 : Dendrogramme CAH
dendrogram(
    Z, 
    truncate_mode='lastp', 
    p=25, 
    color_threshold=Z[-best_k+1, 2], 
    above_threshold_color='gray',
    ax=axes[0, 0]
)
axes[0, 0].axhline(y=Z[-best_k+1, 2], color='r', linestyle='--', label=f'Seuil coupe optimal (k={best_k})')
axes[0, 0].set_title("1. Dendrogramme CAH (Tronqué)", fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel("Inertie Inter-classe")
axes[0, 0].legend(loc='upper right')

# Graphique 2 : Projection ACP
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1], 
    hue=df['Cluster'], palette='Set1', 
    style=df['Outcome'], markers=['o', 'X'],
    s=50, alpha=0.8, ax=axes[0, 1]
)
axes[0, 1].set_title(f"2. Projection ACP (Variance : {sum(var_exp)*100:.1f}%)", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
axes[0, 1].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")

# Graphique 3 : Profils des Clusters (Normalisés) - CORRIGÉ POUR LISIBILITÉ
df_scaled_df = pd.DataFrame(X_scaled, columns=features)
df_scaled_df['Cluster'] = df['Cluster']
df_melted_scaled = df_scaled_df.melt(id_vars=['Cluster'], value_vars=features)

sns.barplot(data=df_melted_scaled, x='variable', y='value', hue='Cluster', palette='Set1', errorbar=None, ax=axes[1, 0])
axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[1, 0].set_title("3. Profils Cliniques (Z-Scores)", fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel("Écart à la moyenne (Std Dev)")
axes[1, 0].tick_params(axis='x', rotation=45, labelsize=9)
plt.setp(axes[1, 0].get_xticklabels(), ha="right")

# Graphique 4 : Distribution Diabète (Outcome) par Cluster
df_prop = df.groupby('Cluster')['Outcome'].value_counts(normalize=True).mul(100).rename('Pourcentage').reset_index()
sns.barplot(data=df_prop, x='Cluster', y='Pourcentage', hue='Outcome', palette='Set2', ax=axes[1, 1])
axes[1, 1].set_title("4. Prévalence du Diabète (%)", fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel("Pourcentage (%)")
axes[1, 1].set_ylim(0, 100)
axes[1, 1].legend(title='Outcome', labels=['Sain (0)', 'Diabétique (1)'])

plt.tight_layout()

# Sauvegarde des images individuelles et globales
plt.savefig("reports/dashboard_complet.png", dpi=300)

# Enregistrement des figures individuelles pour les rapports
fig1, ax1 = plt.subplots(figsize=(8, 5))
dendrogram(Z, truncate_mode='lastp', p=25, color_threshold=Z[-best_k+1, 2], above_threshold_color='gray', ax=ax1)
ax1.set_title("Dendrogramme CAH")
fig1.savefig("reports/1_dendrogramme.png", dpi=300)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='Set1', style=df['Outcome'], ax=ax2)
ax2.set_title("Projection ACP 2D")
fig2.savefig("reports/2_projection_acp.png", dpi=300)
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(9, 5))
sns.barplot(data=df_melted_scaled, x='variable', y='value', hue='Cluster', palette='Set1', errorbar=None, ax=ax3)
ax3.tick_params(axis='x', rotation=45, labelsize=9)
plt.setp(ax3.get_xticklabels(), ha="right")
ax3.set_title("Profils Cliniques par Cluster")
fig3.savefig("reports/3_profils_clusters.png", dpi=300)
plt.close(fig3)

fig4, ax4 = plt.subplots(figsize=(7, 5))
sns.barplot(data=df_prop, x='Cluster', y='Pourcentage', hue='Outcome', palette='Set2', ax=ax4)
ax4.set_title("Prévalence du Diabète par Cluster (%)")
fig4.savefig("reports/4_repartition_diabete.png", dpi=300)
plt.close(fig4)

# Affichage final du tableau de bord complet
plt.show()

# ============================================================================
# 4. EXPORT DES RÉSULTATS
# ============================================================================
df.to_csv("data/pima_diabetes_segmented.csv", index=False, sep=";")
print("=" * 70)
print(f"✅ TEST RÉUSSI ! Partition idéale : {best_k} clusters.")
print("✅ Le dossier 'reports/' a été mis à jour avec des graphiques parfaitement lisibles.")
print("=" * 70)