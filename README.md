# 🧬 Diabetes Dataset Segmentation with Hierarchical Clustering

An academic data-science project exploring **Hierarchical Agglomerative Clustering (HAC)** on the Pima diabetes dataset.

## 🎯 Objective

The objective is to explore whether patients in the dataset can be grouped according to similarities in their measured characteristics.

The clustering is performed without using the diabetes outcome as an input feature.

## 🔬 Method

The workflow includes:

1. Cleaning selected zero values treated as missing observations.
2. Standardizing numerical variables with \`StandardScaler\`.
3. Applying Ward hierarchical clustering.
4. Comparing candidate cluster counts with the silhouette score.
5. Using PCA to visualize the resulting groups.
6. Describing cluster-level feature patterns and comparing them with the observed outcome distribution.

## 📊 Interpretation

The clusters describe patterns present in this particular dataset. They should be interpreted as **data-analysis profiles**, not as validated clinical risk groups or treatment recommendations.

## 🛠️ Technologies

Python · Pandas · NumPy · SciPy · scikit-learn · Matplotlib · Seaborn · Streamlit

## ⚠️ Limitations

The dataset is limited in size and population. Clustering results depend on preprocessing choices, variables and the selected number of clusters. The observed associations do not establish causality or clinical risk.

## 🚀 Run

\`\`\`bash
pip install -r requirements.txt
python cah_diabete.py
streamlit run app.py
\`\`\`
