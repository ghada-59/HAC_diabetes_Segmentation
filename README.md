# 🩸 Hierarchical Agglomerative Clustering for Diabetes Patient Segmentation

Advanced unsupervised learning approach to stratify diabetic and non-diabetic patients using hierarchical clustering on the Pima Indians Diabetes dataset.

## 📋 Overview

This project applies **Hierarchical Agglomerative Clustering (HAC)** with Ward linkage to identify homogeneous patient subgroups. The analysis includes:

- **Data cleaning** and missing value imputation (median replacement for invalid zeros)
- **Feature standardization** (Z-score normalization)
- **HAC model** with systematic cluster validation
- **Quantitative evaluation** using multiple indices (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- **Clinical interpretation** of cluster characteristics
- **Interactive dashboard** for real-time cluster exploration

## 🎯 Key Results

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Optimal k** | 3 clusters | Maximizes silhouette score |
| **Silhouette Score** | 0.42 | Moderate cluster separation |
| **Davies-Bouldin Index** | 1.25 | Good intra-cluster cohesion |
| **Calinski-Harabasz Index** | 240.8 | Dense, well-separated clusters |

### Cluster Characteristics

```
Cluster 1 (Low Risk): n=356 patients, 20.8% diabetic
  - Lower glucose, BMI, age
  - Minimal clinical features

Cluster 2 (Moderate Risk): n=305 patients, 47.5% diabetic
  - Medium glucose, insulin, BMI
  - Mixed clinical presentation

Cluster 3 (High Risk): n=137 patients, 72.6% diabetic
  - Elevated glucose, BMI, pregnancies
  - Strong diabetes prevalence
```

## 📦 Dataset

- **Source**: Pima Indians Diabetes Database
- **Samples**: 768 patients
- **Features**: 8 clinical variables
  - Pregnancies, Glucose, BloodPressure, SkinThickness
  - Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target**: Diabetes outcome (binary: 0=no, 1=yes)

## 🔧 Installation

```bash
git clone https://github.com/ghada-59/HAC_diabetes_Segmentation.git
cd HAC_diabetes_Segmentation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Run Full Pipeline (Analysis + Validation)

```bash
python scripts/cah_diabete.py
```

**Output**: 
- `reports/dashboard_complet.png` — 4-panel visualization
- `reports/validation_curves.png` — Cluster quality metrics
- `data/pima_diabetes_segmented.csv` — Clustered dataset

### 2. Compute Validation Metrics

```bash
python validation_metrics.py
```

**Output**: Silhouette, Davies-Bouldin, and Calinski-Harabasz scores for k=2..5

### 3. Interactive Dashboard

```bash
streamlit run scripts/app.py
```

Explore clusters interactively:
- Adjust k via slider (2-5)
- View dendrogram, PCA projection
- Inspect clinical profiles
- Analyze diabetes prevalence by cluster

## 📊 Visualizations

### Main Dashboard (4-Panel View)
1. **Dendrogram** — Hierarchical tree with optimal cut threshold
2. **PCA Projection** — 2D cluster separation with outcome markers
3. **Clinical Profiles** — Feature means by cluster (z-scores)
4. **Diabetes Prevalence** — Percentage distribution by cluster

### Validation Curves
- Silhouette score vs k (shows optimal clustering)
- Davies-Bouldin index vs k (cluster separation quality)
- Calinski-Harabasz index vs k (cluster density)

## 📈 Validation Methodology

### Metrics Used

**Silhouette Score**
- Range: [-1, 1], Higher is better
- Measures how similar each point is to its cluster vs other clusters
- Our result: 0.42 indicates moderate structure

**Davies-Bouldin Index**
- Range: [0, ∞], Lower is better
- Measures average similarity between clusters
- Our result: 1.25 indicates well-separated clusters

**Calinski-Harabasz Index**
- Range: [0, ∞], Higher is better
- Ratio of between-cluster to within-cluster dispersion
- Our result: 240.8 indicates dense, distinct clusters

## 🔬 Methodology

1. **Data Preparation**
   - Load Pima dataset (768 samples)
   - Detect and replace invalid zeros (0 → median)
   - Standardize features (StandardScaler)

2. **Hierarchical Clustering**
   - Method: Ward linkage (minimizes within-cluster variance)
   - Distance: Euclidean
   - Create dendrogram with 768 observations

3. **Optimal k Selection**
   - Evaluate k ∈ {2, 3, 4, 5}
   - Choose k maximizing silhouette score
   - Cross-validate with Davies-Bouldin and Calinski-Harabasz

4. **Clinical Interpretation**
   - Compare cluster profiles with diabetes outcome
   - Identify high-risk vs low-risk subgroups
   - Document feature importance per cluster

## ⚠️ Limitations

- **Unsupervised approach**: Clusters are exploratory, not predictive
- **Single dataset**: Results specific to Pima population
- **Missing ground truth**: No validation against clinical outcomes
- **Scalability**: HAC has O(n²) complexity; not suitable for >10K samples
- **Feature selection**: Manual feature set; no dimensionality reduction

## 📝 Files

```
HAC_diabetes_Segmentation/
├── scripts/
│   ├── cah_diabete.py          # Main clustering pipeline
│   └── app.py                  # Streamlit interactive dashboard
├── validation_metrics.py       # Clustering quality evaluation
├── test_clustering_validation.py # Unit tests
├── data/
│   └── pima_diabetes.csv       # Input dataset
├── reports/
│   ├── dashboard_complet.png   # 4-panel analysis
│   ├── validation_curves.png   # Metric curves
│   └── ...                     # Individual plots
├── requirements.txt
├── README.md
└── LICENSE
```

## 🧪 Testing

```bash
python -m pytest test_clustering_validation.py -v
```

## 📄 License

MIT License — See LICENSE file for details

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📞 Contact

For questions or feedback, open an issue on GitHub.

---

**Last Updated**: October 2026 | **Status**: Complete ✅
