# 🩸 Hierarchical Agglomerative Clustering for Diabetes Patient Segmentation

Educational unsupervised-learning project applying **Hierarchical Agglomerative Clustering (HAC)** with Ward linkage to the Pima Indians Diabetes dataset.

## 📋 Overview

The project explores whether patients can be grouped into similar feature profiles using:

- invalid-zero handling and median imputation
- feature standardization with Z-score normalization
- Ward hierarchical clustering with Euclidean distance
- internal validation with Silhouette, Davies-Bouldin, and Calinski-Harabasz indices
- PCA visualization and cluster profiles
- an interactive Streamlit dashboard

The `Outcome` column is **not used to build the clusters**. It is used only afterward to describe the observed outcome distribution within each cluster.

## 📦 Dataset

- **Source:** Pima Indians Diabetes Database
- **Samples:** 768
- **Clustering features:** 8
- **Outcome:** binary dataset label (0/1), excluded from clustering

## 🔧 Installation

```bash
git clone https://github.com/ghada-59/HAC_diabetes_Segmentation.git
cd HAC_diabetes_Segmentation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Run the main HAC analysis

```bash
python scripts/cah_diabete.py
```

This generates analysis figures under `reports/` and the segmented dataset under `data/processed/`.

### 2. Evaluate cluster counts

```bash
python validation_metrics.py
```

This computes Silhouette, Davies-Bouldin, and Calinski-Harabasz scores for `k=2..5` and reports the value of `k` selected by the highest Silhouette score.

### 3. Launch the dashboard

```bash
streamlit run scripts/app.py
```

The dashboard lets you change `k` from 2 to 5 and inspect:

- the HAC dendrogram;
- a 2D PCA projection;
- standardized cluster profiles;
- the observed Outcome distribution within clusters;
- the Silhouette score for the selected `k`.

### 4. Run tests

```bash
python -m pytest test_clustering_validation.py -v
```

## 📊 Validation

The project uses three **internal clustering metrics**:

| Metric | Interpretation |
|---|---|
| Silhouette | Higher values indicate better separation/cohesion |
| Davies-Bouldin | Lower values indicate better separation/cohesion |
| Calinski-Harabasz | Higher values indicate stronger between-cluster dispersion relative to within-cluster dispersion |

Silhouette is used as the primary criterion for selecting `k` in the automated analysis. The other metrics provide complementary information.

**Important:** these are mathematical clustering-quality measures, not clinical validation measures.

## ⚠️ Limitations

- This is an **exploratory unsupervised-learning project**, not a diagnostic or risk-prediction system.
- Results are specific to the Pima dataset and should not be generalized to other populations without additional validation.
- Median imputation and standardization are applied to the complete dataset because the task is exploratory clustering rather than supervised train/test prediction.
- Cluster IDs are arbitrary labels; a cluster number does not inherently represent higher or lower diabetes risk.
- `Outcome` is intentionally excluded from clustering, so observed differences in Outcome distribution are descriptive rather than evidence of causal relationships.
- Ward hierarchical clustering has quadratic memory/time characteristics and is intended here for a small educational dataset.

## 📁 Project Structure

```text
HAC_diabetes_Segmentation/
├── scripts/
│   ├── cah_diabete.py
│   └── app.py
├── validation_metrics.py
├── test_clustering_validation.py
├── data/
│   ├── pima_diabetes.csv
│   └── processed/
├── reports/
├── requirements.txt
├── README.md
└── LICENSE
```

## 📝 Project Type

**Academic / practice project.** The work demonstrates an end-to-end unsupervised-learning workflow and is not a clinically validated application.

## 📄 License

MIT License — see `LICENSE`.