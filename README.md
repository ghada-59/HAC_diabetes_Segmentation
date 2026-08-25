
# 🩸 Hierarchical Agglomerative Clustering (HAC) - Patient Segmentation (Pima Dataset)

This project applies unsupervised learning techniques (Clustering) to identify **typical patient profiles (phenotypes) who have or are at risk of Type 2 diabetes** using the real-world **Pima Indians Diabetes** dataset (`pima_diabetes.csv`).

The analysis is based on **Hierarchical Agglomerative Clustering (HAC)** with **Ward's linkage criterion**, making it possible to group the 768 patients together based on the similarity of their biological constants, without using the target variable (`Outcome`) during the training phase.

---

## 🎯 Objectives and Medical Context

In diabetes management, two patients diagnosed as positive can present very different physiological realities (e.g., a young person with high insulin levels vs. an older person with high blood pressure and a high BMI).

The strategic objectives of the project are:
1. **Identify homogeneous sub-groups of patients** based solely on their clinical characteristics (`Glucose`, `BMI`, `Insulin`, `Age`, `BloodPressure`, etc.).
2. **Clean and prepare medical data** by handling outlier values (e.g., blood glucose or blood pressure equal to zero).
3. **Evaluate and clinically interpret the resulting clusters** by cross-referencing them *a posteriori* with the actual prevalence of diabetes (`Outcome`).
4. **Deploy an interactive decision-making interface** via Streamlit to allow practitioners to vary the number of clusters ($k$).

---

## 🧠 Methodological and Technical Choices

### 1. Why Unsupervised Learning (Clustering)?
The goal is not to build a simple "sick / healthy" predictor, but to discover the underlying structure of the population. The algorithm works "blindly" with respect to the `Outcome` column to isolate real biological profiles.

### 2. Why HAC (Hierarchical Agglomerative Clustering)?
* **Dendrogram Visualization:** Unlike K-Means, HAC produces a hierarchical tree that shows exactly how patients and groups cluster step by step.
* **Flexible Choice of Cluster Count:** The number of groups ($k$) can be defined or adjusted *a posteriori* by the Data Scientist or physician by observing the fusion heights.
* **Deterministic Method:** It guarantees stable and reproducible results on every run (no random initialization).

### 3. Why Ward's Criterion?
Ward's criterion aims to **minimize intra-class inertia** (variance within each group) and **maximize inter-class inertia** (distance between groups). It produces very compact, homogeneous, and balanced clusters, which is ideal for medical segmentation.

### 4. Why Z-Score Standardization?
The variables in the dataset have very different scales (Age ranges from 21 to 81 years, Glucose from 44 to 199 mg/dL, Insulin from 14 to 846 $\mu\text{U/mL}$). Without standardization, the variable with the largest values would artificially dominate the Euclidean distance calculations. 

$$\text{Z-score} = \frac{x - \mu}{\sigma}$$

---

## 🔬 Data Processing Pipeline

### 1. Cleaning Aberrant Zeros
In medical data, values equal to `0` for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, or `BMI` are biologically impossible. These values were identified as missing data and replaced by the **median** of their respective columns.

### 2. Standardization and Linkage Matrix
All physiological characteristics are normalized via `StandardScaler`. The linkage matrix is computed using Ward's method (`linkage(X_scaled, method='ward')`).

### 3. Optimal $k$ Search & Silhouette Score
The script automatically calculates the **Silhouette Score** for different cuts ($k \in [2, 5]$) to determine the number of clusters that best separates the data.

### 4. Generated Visualizations and Diagnostics
* **Truncated Dendrogram:** Visualization of the aggregation hierarchy and the cut-off threshold.
* **PCA Projection (2D):** Principal Component Analysis making it possible to project individuals onto 2 axes to visualize the spatial separation of clusters.
* **Clinical Profiles (Z-Scores):** Bar chart showing whether a cluster is above or below average for each medical parameter.
* **Actual Diabetes Prevalence (%):** Percentage of positive cases (`Outcome = 1`) contained within each cluster.

---

## 💡 Advanced & Clinical Interpretations

### 1. Biological Interpretation of PCA Axes
On the 2D PCA plot, variance is organized along two major dimensions:
* **Axis 1 (PC1 - Horizontal):** Represents **global metabolic intensity** (synergy of age, glucose, and blood pressure). The further a patient shifts to the right, the more altered their glycemic and cardiovascular profile is.
* **Axis 2 (PC2 - Vertical):** Represents **body morphology** (combination of BMI and skin thickness). It separates profiles with high overweight burden from leaner profiles.

### 2. Validation of Structure via Silhouette Score
In the absence of supervised classes during training, the Silhouette score validates the geometric cohesion of the groups. The score peaks obtained for $k=3$ or $k=4$ confirm that the population naturally fragments along clear biological boundaries.

### 3. Dissociation: Genetic Risk vs. Acquired Risk
The analysis of the `DiabetesPedigreeFunction` variable (heredity factor) combined with the clusters reveals two distinct dynamics:
* Groups where glycemic imbalance and high BMI are explained mainly by environmental factors and lifestyle.
* Profiles where the hereditary load is very strong, impacting sometimes younger patients but presenting a marked predisposition.


### 4. Clinical Recommendations and Care Stratification

| Identified Cluster | Physiological Profile | Recommended Medical Action |
| :--- | :--- | :--- |
| **Profile 1: Healthy / Young** | Normal constants, moderate BMI and glucose | Routine preventive follow-up, lifestyle advice. |
| **Profile 2: Metabolic Risk** | Overweight (high BMI), high insulin, borderline glucose | Personalized physical activity program and emergency nutritional follow-up to curb the progression toward diabetes. |
| **Profile 3: Advanced / Older Diabetes** | Very high blood glucose, hypertension, advanced age | Enhanced pharmacological management, early screening for renal and vascular complications. |

---

## 🛠️ Library Choices and Technical Stack

| Library | Main Usage | Technical Justification |
| :--- | :--- | :--- |
| **Python 3** | Main language | Essential standard for Data Science and healthcare. |
| **Pandas & NumPy** | Data manipulation | CSV loading, zero cleaning, and matrix calculations. |
| **SciPy (`cluster.hierarchy`)** | HAC algorithm | `linkage`, `dendrogram`, and `fcluster` functions for the hierarchical tree. |
| **Scikit-Learn** | Preprocessing & Metrics | `StandardScaler` for normalization, `PCA` for dimension reduction, and `silhouette_score`. |
| **Matplotlib & Seaborn** | Static visualization | Generating charts and saving analysis reports. |
| **Streamlit** | Web Interface | Deploying an interactive app allowing dynamic testing of multiple values of $k$. |

---

## 🚀 Project Structure & Execution Guide

### Folder Structure
```text
├── data/
│   ├── pima_diabetes.csv             # Initial medical dataset (768 patients)
│   └── pima_diabetes_segmented.csv   # Final dataset exported with the 'Cluster' column
├── reports/                          # Automatically saved graphs
│   ├── dashboard_complet.png
│   ├── 1_dendrogramme.png
│   ├── 2_projection_acp.png
│   ├── 3_profils_clusters.png
│   └── 4_repartition_diabete.png
├── cah_diabete.py                    # Batch analysis script & report generation
├── app.py                            # Interactive Web Application (Streamlit)
└── README.md                         # Project documentation

```

### To Run the Project

1. **Clone the Git repository:**

```bash
git clone
cd cah-diabete-segmentation

```

2. **Run the basic analysis script (generates reports and segmented CSV):**

```bash
python cah_diabete.py

```

3. **Launch the Streamlit Interactive Web Dashboard:**

```bash
streamlit run app.py

```

---

## 📊 Summary of Obtained Clinical Profiles

Thanks to the HAC segmentation on the dataset, the formed groups highlight distinct typical profiles:

* **Low-Risk Cluster (Healthy / Young Profile):** Lower average age, moderate BMI, and normal baseline glucose levels.
* **Metabolic Risk Cluster:** High overweight burden (elevated BMI), high insulin resistance indicators, and borderline glycemic values.
* **Advanced / Older Diabetes Cluster:** High blood pressure, significantly advanced age, and critical blood glucose levels requiring intensive care.
