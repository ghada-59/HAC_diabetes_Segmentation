# 🩸 Classification Ascendante Hiérarchique (CAH) - Segmentation des Patients (Dataset Pima)

Ce projet applique des techniques d'apprentissage non supervisé (Clustering) pour identifier des **profils types (phénotypes) de patients atteints ou à risque de diabète de type 2** à partir du jeu de données réel **Pima Indians Diabetes** (`pima_diabetes.csv`). 

L'analyse repose sur la **Classification Ascendante Hiérarchique (CAH)** avec le **critère de Ward**, permettant de regrouper les 768 patients selon la similarité de leurs constantes biologiques, sans utiliser la variable cible (`Outcome`) lors de la phase d'apprentissage.

---

## 🎯 Objectifs et Contexte Médical

Dans la prise en charge du diabète, deux patients diagnostiqués positifs peuvent présenter des réalités physiologiques très différentes (ex. une personne jeune avec un taux d'insuline élevé vs une personne âgée avec une forte pression artérielle et un IMC élevé).

Les objectifs stratégiques du projet sont :
1. **Identifier des sous-groupes homogènes de patients** basés uniquement sur leurs caractéristiques cliniques (`Glucose`, `BMI`, `Insulin`, `Age`, `BloodPressure`, etc.).
2. **Nettoyer et préparer les données médicales** en traitant les valeurs aberrantes (ex. glycémie ou tension égale à zéro).
3. **Évaluer et interpréter cliniquement les clusters obtenus** en les croisant a posteriori avec la prévalence réelle du diabète (`Outcome`).
4. **Déployer une interface décisionnelle interactive** via Streamlit pour permettre aux praticiens de faire varier le nombre de groupes ($k$).

---

## 🧠 Choix Méthodologiques et Techniques

### 1. Pourquoi l'Apprentissage Non Supervisé (Clustering) ?
L'objectif n'est pas de construire un simple prédicteur "malade / non malade", mais de découvrir la structure sous-jacente de la population. L'algorithme travaille en "aveugle" par rapport à la colonne `Outcome` pour isoler des profils biologiques réels.

### 2. Pourquoi la CAH (Classification Ascendante Hiérarchique) ?
* **Visualisation par Dendrogramme :** Contrairement à K-Means, la CAH produit un arbre hiérarchique qui montre exactement comment les patients et les groupes se regroupent étape par étape.
* **Choix flexible du nombre de clusters :** Le nombre de groupes ($k$) peut être défini ou ajusté *a posteriori* par le Data Scientist ou le médecin en observant les hauteurs de fusion.
* **Méthode Déterministe :** Elle garantit des résultats stables et reproductibles à chaque exécution (pas d'initialisation aléatoire).

### 3. Pourquoi le Critère de Ward ?
Le critère de Ward vise à **minimiser l'inertie intra-classe** (variance à l'intérieur de chaque groupe) et à **maximiser l'inertie inter-classes** (distance entre les groupes). Il permet d'obtenir des clusters très compacts, homogènes et équilibrés, ce qui est idéal pour la segmentation médicale.

### 4. Pourquoi la Standardisation Z-Score ?
Les variables du dataset possèdent des échelles très différentes (l'Âge varie de 21 à 81 ans, le Glucose de 44 à 199 mg/dL, l'Insuline de 14 à 846 $\mu\text{U/mL}$). Sans standardisation, la variable avec les plus grandes valeurs dominerait artificiellement le calcul des distances euclidiennes. 

$$\text{Z-score} = \frac{x - \mu}{\sigma}$$

---

## 🔬 Pipeline de Traitement des Données


### 1. Nettoyage des zéros aberrants
Dans les données médicales, des valeurs égales à `0` pour le `Glucose`, la `BloodPressure`, le `SkinThickness`, l'`Insulin` ou le `BMI` sont biologiquement impossibles. Ces valeurs ont été identifiées comme des données manquantes et remplacées par la **médiane** de leurs colonnes respectives.

### 2. Standardisation et Matrice de Liaison
Toutes les caractéristiques physiologiques sont normalisées via `StandardScaler`. La matrice de liaison est calculée à l'aide de la méthode de Ward (`linkage(X_scaled, method='ward')`).

### 3. Recherche du $k$ Optimal & Score de Silhouette
Le script calcule automatiquement le **Score de Silhouette** pour différents découpages ($k \in [2, 5]$) afin de déterminer le nombre de clusters qui sépare le mieux les données.

### 4. Visualisations et Diagnostics Générés
* **Dendrogramme Tronqué :** Visualisation de la hiérarchie d'agrégation et du seuil de coupe.
* **Projection ACP (2D) :** Analyse en Composantes Principales permettant de projeter les individus sur 2 axes pour visualiser la séparation spatiale des clusters.
* **Profils Cliniques (Z-Scores) :** Graphique en barres montrant si un cluster est au-dessus ou en dessous de la moyenne pour chaque paramètre médical.
* **Prévalence Réelle du Diabète (%) :** Pourcentage de cas positifs (`Outcome = 1`) contenus dans chaque cluster.

---


---

## 💡 Interprétations Avancées & Cliniques

### 1. Interprétation Biologique des Axes de l'ACP
Sur le graphique 2D de l'ACP, la variance s'organise selon deux dimensions majeures :
* **Axe 1 (PC1 - Horizontal) :** Représente l'**intensité métabolique globale** (synergie de l'âge, du glucose et de la pression artérielle). Plus un patient se décale vers la droite, plus son profil glycémique et cardiovasculaire est altéré.
* **Axe 2 (PC2 - Vertical) :** Représente la **morphologie corporelle** (combinaison de l'IMC et de l'épaisseur de la peau). Il sépare les profils à forte surcharge pondérale des profils plus minces.

### 2. Validation de la Structure par le Score de Silhouette
En l'absence de classes supervisées lors de l'apprentissage, le score de Silhouette valide la cohésion géométrique des groupes. Les pics de score obtenus pour $k=3$ ou $k=4$ confirment que la population se fragmente naturellement selon des frontières biologiques nettes.

### 3. Dissociation : Risque Génétique vs Risque Acquis
L'analyse de la variable `DiabetesPedigreeFunction` (facteur d'hérédité) combinée aux clusters révèle deux dynamiques distinctes :
* Des groupes où le déséquilibre glycémique et l'IMC élevé s'expliquent principalement par des facteurs environnementaux et le mode de vie.
* Des profils où la charge héréditaire est très forte, impactant des patients parfois plus jeunes mais présentant une prédisposition marquée.

### 4. Recommandations Cliniques et Stratification des Soins

| Cluster Identifié | Profil Physiologique | Action Médicale Recommandée |
| :--- | :--- | :--- |
| **Profil 1 : Sain / Jeune** | Constantes normales, IMC et glucose modérés | Suivi préventif de routine, conseils en hygiène de vie. |
| **Profil 2 : Risque Métabolique** | Surcharge pondérale (IMC élevé), insuline forte, glucose limite | Programme personnalisé d'activité physique et suivi nutritionnel d'urgence pour freiner l'évolution vers le diabète. |
| **Profil 3 : Diabète Avancé / Âger** | Glycémie très élevée, hypertension, âge avancé | Prise en charge pharmacologique renforcée, dépistage précoce des complications rénales et vasculaires. |

---

## 🛠️ Choix des Bibliothèques et Stack Technique

| Bibliothèque | Usage principal | Justification technique |
| :--- | :--- | :--- |
| **Python 3** | Langage principal | Standard incontournable pour la Data Science et la santé. |
| **Pandas & NumPy** | Manipulation de données | Chargement du CSV, nettoyage des zéros et calculs matriciels. |
| **SciPy (`cluster.hierarchy`)** | Algorithme CAH | Fonctions `linkage`, `dendrogram` et `fcluster` pour l'arbre hiérarchique. |
| **Scikit-Learn** | Preprocessing & Métriques | `StandardScaler` pour la normalisation, `PCA` pour la réduction de dimension et `silhouette_score`. |
| **Matplotlib & Seaborn** | Visualisation statique | Génération des graphiques et sauvegarde des rapports d'analyse. |
| **Streamlit** | Interface Web | Déploiement d'une application interactive permettant de tester dynamiquement plusieurs valeurs de $k$. |

---

## 🚀 Structure du Projet & Guide d'Exécution

### Structure des dossiers
```text
├── data/
│   ├── pima_diabetes.csv             # Dataset médical initial (768 patients)
│   └── pima_diabetes_segmented.csv   # Dataset final exporté avec la colonne 'Cluster'
├── reports/                          # Graphiques sauvegardés automatiquement
│   ├── dashboard_complet.png
│   ├── 1_dendrogramme.png
│   ├── 2_projection_acp.png
│   ├── 3_profils_clusters.png
│   └── 4_repartition_diabete.png
├── cah_diabete.py                    # Script d'analyse batch & génération de rapports
├── app.py                            # Application Web interactive (Streamlit)
└── README.md                         # Documentation du projet

```

### pour lancer le projet

1. **Cloner le dépôt Git :**
```bash
git clone
cd cah-diabete-segmentation

```

2. **Exécuter le script d'analyse basique (génère les rapports et le CSV segmenté) :**
```bash
python cah_diabete.py

```

3. **Lancer le Dashboard Web Interactif Streamlit :**
```bash
streamlit run app.py

```


---

## 📊 Synthèse des Profils Cliniques Obtenus

Grâce à la segmentation par CAH sur le dataset , les groupes formés mettent en évidence des profils types distincts :

* **Cluster à Bas Risque (Profil Sain / Jeune) :** Moyenne d'âge plus basse, IMC modéré, glycémie et insuline normales. La prévalence du diabète y est minimale.
* **Cluster à Risque Métabolique (Surpoids / Prédiabète) :** Patients caractérisés par un IMC très élevé et une insuline forte, nécessitant une prévention axée sur le poids.
* **Cluster à Haut Risque (Diabète Sévère / Âgé) :** Patients plus âgés présentant une glycémie très élevée et une pression artérielle forte. La prévalence réelle du diabète y est la plus forte.

```

```