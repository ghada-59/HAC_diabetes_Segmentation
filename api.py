"""Local API for hierarchical clustering exploration on the Pima dataset."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "pima_diabetes.csv"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
COLUMNS = FEATURES + ["Outcome"]
INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
app = FastAPI(title="Data Cartography API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _load():
    if not DATA_PATH.is_file():
        raise FileNotFoundError("Dataset not found at data/pima_diabetes.csv.")
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    for col in INVALID_ZERO_COLUMNS:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    scaled = StandardScaler().fit_transform(df[FEATURES])
    tree = linkage(scaled, method="ward", metric="euclidean")
    projection = PCA(n_components=2).fit_transform(scaled)
    return df, scaled, tree, projection


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Data Cartography API"}


@app.get("/api/analysis")
def analysis(k: int = Query(4, ge=2, le=5)):
    try:
        df, scaled, tree, projection = _load()
        labels = fcluster(tree, t=k, criterion="maxclust")
        unique = np.unique(labels)
        if len(unique) < 2:
            raise ValueError("The selected cluster count did not produce multiple groups.")
        silhouette = float(silhouette_score(scaled, labels))
        explained = PCA(n_components=2).fit(scaled).explained_variance_ratio_
        points = [
            {"x": float(projection[i, 0]), "y": float(projection[i, 1]),
             "cluster": int(labels[i]), "outcome": int(df.iloc[i]["Outcome"])}
            for i in range(len(df))
        ]
        profiles = []
        for cluster in sorted(unique):
            indices = np.where(labels == cluster)[0]
            profiles.append({
                "cluster": int(cluster),
                "size": int(len(indices)),
                "outcome_0": int(np.sum(df.iloc[indices]["Outcome"].to_numpy() == 0)),
                "outcome_1": int(np.sum(df.iloc[indices]["Outcome"].to_numpy() == 1)),
                "features": {
                    feature: float(np.mean(scaled[indices, j]))
                    for j, feature in enumerate(FEATURES)
                },
            })
        return {
            "dataset_rows": int(len(df)),
            "feature_count": len(FEATURES),
            "k_requested": int(k),
            "k_actual": int(len(unique)),
            "silhouette": silhouette,
            "pca_variance": [float(explained[0]), float(explained[1])],
            "points": points,
            "profiles": profiles,
            "features": FEATURES,
            "method": "Ward linkage with Euclidean distance after standard scaling",
            "note": "Clusters are unsupervised groupings of dataset rows. Outcome is excluded from clustering and shown only for post-hoc exploration. This is not a clinical stratification tool.",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Clustering analysis failed: " + str(exc)) from exc
