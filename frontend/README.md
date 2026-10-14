# Data Cartography — React interface

This optional React/Vite frontend wraps the repository's hierarchical clustering workflow with a local FastAPI endpoint. It uses the existing Pima dataset, median imputation for zero-coded invalid measurements, standard scaling, Ward linkage and PCA.

## Setup

From the repository root:

~~~bash
python -m pip install -r requirements-react-api.txt
~~~

Start the API in terminal 1:

~~~bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
~~~

In terminal 2:

~~~bash
cd frontend
npm install
npm run dev
~~~

Open the local URL printed by Vite, normally http://127.0.0.1:5173/.

## Interface

- Choose k from 2 to 5 to recut the same Ward linkage tree
- Review a PCA projection coloured by cluster
- Inspect cluster sizes, silhouette score and standardized feature profiles
- Compare the observed Outcome distribution after clustering

## Interpretation

Outcome is not included in clustering features; it is displayed only after clustering to support exploratory analysis. These groups are not validated patient categories or a clinical risk tool. PCA is a two-dimensional projection and does not preserve all feature-space structure.
