# src/clustering/kmeans.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42
) -> pd.DataFrame:
    """
    Ejecuta KMeans para cada k en k_range y devuelve un DataFrame con
    inercia y silhouette score para cada k.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        sil = silhouette_score(X, labels)
        results.append({"k": k, "inertia": inertia, "silhouette": sil})
    return pd.DataFrame(results)

def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> KMeans:
    """
    Entrena un KMeans con n_clusters sobre X y devuelve el modelo ajustado.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    km.fit(X)
    return km
