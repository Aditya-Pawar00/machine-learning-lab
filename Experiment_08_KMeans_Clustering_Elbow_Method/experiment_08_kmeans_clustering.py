"""
Experiment 08: K-Means Clustering and Elbow Method
Dataset: Iris (4 numerical features)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    print("=" * 70)
    print("EXPERIMENT 08: K-MEANS CLUSTERING & ELBOW METHOD")
    print("=" * 70)

    # 1. Load dataset & scale
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Elbow method (K=1 to 10)
    print("Computing inertia for K = 1 to 10:")
    inertias = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"  K={k:2d}: Inertia (WCSS) = {km.inertia_:.3f}")

    # 3. Fit optimal K=3
    final_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = final_kmeans.fit_predict(X_scaled)
    print("\nCluster distribution for K=3:")
    print(pd.Series(labels).value_counts().sort_index())

    # 4. Silhouette scores
    print("\nSilhouette Scores (K=2 to 6):")
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lk = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, lk)
        print(f"  K={k}: Silhouette Score = {score:.4f}")

if __name__ == "__main__":
    main()
