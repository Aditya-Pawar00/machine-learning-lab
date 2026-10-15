"""
Experiment 09: DBSCAN Density-Based Clustering and Noise Detection
Dataset: Two-Moons (Synthetic Non-Spherical Clusters)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

def main():
    print("=" * 70)
    print("EXPERIMENT 09: DBSCAN DENSITY-BASED CLUSTERING")
    print("=" * 70)

    # 1. Generate non-spherical dataset
    X, true_labels = make_moons(n_samples=500, noise=0.06, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Fit DBSCAN
    dbscan = DBSCAN(eps=0.25, min_samples=5)
    cluster_labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int(np.sum(cluster_labels == -1))

    print(f"Clusters found: {n_clusters}")
    print(f"Noise points (-1): {n_noise} ({100 * n_noise / len(X):.2f}%)")
    print(f"Adjusted Rand Index: {adjusted_rand_score(true_labels, cluster_labels):.4f}")

    # 3. Parameter sweep
    print("\nParameter Sweep for eps:")
    for eps in [0.15, 0.20, 0.25, 0.30, 0.35]:
        m = DBSCAN(eps=eps, min_samples=5)
        lbls = m.fit_predict(X_scaled)
        c_count = len(set(lbls)) - (1 if -1 in lbls else 0)
        n_pts = int(np.sum(lbls == -1))
        print(f"  eps={eps:.2f}: Clusters={c_count}, Noise={n_pts}")

if __name__ == "__main__":
    main()
