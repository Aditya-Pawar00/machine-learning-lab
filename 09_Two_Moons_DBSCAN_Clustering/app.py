import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="DBSCAN Clustering", page_icon="🌙", layout="centered")

st.title("🌙 DBSCAN Density-Based Clustering & Noise Detection")
st.write("Demonstration of non-spherical cluster discovery and outlier detection on the Two-Moons dataset.")

df = pd.read_csv("two_moons.csv")
X = df.values
X_scaled = StandardScaler().fit_transform(X)

col1, col2 = st.columns(2)
with col1:
    eps = st.slider("Epsilon (eps)", 0.10, 0.50, 0.25, 0.05)
with col2:
    min_samples = st.slider("Min Samples", 2, 15, 5)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int(np.sum(labels == -1))

st.info(f"**Discovered Clusters:** {n_clusters} | **Noise Outliers:** {n_noise}")

fig, ax = plt.subplots(figsize=(6, 4))
unique_lbls = sorted(set(labels))
colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_lbls), 1)))

for c, l in zip(colors, unique_lbls):
    mask = labels == l
    if l == -1:
        ax.scatter(X[mask, 0], X[mask, 1], c="black", marker="x", s=50, label="Noise (-1)")
    else:
        ax.scatter(X[mask, 0], X[mask, 1], color=c, s=25, label=f"Cluster {l}")

ax.set_title(f"DBSCAN (eps={eps}, min_samples={min_samples})")
ax.legend()
st.pyplot(fig)
