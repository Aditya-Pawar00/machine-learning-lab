import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Iris K-Means Clustering", page_icon="🌿", layout="centered")

@st.cache_resource
def load_data():
    with open("model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    df = pd.read_csv("Iris.csv")
    return artifacts, df

artifacts, df = load_data()
X = df.drop(columns=["target"])
X_scaled = artifacts["scaler"].transform(X)

st.title("🌿 Iris K-Means Clustering & Elbow Method")
st.write("Unsupervised clustering on the Iris dataset with dynamic cluster count.")

k = st.slider("Select Number of Clusters (K)", 1, 8, 3)
km = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(X["sepal length (cm)"], X["petal length (cm)"], c=labels, cmap="tab10", alpha=0.8)
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Petal Length")
ax.set_title(f"K-Means Clustering with K={k}")
st.pyplot(fig)
