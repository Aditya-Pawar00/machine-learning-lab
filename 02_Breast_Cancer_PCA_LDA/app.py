import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer PCA & LDA Analysis", page_icon="🔬", layout="wide")

@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()
scaler = artifacts["scaler"]
pca = artifacts["pca"]
lda = artifacts["lda"]

st.title("🔬 Breast Cancer Dimensionality Reduction (PCA & LDA)")
st.write("Demonstration of unsupervised PCA and supervised LDA dimensionality reduction on 30 features.")

df = pd.read_csv("breast_cancer.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)
X_lda = lda.transform(X_scaled)

col1, col2 = st.columns(2)
with col1:
    st.subheader("PCA 2D Projection (Unsupervised)")
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    st.pyplot(fig)

with col2:
    st.subheader("LDA 1D Projection (Supervised)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(X_lda[:, 0], y + np.random.normal(0, 0.05, len(y)), c=y, cmap="coolwarm", alpha=0.7)
    ax2.set_xlabel("Linear Discriminant 1")
    ax2.set_ylabel("Class")
    st.pyplot(fig2)
