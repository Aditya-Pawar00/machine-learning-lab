import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ridge & Lasso Regularization", page_icon="⚖️", layout="wide")

@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

data = load_models()
features = data["features"]
ridge_coef = data["ridge"].named_steps["clf"].coef_.ravel()
lasso_coef = data["lasso"].named_steps["clf"].coef_.ravel()

st.title("⚖️ Regularization Analysis (Ridge L2 vs Lasso L1)")
st.write("Comparing coefficient shrinkage and sparsity in high-dimensional classification.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Ridge (L2) Non-Zero Coefficients", np.count_nonzero(np.abs(ridge_coef) > 1e-5), f"out of {len(features)}")
with col2:
    st.metric("Lasso (L1) Non-Zero Coefficients", np.count_nonzero(np.abs(lasso_coef) > 1e-5), f"out of {len(features)}")

fig, ax = plt.subplots(figsize=(10, 4))
idx = np.arange(len(features))
ax.bar(idx - 0.2, ridge_coef, width=0.4, label="Ridge (L2)")
ax.bar(idx + 0.2, lasso_coef, width=0.4, label="Lasso (L1)")
ax.set_xticks(idx)
ax.set_xticklabels(features, rotation=90, fontsize=8)
ax.set_ylabel("Coefficient Value")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
