import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Ensemble Learning Comparison", page_icon="🌲", layout="centered")

@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

models = load_models()

st.title("🌲 Ensemble Learning Comparison")
st.write("Compare predictions between **Random Forest**, **AdaBoost**, and **Logistic Regression**.")

df = pd.read_csv("breast_cancer.csv")
defaults = df.mean().drop("target")

sample = defaults.to_frame().T
preds = {
    "Logistic Regression (Baseline)": "Benign" if models["lr"].predict(sample)[0] == 1 else "Malignant",
    "Random Forest (Bagging)": "Benign" if models["rf"].predict(sample)[0] == 1 else "Malignant",
    "AdaBoost (Boosting)": "Benign" if models["ada"].predict(sample)[0] == 1 else "Malignant"
}

st.subheader("Model Predictions for Sample Tumor:")
for model_name, pred_val in preds.items():
    st.write(f"- **{model_name}:** {pred_val}")
