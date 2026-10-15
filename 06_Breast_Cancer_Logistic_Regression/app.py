import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Breast Cancer Classification", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🩺 Breast Cancer Binary Classifier")
st.write("Predict whether a tumor is **Malignant** or **Benign** using Logistic Regression.")

df = pd.read_csv("breast_cancer.csv")
defaults = df.mean()

with st.form("diag_form"):
    mean_radius = st.slider("Mean Radius", 6.0, 30.0, float(defaults["mean radius"]))
    mean_texture = st.slider("Mean Texture", 9.0, 40.0, float(defaults["mean texture"]))
    mean_perimeter = st.slider("Mean Perimeter", 40.0, 190.0, float(defaults["mean perimeter"]))
    mean_area = st.slider("Mean Area", 140.0, 2500.0, float(defaults["mean area"]))
    submitted = st.form_submit_button("Classify Tumor")

if submitted:
    sample = defaults.drop("target").to_frame().T
    sample["mean radius"] = mean_radius
    sample["mean texture"] = mean_texture
    sample["mean perimeter"] = mean_perimeter
    sample["mean area"] = mean_area
    
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0]
    
    if pred == 1:
        st.error(f"⚠️ **Malignant** (Probability: {prob[1]*100:.1f}%)")
    else:
        st.success(f"✅ **Benign** (Probability: {prob[0]*100:.1f}%)")
