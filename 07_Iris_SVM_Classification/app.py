import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Iris SVM Classifier", page_icon="🌸", layout="centered")

@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

models = load_models()

st.title("🌸 Iris Flower SVM Classifier (Linear vs RBF)")
st.write("Classify Iris Setosa vs Iris Versicolor using Support Vector Machines.")

kernel = st.radio("SVM Kernel", ["Linear", "RBF"])
selected_model = models["linear"] if kernel == "Linear" else models["rbf"]

with st.form("iris_form"):
    sl = st.slider("Sepal Length (cm)", 4.0, 7.5, 5.0)
    sw = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    pl = st.slider("Petal Length (cm)", 1.0, 5.5, 1.4)
    pw = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
    submitted = st.form_submit_button("Classify")

if submitted:
    df_in = pd.DataFrame([[sl, sw, pl, pw]], columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])
    pred = selected_model.predict(df_in)[0]
    prob = selected_model.predict_proba(df_in)[0]
    names = {0: "Iris Setosa", 1: "Iris Versicolor"}
    st.subheader(f"Predicted Species: **{names[pred]}**")
    st.info(f"Confidence: Setosa: {prob[0]*100:.1f}%, Versicolor: {prob[1]*100:.1f}%")
