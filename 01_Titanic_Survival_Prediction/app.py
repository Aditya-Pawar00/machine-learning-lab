import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_artifacts()

st.title("🚢 Titanic Survival Prediction System")
st.markdown("Predict passenger survival on the Titanic using an optimized **Random Forest Pipeline**.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Gender", ["male", "female"])
        age = st.slider("Age", 0.5, 80.0, 28.0)
        fare = st.slider("Ticket Fare ($)", 0.0, 500.0, 32.0)
    with col2:
        sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
        parch = st.number_input("Parents / Children Aboard", 0, 10, 0)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
        
    submitted = st.form_submit_button("Predict Survival")

if submitted:
    input_df = pd.DataFrame([{
        "pclass": pclass, "sex": sex, "age": age,
        "sibsp": sibsp, "parch": parch, "fare": fare, "embarked": embarked
    }])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    
    st.subheader("Prediction Result")
    if pred == 1:
        st.success(f"🎉 **Survived** (Confidence: {prob[1]*100:.1f}%)")
    else:
        st.error(f"💀 **Did Not Survive** (Confidence: {prob[0]*100:.1f}%)")
