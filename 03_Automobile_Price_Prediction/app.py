import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Automobile Price Prediction", page_icon="🚗", layout="centered")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🚗 Automobile Price Prediction System")
st.write("Predict automobile retail price based on key mechanical and body specifications.")

with st.form("car_form"):
    engine_size = st.slider("Engine Size (cc / cu in)", 60, 350, 130)
    horsepower = st.slider("Horsepower", 45, 300, 105)
    curb_weight = st.slider("Curb Weight (lbs)", 1500, 4200, 2500)
    highway_mpg = st.slider("Highway MPG", 15, 60, 30)
    body_style = st.selectbox("Body Style", ["sedan", "hatchback", "wagon", "hardtop", "convertible"])
    submitted = st.form_submit_button("Estimate Price")

if submitted:
    df_in = pd.DataFrame([{
        "engine-size": engine_size, "horsepower": horsepower,
        "curb-weight": curb_weight, "highway-mpg": highway_mpg,
        "body-style": body_style
    }])
    price = model.predict(df_in)[0]
    st.subheader(f"Estimated Price: **${price:,.2f}**")
