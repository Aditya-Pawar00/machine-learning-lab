import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Auto MPG Polynomial Regression", page_icon="⛽", layout="centered")

@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

models = load_models()

st.title("⛽ Auto MPG Polynomial Regression & Bias-Variance")
st.write("Examine the bias-variance trade-off by comparing different polynomial degrees.")

weight = st.slider("Vehicle Weight (lbs)", 1500, 5200, 3000)
degree = st.selectbox("Select Polynomial Degree", [1, 2, 3, 5])

pred = models[degree].predict(pd.DataFrame([[weight]], columns=["weight"]))[0]
st.subheader(f"Predicted Fuel Efficiency: **{pred:.2f} MPG**")

df_m = pd.read_csv("auto-mpg.data", sep=r"\s+", names=["mpg", "cyl", "disp", "hp", "weight", "acc", "year", "orig", "name"], na_values="?", comment="\t").dropna()
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(df_m["weight"], df_m["mpg"], alpha=0.3, color="gray", label="Data Points")

w_line = np.linspace(df_m["weight"].min(), df_m["weight"].max(), 200).reshape(-1, 1)
y_line = models[degree].predict(pd.DataFrame(w_line, columns=["weight"]))
ax.plot(w_line, y_line, color="red", label=f"Degree {degree} Curve")
ax.scatter([weight], [pred], color="blue", s=100, zorder=5, label="Prediction")
ax.set_xlabel("Weight")
ax.set_ylabel("MPG")
ax.legend()
st.pyplot(fig)
