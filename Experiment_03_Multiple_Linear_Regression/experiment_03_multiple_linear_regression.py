"""
Experiment 03: Multiple Linear Regression for Automobile Price Prediction
Savitribai Phule Pune University (SPPU) Machine Learning Laboratory
"""

import os
import ssl
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    print("=" * 70)
    print("EXPERIMENT 03: MULTIPLE LINEAR REGRESSION (AUTOS DATASET)")
    print("=" * 70)

    csv_path = "autos_dataset.csv"
    if not os.path.exists(csv_path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
        columns = [
            "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
            "num-of-doors", "body-style", "drive-wheels", "engine-location",
            "wheel-base", "length", "width", "height", "curb-weight",
            "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
            "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
            "city-mpg", "highway-mpg", "price"
        ]
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.urlopen(url, context=ctx)
        df_raw = pd.read_csv(req, header=None, names=columns)
        df_raw.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    df = df.replace("?", np.nan)

    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            orig_valid = df[col].notna().sum()
            conv_valid = converted.notna().sum()
            if orig_valid > 0 and conv_valid / orig_valid > 0.5:
                df[col] = converted

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).copy()
    print("Cleaned dataset shape:", df.shape)

    X = df.drop(columns=["price"])
    y = df["price"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nEvaluation Metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f}")

    comparison = pd.DataFrame({
        "Actual price": y_test.reset_index(drop=True),
        "Predicted price": y_pred,
        "Residual": y_test.reset_index(drop=True) - y_pred
    })
    comparison.to_csv("autos_price_predictions.csv", index=False)
    print("Saved predictions to 'autos_price_predictions.csv'.")
    print("\nExperiment 03 executed successfully!")

if __name__ == "__main__":
    main()
