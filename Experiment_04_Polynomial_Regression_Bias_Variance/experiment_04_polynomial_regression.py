"""
Experiment 04: Polynomial Regression and Bias-Variance Trade-off
Savitribai Phule Pune University (SPPU) Machine Learning Laboratory
"""

import os
import ssl
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("=" * 70)
    print("EXPERIMENT 04: POLYNOMIAL REGRESSION & BIAS-VARIANCE TRADEOFF")
    print("=" * 70)

    local_file = "auto-mpg.data"
    columns = [
        "mpg", "cylinders", "displacement", "horsepower",
        "weight", "acceleration", "model_year", "origin", "car_name"
    ]

    if not os.path.exists(local_file):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.urlopen(url, context=ctx)
        with open(local_file, "wb") as f:
            f.write(req.read())

    df = pd.read_csv(
        local_file, sep=r"\s+", names=columns,
        na_values="?", comment="\t", skipinitialspace=True
    )

    data = df[["weight", "mpg"]].dropna().copy()
    print("Loaded data shape:", data.shape)

    X = data[["weight"]]
    y = data["mpg"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    degrees = [1, 2, 3, 5, 10, 15]
    results = []

    for degree in degrees:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
            ("regression", LinearRegression())
        ])
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        results.append({
            "Degree": degree,
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse,
            "Train R2": train_r2,
            "Test R2": test_r2
        })

    res_df = pd.DataFrame(results)
    print("\nPolynomial Degrees Evaluation:")
    print(res_df.to_string(index=False))

    best_idx = res_df["Test RMSE"].idxmin()
    best = res_df.loc[best_idx]
    print(f"\nBest degree by Test RMSE: {int(best['Degree'])} (RMSE: {best['Test RMSE']:.4f})")
    print("\nExperiment 04 executed successfully!")

if __name__ == "__main__":
    main()
