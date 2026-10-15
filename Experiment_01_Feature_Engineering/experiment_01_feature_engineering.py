"""
Experiment 01: Feature Engineering on Titanic Dataset
Savitribai Phule Pune University (SPPU) Machine Learning Laboratory
"""

import os
import urllib.request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    print("=" * 70)
    print("EXPERIMENT 01: FEATURE ENGINEERING ON TITANIC DATASET")
    print("=" * 70)

    # Step 2: Load and Explore Dataset
    csv_path = "titanic.csv"
    if not os.path.exists(csv_path):
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        print(f"Downloading Titanic dataset from {url}...")
        urllib.request.urlretrieve(url, csv_path)

    df = pd.read_csv(csv_path)
    print("Dataset dimensions:", df.shape)
    print("Column names:", df.columns.tolist())
    print("\nMissing values per column:\n", df.isnull().sum())

    # Step 3: Select Features and Target
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X = df[features].copy()
    y = df["survived"].copy()

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    # Step 4: Train-Test Split (80:20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTraining records: {X_train.shape[0]}, Testing records: {X_test.shape[0]}")

    # Step 5: Categorical and Numerical features
    numeric_features = ["age", "sibsp", "parch", "fare"]
    categorical_features = ["pclass", "sex", "embarked"]

    # Step 9: Integrated Preprocessing Pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", one_hot_encoder)
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_ready, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_ready, columns=feature_names, index=X_test.index)

    # Step 10: Save model-ready datasets
    train_ready = X_train_df.copy()
    train_ready["survived"] = y_train

    test_ready = X_test_df.copy()
    test_ready["survived"] = y_test

    train_ready.to_csv("titanic_train_model_ready.csv", index=False)
    test_ready.to_csv("titanic_test_model_ready.csv", index=False)

    print("\nProcessed training shape:", train_ready.shape)
    print("Processed testing shape:", test_ready.shape)
    print("Missing values in processed train data:", train_ready.isnull().sum().sum())
    print("Saved 'titanic_train_model_ready.csv' and 'titanic_test_model_ready.csv'.")
    print("\nExperiment 01 executed successfully!")

if __name__ == "__main__":
    main()
