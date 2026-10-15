"""
Experiment 05: Classification Using Regularization Techniques (Ridge & Lasso)
Savitribai Phule Pune University (SPPU) Machine Learning Laboratory
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

def main():
    print("=" * 70)
    print("EXPERIMENT 05: REGULARIZATION TECHNIQUES (RIDGE & LASSO)")
    print("=" * 70)

    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    print(f"Dataset shape: {X.shape}, Target classes: {dict(enumerate(dataset.target_names))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic baseline (weak L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1e6, max_iter=10000, random_state=42))
        ]),
        "Ridge Classifier (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeClassifier(alpha=1.0))
        ]),
        "L1 Logistic (Lasso-style)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="l1", solver="liblinear", C=1.0,
                max_iter=10000, random_state=42
            ))
        ])
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        tr_pred = model.predict(X_train)
        te_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Train Acc": accuracy_score(y_train, tr_pred),
            "Test Acc": accuracy_score(y_test, te_pred),
            "Test F1": f1_score(y_test, te_pred, zero_division=0)
        })

    res_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(res_df.to_string(index=False))

    # Cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n5-Fold Stratified Cross-Validation Accuracy:")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"  {name:30s}: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}")

    print("\nExperiment 05 executed successfully!")

if __name__ == "__main__":
    main()
