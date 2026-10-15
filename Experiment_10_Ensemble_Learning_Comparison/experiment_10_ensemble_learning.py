"""
Experiment 10: Ensemble Learning Comparison (Random Forest & AdaBoost vs Logistic Regression)
Dataset: Breast Cancer Wisconsin (Diagnostic)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def main():
    print("=" * 70)
    print("EXPERIMENT 10: ENSEMBLE LEARNING COMPARISON")
    print("=" * 70)

    # 1. Load data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression (Baseline)": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=42)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=1.0, random_state=42
        )
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (Macro)": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall (Macro)": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1 (Macro)": f1_score(y_test, pred, average="macro", zero_division=0)
        })

    print("\nModel Comparison on Test Set:")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
