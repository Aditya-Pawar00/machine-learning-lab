"""
Experiment 07: SVM Classifiers (Linear vs Nonlinear RBF Kernels)
Dataset: Iris (Setosa vs Versicolor)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def main():
    print("=" * 70)
    print("EXPERIMENT 07: SVM LINEAR VS NONLINEAR (RBF) KERNELS")
    print("=" * 70)

    # 1. Load dataset & filter to 2 classes
    iris = load_iris(as_frame=True)
    df = iris.frame[iris.frame["target"].isin([0, 1])].copy()
    X = df[iris.feature_names].copy()
    y = df["target"].astype(int)

    print(f"Dataset shape: {X.shape}")

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", C=1.0))
        ]),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
        ])
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, pred, pos_label=1, zero_division=0),
            "F1-score": f1_score(y_test, pred, pos_label=1, zero_division=0)
        })
        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_test, pred, target_names=["Setosa", "Versicolor"], digits=4))

    print("\nComparison:")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
