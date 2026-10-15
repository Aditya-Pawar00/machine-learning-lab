"""
Experiment 06: Binary Classification using Logistic Regression
Dataset: Breast Cancer Wisconsin (Diagnostic)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score
)

def main():
    print("=" * 70)
    print("EXPERIMENT 06: LOGISTIC REGRESSION BINARY CLASSIFICATION")
    print("=" * 70)

    # 1. Load dataset & recode target: 1 = malignant, 0 = benign
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = (dataset.target == 0).astype(int)
    y.name = "malignant"

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: Benign (0)={np.sum(y==0)}, Malignant (1)={np.sum(y==1)}")

    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Model pipeline with StandardScaler + LogisticRegression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42))
    ])
    model.fit(X_train, y_train)

    # 4. Predict & Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix:\n{cm}")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"\nMetrics:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"], digits=4))

if __name__ == "__main__":
    main()
