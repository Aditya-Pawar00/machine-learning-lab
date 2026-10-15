"""
Experiment 02: Dimensionality Reduction using PCA and LDA
Savitribai Phule Pune University (SPPU) Machine Learning Laboratory
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

def main():
    print("=" * 70)
    print("EXPERIMENT 02: DIMENSIONALITY REDUCTION (PCA & LDA)")
    print("=" * 70)

    # Step 2: Load dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()
    print(f"Dataset shape: {X.shape}, Classes: {dict(enumerate(dataset.target_names))}")

    # Step 3: Train-test split & Scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    def evaluate(Xtr, Xte, method_name):
        model = LogisticRegression(max_iter=5000, random_state=42)
        t0 = time.perf_counter()
        model.fit(Xtr, y_train)
        fit_t = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        pred = model.predict(Xte)
        pred_t = time.perf_counter() - t1
        
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        results.append({
            "Method": method_name,
            "Features": Xtr.shape[1],
            "Accuracy": acc,
            "F1-score (macro)": f1,
            "Fit time (s)": fit_t,
            "Prediction time (s)": pred_t
        })
        return model, pred

    # Baseline (30 features)
    _, baseline_pred = evaluate(X_train_scaled, X_test_scaled, "Original features (30)")

    # PCA 95% variance
    pca_95 = PCA(n_components=0.95, svd_solver="full")
    X_tr_pca95 = pca_95.fit_transform(X_train_scaled)
    X_te_pca95 = pca_95.transform(X_test_scaled)
    _, pca95_pred = evaluate(X_tr_pca95, X_te_pca95, f"PCA (95% var, {pca_95.n_components_} comp)")

    # PCA 2 components
    pca_2 = PCA(n_components=2)
    X_tr_pca2 = pca_2.fit_transform(X_train_scaled)
    X_te_pca2 = pca_2.transform(X_test_scaled)
    _, pca2_pred = evaluate(X_tr_pca2, X_te_pca2, "PCA (2 components)")

    # LDA 1 component
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_tr_lda = lda.fit_transform(X_train_scaled, y_train)
    X_te_lda = lda.transform(X_test_scaled)
    _, lda_pred = evaluate(X_tr_lda, X_te_lda, "LDA (1 component)")

    results_df = pd.DataFrame(results)
    print("\nComparison Results:")
    print(results_df.to_string(index=False))

    print("\nExperiment 02 executed successfully!")

if __name__ == "__main__":
    main()
