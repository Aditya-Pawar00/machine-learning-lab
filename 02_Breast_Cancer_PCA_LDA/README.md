# Breast Cancer Dimensionality Reduction (PCA & LDA)

Evaluates **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)** for feature compression and classification.

## 🚀 Features
- 30 numerical features reduced to 2 principal components and 1 linear discriminant
- Comparison of unsupervised (PCA) vs supervised (LDA) projections
- Interactive visualization app

## 🧠 ML Workflow
```text
30 Features → StandardScaler → PCA (Variance Preservation) / LDA (Class Separation) → Logistic Regression
```

## ⚙️ Usage
```bash
streamlit run app.py
```
