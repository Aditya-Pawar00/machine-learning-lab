# Machine Learning Laboratory (SPPU Curriculum)

This repository contains clean, modular, and directly executable Python scripts (`.py`), interactive Jupyter Notebooks (`.ipynb`), datasets, and evaluations for the **Machine Learning Laboratory course (Savitribai Phule Pune University - SPPU)**.

---

## 📁 Repository Structure

```text
machine-learning-lab-sppu/
│
├── datasets/                                           # Central datasets directory
│   ├── titanic.csv                                     # Exp 01 dataset
│   ├── autos_dataset.csv                               # Exp 03 dataset
│   └── auto-mpg.data                                   # Exp 04 dataset
│
├── Experiment_01_Feature_Engineering/
│   ├── 1_Feature_Engineering_Titanic_SPPU_Lab.ipynb
│   ├── experiment_01_feature_engineering.py
│   └── titanic.csv
│
├── Experiment_02_PCA_and_LDA/
│   ├── 2_Dimensionality_Reduction_Using_PCA_and_LDA.ipynb
│   └── experiment_02_pca_and_lda.py
│
├── Experiment_03_Multiple_Linear_Regression/
│   ├── 3_autos_price_multiple_linear_regression.ipynb
│   ├── experiment_03_multiple_linear_regression.py
│   └── autos_dataset.csv
│
├── Experiment_04_Polynomial_Regression_Bias_Variance/
│   ├── 4_polynomial_regression_bias_variance_sppu.ipynb
│   ├── experiment_04_polynomial_regression.py
│   └── auto-mpg.data
│
├── Experiment_05_Regularization_Ridge_and_Lasso/
│   ├── 5_Classify_data_using_regularization_Ridge_Lasso.ipynb
│   └── experiment_05_regularization_ridge_lasso.py
│
├── Experiment_06_Binary_Classification_Logistic_Regression/
│   ├── 6_binary_classification_logistic_regression.ipynb
│   └── experiment_06_logistic_regression.py
│
├── Experiment_07_SVM_Linear_vs_Nonlinear_Kernels/
│   ├── 7_svm_linear_vs_nonlinear_kernels.ipynb
│   └── experiment_07_svm_kernels.py
│
├── Experiment_08_KMeans_Clustering_Elbow_Method/
│   ├── 8_kmeans_clustering_elbow_method.ipynb
│   └── experiment_08_kmeans_clustering.py
│
├── Experiment_09_DBSCAN_Clustering_Noise_Detection/
│   ├── 9_dbscan_clustering_noise_detection.ipynb
│   └── experiment_09_dbscan_clustering.py
│
├── Experiment_10_Ensemble_Learning_Comparison/
│   ├── 10_ensemble_learning_comparison.ipynb
│   └── experiment_10_ensemble_learning.py
│
├── requirements.txt                                    # Python dependencies
├── .gitignore                                          # Git ignore configurations
└── README.md                                           # Documentation and instructions
```

---

## 📚 Experiment Index

| Exp No. | Topic / Practical Title | Dataset Used | Techniques / Algorithms | Status |
|:---:|:---|:---|:---|:---:|
| **01** | **Feature Engineering** | Titanic Passenger Dataset | Imputation, One-Hot Encoding, StandardScaler, ColumnTransformer | ✅ Completed |
| **02** | **Dimensionality Reduction** | Breast Cancer Wisconsin | PCA (95% variance & 2D), LDA (1D), Logistic Regression | ✅ Completed |
| **03** | **Multiple Linear Regression** | UCI Automobile Price | Multiple Linear Regression, Missing Value Handling, MSE, RMSE, R2, MAE | ✅ Completed |
| **04** | **Polynomial Regression & Bias-Variance** | UCI Auto MPG | Polynomial features (Degrees 1 to 15), Bias-Variance trade-off | ✅ Completed |
| **05** | **Regularization (Ridge & Lasso)** | Breast Cancer Wisconsin | RidgeClassifier (L2), L1-Logistic (Lasso), Multicollinearity (VIF), K-Fold CV | ✅ Completed |
| **06** | **Binary Classification via Logistic Regression** | Breast Cancer Wisconsin | Logistic Regression, Confusion Matrix, Precision, Recall, F1-Score | ✅ Completed |
| **07** | **SVM Linear vs Nonlinear Kernels** | Iris Dataset (Setosa vs Versicolor) | Linear SVM vs RBF Kernel SVM, Evaluation Metrics | ✅ Completed |
| **08** | **K-Means Clustering & Elbow Method** | Iris Dataset (4 features) | K-Means, WCSS / Inertia Elbow Curve, Silhouette Analysis, PCA Visualization | ✅ Completed |
| **09** | **DBSCAN Clustering & Noise Detection** | Synthetic Two-Moons | Density-Based Clustering, Core/Border/Noise Identification, Parameter Sweep | ✅ Completed |
| **10** | **Ensemble Learning Comparison** | Breast Cancer Wisconsin | Logistic Regression (Baseline) vs Random Forest vs AdaBoost | ✅ Completed |
| **11** | *Experiment 11 (Final)* | *Pending upload* | - | ⏳ Pending |

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Python Scripts Directly
You can run any experiment script directly:
```bash
python Experiment_01_Feature_Engineering/experiment_01_feature_engineering.py
python Experiment_02_PCA_and_LDA/experiment_02_pca_and_lda.py
python Experiment_03_Multiple_Linear_Regression/experiment_03_multiple_linear_regression.py
python Experiment_04_Polynomial_Regression_Bias_Variance/experiment_04_polynomial_regression.py
python Experiment_05_Regularization_Ridge_and_Lasso/experiment_05_regularization_ridge_lasso.py
python Experiment_06_Binary_Classification_Logistic_Regression/experiment_06_logistic_regression.py
python Experiment_07_SVM_Linear_vs_Nonlinear_Kernels/experiment_07_svm_kernels.py
python Experiment_08_KMeans_Clustering_Elbow_Method/experiment_08_kmeans_clustering.py
python Experiment_09_DBSCAN_Clustering_Noise_Detection/experiment_09_dbscan_clustering.py
python Experiment_10_Ensemble_Learning_Comparison/experiment_10_ensemble_learning.py
```

### 3. Open in Jupyter Notebook / Lab
```bash
jupyter notebook
```

---

## 📤 How to Push to GitHub

1. Create a new repository on [GitHub](https://github.com/new) (e.g. `machine-learning-lab-sppu`).
2. Run the following commands in this directory:
```bash
git remote add origin https://github.com/<YOUR-USERNAME>/machine-learning-lab-sppu.git
git branch -M main
git push -u origin main
```
