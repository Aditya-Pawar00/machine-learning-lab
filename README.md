# Machine Learning Laboratory (SPPU Curriculum)

This repository contains practical experiments, Jupyter Notebooks, executable Python scripts, datasets, and viva-voce preparations for the **Machine Learning Laboratory course (Savitribai Phule Pune University - SPPU)**.

---

## 📁 Repository Structure

```text
machine-learning-lab-sppu/
│
├── datasets/                                           # Central repository datasets
│   ├── titanic.csv                                     # Experiment 1 dataset
│   ├── autos_dataset.csv                               # Experiment 3 dataset
│   └── auto-mpg.data                                   # Experiment 4 dataset
│
├── Experiment_01_Feature_Engineering/
│   ├── 1_Feature_Engineering_Titanic_SPPU_Lab.ipynb    # Jupyter Notebook
│   ├── experiment_01_feature_engineering.py            # Standalone Python script
│   └── titanic.csv                                     # Local dataset copy
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
├── requirements.txt                                    # Python dependencies
├── .gitignore                                          # Git ignore configurations
└── README.md                                           # Documentation and instructions
```

---

## 📚 Experiment Index

| Exp No. | Topic / Practical Title | Dataset Used | Techniques / Algorithms | Status |
|:---:|:---|:---|:---|:---:|
| **01** | **Feature Engineering** | Titanic Passenger Dataset | Median/mode imputation, One-Hot Encoding, StandardScaler, ColumnTransformer | ✅ Completed |
| **02** | **Dimensionality Reduction** | Breast Cancer Wisconsin | PCA (95% variance & 2D), LDA (1D), Logistic Regression | ✅ Completed |
| **03** | **Multiple Linear Regression** | UCI Automobile Price | Multiple Linear Regression, Missing Value Cleaning, MSE, RMSE, R2, MAE | ✅ Completed |
| **04** | **Polynomial Regression & Bias-Variance** | UCI Auto MPG | Polynomial features (Degrees 1, 2, 3, 5, 10, 15), Bias-Variance trade-off | ✅ Completed |
| **05** | **Regularization Techniques (Ridge & Lasso)** | Breast Cancer Wisconsin | RidgeClassifier (L2), L1-Logistic (Lasso), Multicollinearity (VIF), Stratified K-Fold CV | ✅ Completed |
| **06** | *Experiment 06 (Upcoming)* | *Pending next upload* | - | ⏳ Pending |
| **07** | *Experiment 07 (Upcoming)* | *Pending next upload* | - | ⏳ Pending |
| **08** | *Experiment 08 (Upcoming)* | *Pending next upload* | - | ⏳ Pending |
| **09** | *Experiment 09 (Upcoming)* | *Pending next upload* | - | ⏳ Pending |
| **10** | *Experiment 10 (Upcoming)* | *Pending next upload* | - | ⏳ Pending |
| **11** | *Experiment 11 (Upcoming)* | *Pending next upload* | - | ⏳ Pending |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your computer.

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Experiments

### Option A: Using Jupyter Notebook / Google Colab
Launch JupyterLab or Notebook:
```bash
jupyter notebook
```
Navigate to any experiment directory (e.g., `Experiment_01_Feature_Engineering/`) and open the `.ipynb` file.

### Option B: Running Standalone Python Scripts
Run each script directly from the terminal:
```bash
# Experiment 01
cd Experiment_01_Feature_Engineering
python experiment_01_feature_engineering.py

# Experiment 02
cd ../Experiment_02_PCA_and_LDA
python experiment_02_pca_and_lda.py

# Experiment 03
cd ../Experiment_03_Multiple_Linear_Regression
python experiment_03_multiple_linear_regression.py

# Experiment 04
cd ../Experiment_04_Polynomial_Regression_Bias_Variance
python experiment_04_polynomial_regression.py

# Experiment 05
cd ../Experiment_05_Regularization_Ridge_and_Lasso
python experiment_05_regularization_ridge_lasso.py
```

---

## 📌 Summary of Completed Experiments

### Experiment 01: Feature Engineering
- **Aim:** Prepare a clean, model-ready dataset using imputation, one-hot encoding, and scaling.
- **Key Takeaways:**
  - Standard scaling prevents scale-sensitive estimators from being dominated by large numeric ranges.
  - Imputers and scalers must be fitted **only** on the training set to prevent data leakage.
  - Generates `titanic_train_model_ready.csv` and `titanic_test_model_ready.csv`.

### Experiment 02: Dimensionality Reduction using PCA and LDA
- **Aim:** Compare unsupervised PCA and supervised LDA on classification performance and computation speed.
- **Key Takeaways:**
  - PCA with 95% variance reduces 30 features to ~10 with negligible drop in accuracy.
  - LDA compresses the binary target into a single highly discriminative 1D component.
  - Supervised projection (LDA) achieves high separation using class labels.

### Experiment 03: Multiple Linear Regression
- **Aim:** Predict automobile prices using multiple numerical and categorical specifications.
- **Key Takeaways:**
  - Data cleaning handles missing value markers (`?`).
  - Preprocessing pipeline integrates imputation and categorical encoding cleanly.
  - Evaluates models with MSE, RMSE, MAE, and $R^2$ metrics.

### Experiment 04: Polynomial Regression and Bias-Variance Trade-off
- **Aim:** Model nonlinear vehicle weight vs. MPG and study the bias-variance trade-off across degrees 1 to 15.
- **Key Takeaways:**
  - Degree 1 exhibits high bias (underfitting).
  - Degree 2 to 3 provides the optimal balance of bias and variance.
  - Higher degrees (10, 15) risk wild oscillations at boundary points.

### Experiment 05: Regularization (Ridge and Lasso Regression)
- **Aim:** Compare L2 (Ridge) and L1 (Lasso) regularization for classification under severe multicollinearity.
- **Key Takeaways:**
  - Correlation heatmaps and VIF reveal high collinearity between radius, perimeter, and area.
  - Ridge distributes weights evenly across correlated features.
  - L1 performs sparse feature selection, zeroing out uninformative weights.

---

## 📤 How to Push to GitHub

1. Create a new empty repository on [GitHub](https://github.com/new). Name it e.g. `machine-learning-lab-sppu`.
2. In your terminal inside this folder, run:
```bash
git remote add origin https://github.com/<your-github-username>/machine-learning-lab-sppu.git
git branch -M main
git push -u origin main
```
