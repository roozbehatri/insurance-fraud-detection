# 🚗 Insurance Fraud Detection using XGBoost and SHAP

## 📄 Project Overview

This project focuses on detecting fraudulent vehicle insurance claims using machine learning techniques. The primary goal is to build a robust fraud classification model that balances high recall (detecting fraud) while minimizing false positives.

We explore:
- Data preprocessing
- Class imbalance handling using **SMOTE**
- Hyperparameter tuning
- Model explainability using **SHAP values**

---

## 📊 Dataset

- **Source:** Vehicle Insurance Fraud Detection Dataset on Kaggle  
- **Rows:** ~15,000 insurance claims  
- **Target Variable:** `FraudFound` (Yes/No)  
- **Features:** Policy details, driver demographics, vehicle information, accident specifics, and claims history.

---

## 🛠️ Project Structure

```text
insurance-fraud/
├── data/                      # Raw and processed data
├── notebooks/                 # Jupyter notebooks for each stage
│   ├── 01_exploratory_data_analysis.ipynb         # Exploratory Data Analysis
│   ├── 02_Modeling_Baseline.ipynb                 # Baseline models (class-weighted & SMOTE)
│   ├── 03_Hyperparameter_Tuning.ipynb             # RandomizedSearchCV tuning
│   ├── 04_Model_Explainability.ipynb              # SHAP explainability
├── results/                   # Saved models, SHAP values, performance metrics
├── environment.yml            # Conda environment file
├── README.md                  # Project overview
