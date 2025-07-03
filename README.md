🚗 Insurance Fraud Detection using XGBoost and SHAP

📄 Project Overview

This project focuses on detecting fraudulent vehicle insurance claims using machine learning techniques. The primary goal is to build a robust fraud classification model that balances high recall (detecting fraud) while minimizing false positives.

We explore data preprocessing, class imbalance handling using SMOTE, hyperparameter tuning, and model explainability using SHAP values.

⸻

📊 Dataset
	•	Source: Vehicle Insurance Fraud Detection Dataset on Kaggle
	•	Rows: ~15,000 insurance claims
	•	Target Variable: FraudFound (Yes/No)
	•	Features: Policy details, driver demographics, vehicle information, accident specifics, and claims history.

⸻

🛠️ Project Structure

insurance-fraud/
├── data/                      # Raw and processed data
├── notebooks/                 # Jupyter notebooks for each stage
│   ├── 01_exploratory_data_analysis.ipynb           # Exploratory Data Analysis
│   ├── 02_Modeling_Baseline.ipynb  # Baseline models (class-weighted & SMOTE)
│   ├── 03_Hyperparameter_Tuning.ipynb  # RandomizedSearchCV tuning
│   ├── 04_Model_Explainability.ipynb  # SHAP explainability
├── results/                   # Saved models, SHAP values, performance metrics
├── environment.yml            # Conda environment file
├── README.md                  # Project overview


⸻

🔍 Workflow
	1.	Data Exploration:
Performed 01_exploratory_data_analysis to understand class imbalance, feature distributions, and correlations.
	2.	Baseline Modeling:
Trained XGBoost classifiers with:
	•	Class-weighted loss
	•	SMOTE-balanced training data
	3.	Hyperparameter Tuning:
Applied RandomizedSearchCV on SMOTE data to optimize key hyperparameters.
	4.	Model Evaluation:
Focused on:
	•	Precision
	•	Recall
	•	F1-Score
	•	AUC-ROC
	5.	Explainability:
Used SHAP to:
	•	Visualize global feature importance
	•	Investigate individual fraud predictions

⸻

✅ Key Results

Model	Precision	Recall	F1-Score	AUC-ROC
Class-Weighted				
SMOTE				
SMOTE + Tuned				

(👉 You can fill in your actual performance numbers here.)

⸻

💡 Key Learnings
	•	Class imbalance was a critical challenge.
	•	SMOTE significantly improved fraud recall.
	•	SHAP provided valuable insights into the most influential features.

⸻

🚀 Next Steps
	•	Explore advanced SMOTE variants (e.g., SMOTE-NC for categorical features).
	•	Test additional algorithms (e.g., Random Forest, Logistic Regression).
	•	Build a Streamlit demo for interactive fraud prediction.

⸻

⚙️ Setup

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate insurance-fraud

# Launch JupyterLab
jupyter lab
