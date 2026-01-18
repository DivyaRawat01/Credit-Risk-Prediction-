# 🚀 Credit Risk Prediction System

**End-to-End Machine Learning Project with Deployment**

## 📌 Project Overview

This project builds an **end-to-end Credit Risk Prediction system** that classifies loan applicants as **Low Risk (Approved)** or **High Risk (Rejected)** using Machine Learning.

The solution follows **industry-standard ML workflow** — from data preprocessing and feature engineering to model deployment on **Hugging Face** Spaces with a clean UI.

## 🎯 Problem Statement

Financial institutions face losses due to loan defaults.
The objective is to **predict risky loan applicants in advance,** helping banks make safer approval decisions.

## 🧠 Solution Approach

- Analyze applicant financial & demographic data

- Engineer risk-based features

- Train and evaluate ML classification models

- Deploy the final model as an interactive web application

## 📂 Dataset Description

-Initial Features:

1. no_of_dependents

2. education

3. self_employed

4. income_annum

5. loan_amount

6. loan_term

7. cibil_score

8. residential_assets_value

9. commercial_assets_value

10. luxury_assets_value

11. bank_asset_value

12. **loan_status (Target)** 

#### Target Encoding:

- 0 → Loan Approved (Low Risk)

- 1 → Loan Rejected (High Risk)

### 🧩 Feature Engineering

Additional risk-oriented features were created:

1. Loan-to-Income Ratio (LIR)

2. Debt-to-Income Ratio (DIR)

These improve the model’s ability to capture repayment risk.

### 🔍 Exploratory Data Analysis (EDA)

Key insights obtained through:

- Target distribution analysis

- Correlation heatmaps

- Feature importance understanding

- Risk patterns across income, credit score, and assets

### ⚙️ Machine Learning Pipeline

- Data Cleaning & Encoding

- Feature Engineering

- Train–Test Split

- Scaling (Numeric features only)

- Handling Imbalanced Data (SMOTE)

- Model Training

- Hyperparameter Tuning

- Evaluation & Threshold Optimization

### 🤖 Models Used

- Logistic Regression

- Random Forest Classifier ✅ (Final Model)

- Gradient Boosting Classifier

- Evaluation Metrics:

- Precision

- Recall (critical for risk detection)

- F1-Score

- ROC-AUC

- Confusion Matrix

### ⚠️ Business-Driven Thresholding

Instead of using the default 0.5 probability threshold, a custom risk threshold (0.35) was applied to:

- Reduce false loan approvals

- Improve detection of high-risk applicants

### 🌐 Deployment

The final model is deployed using:

- Gradio

- Hugging Face Spaces

- Color-coded results:

1. 🟢 Low Risk (Approved)

2. 🔴 High Risk (Rejected)

## 🗂️ Project Structure
``` text
credit-risk-prediction/
│
├── app.py                    # Gradio UI & inference logic
├── credit_risk_model.pkl     # Trained ML model
├── scaler.pkl                # Trained scaler
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── notebook/
    └── training.ipynb        # Data analysis & model training
```

## 🛠️ Technologies Used

- Python

- Pandas, NumPy

- Scikit-learn

- Imbalanced-learn (SMOTE)

- Gradio

- Hugging Face Spaces

## 🚀 How to Run Locally
``` python
pip install -r requirements.txt
python app.py
```

## 🎓 Learning Outcomes

- Built a production-ready ML pipeline

- Learned risk-based decision modeling

- Implemented real-world evaluation metrics

- Gained experience in ML deployment & UI integration

## 👨‍💻 Author

**Divya Rawat**
- GitHub: [GitHub](https://github.com/DivyaRawat01/)
- LinkedIn:[LinkedIn](https://www.linkedin.com/in/divya-rawat-053940341/)
 
