import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load model & scaler
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

custom_css = """
body {
    background: linear-gradient(135deg, #f9fbff, #eef2f7);
}

.gradio-container {
    max-width: 750px !important;
    margin: auto;
    font-family: 'Segoe UI', sans-serif;
    background: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.06);
}

h1 {
    text-align: center;
    color: #2c3e50;
}

#result-box {
    font-size: 18px;
    font-weight: 600;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
"""


NUM_COLS = [
    'no_of_dependents',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value',
    'loan_to_income_ratio',
    'debt_to_income_ratio'
]


def predict_loan_status(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
):
    if loan_amount is None or loan_amount <= 0:
        return "<div style='color:#d35400;'>⚠️ Please enter a valid Loan Amount</div>"

    if loan_term is None or loan_term <= 0:
        return "<div style='color:#d35400;'>⚠️ Please enter a valid Loan Term</div>"

    if income_annum is None or income_annum <= 0:
         return "<div style='color:#d35400;'>⚠️ Please enter a valid Annual Income</div>"

    # Encode categorical
    education_enc = 1 if education == "Graduate" else 0
    self_employed_enc = 1 if self_employed == "Yes" else 0

    # Feature engineering
    loan_to_income_ratio = loan_amount / income_annum
    debt_to_income_ratio = loan_amount / (income_annum + bank_asset_value)

    # Create numeric DataFrame WITH feature names
    numeric_df = pd.DataFrame([[
        no_of_dependents,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
        loan_to_income_ratio,
        debt_to_income_ratio
    ]], columns=NUM_COLS)

   
    # Scale numeric features
    numeric_scaled = scaler.transform(numeric_df)

    # Combine with categorical (ORDER MATTERS)
    final_input = np.concatenate(
        [numeric_scaled, [[education_enc, self_employed_enc]]],
        axis=1
    )

    # Predict
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    if probability >= 0.35:
        return f"""
    <div style='color:#c0392b;'>
        ❌ <b>High Risk (Loan Rejected)</b><br>
        Risk Probability: {probability:.2f}
    </div>
    """
    else:
        return f"""
    <div style='color:#27ae60;'>
        ✅ <b>Low Risk (Loan Approved)</b><br>
        Risk Probability: {probability:.2f}
    </div>
    """


# Gradio Interface
interface = gr.Interface(
    fn=predict_loan_status,
    inputs=[
        gr.Number(label="No of Dependents"),
        gr.Dropdown(["Graduate", "Not Graduate"], label="Education"),
        gr.Dropdown(["Yes", "No"], label="Self Employed"),
        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term (Months)"),
        gr.Number(label="CIBIL Score"),
        gr.Number(label="Residential Assets Value"),
        gr.Number(label="Commercial Assets Value"),
        gr.Number(label="Luxury Assets Value"),
        gr.Number(label="Bank Asset Value")
    ],
    outputs=gr.HTML(elem_id="result-box"),
    title="Credit Risk Prediction System",
    description="Predict whether a loan applicant is High Risk or Low Risk using Machine Learning",
    css = custom_css
)

interface.launch()
