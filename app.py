import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load model & scaler
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

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
    no_of_dependents, education, self_employed, income_annum,
    loan_amount, loan_term, cibil_score,
    residential_assets_value, commercial_assets_value,
    luxury_assets_value, bank_asset_value
):
    # 🔴 VALIDATION
    if loan_amount is None or loan_amount <= 0:
        return "<div class='warn'>⚠️ Please enter a valid Loan Amount</div>"
    if loan_term is None or loan_term <= 0:
        return "<div class='warn'>⚠️ Please enter a valid Loan Term</div>"
    if income_annum is None or income_annum <= 0:
        return "<div class='warn'>⚠️ Please enter a valid Annual Income</div>"

    # Encode categorical
    education_enc = 1 if education == "Graduate" else 0
    self_employed_enc = 1 if self_employed == "Yes" else 0

    # Feature engineering
    loan_to_income_ratio = loan_amount / income_annum
    debt_to_income_ratio = loan_amount / (income_annum + bank_asset_value)

    numeric_df = pd.DataFrame([[
        no_of_dependents, income_annum, loan_amount, loan_term,
        cibil_score, residential_assets_value, commercial_assets_value,
        luxury_assets_value, bank_asset_value,
        loan_to_income_ratio, debt_to_income_ratio
    ]], columns=NUM_COLS)

    numeric_scaled = scaler.transform(numeric_df)

    final_input = np.concatenate(
        [numeric_scaled, [[education_enc, self_employed_enc]]],
        axis=1
    )

    probability = model.predict_proba(final_input)[0][1]

    if probability >= 0.35:
        return f"""
        <div class='reject'>
            ❌ <b>High Risk</b><br>
            Rejection Probability: {probability:.2f}
        </div>
        """
    else:
        return f"""
        <div class='approve'>
            ✅ <b>Low Risk</b><br>
            Approval Probability: {1 - probability:.2f}
        </div>
        """

# 🎨 CSS (THIS WILL DEFINITELY APPLY)
css = """
body {
    background: linear-gradient(135deg, #f8fbff, #eef3f9);
}

.container {
    max-width: 780px;
    margin: auto;
    background: white;
    padding: 28px;
    border-radius: 16px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    font-family: 'Segoe UI', sans-serif;
}

.approve {
    background: #eafaf1;
    color: #1e8449;
    padding: 16px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
}

.reject {
    background: #fdecea;
    color: #922b21;
    padding: 16px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
}

.warn {
    background: #fff3cd;
    color: #7d6608;
    padding: 14px;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1 style='text-align:center;'>Credit Risk Prediction System</h1>")
    gr.Markdown("<p style='text-align:center;'>Soft UI with validation & risk-based decisioning</p>")

    with gr.Column(elem_classes="container"):
        no_of_dependents = gr.Number(label="No of Dependents")
        education = gr.Dropdown(["Graduate", "Not Graduate"], label="Education")
        self_employed = gr.Dropdown(["Yes", "No"], label="Self Employed")
        income_annum = gr.Number(label="Annual Income")
        loan_amount = gr.Number(label="Loan Amount")
        loan_term = gr.Number(label="Loan Term (Months)")
        cibil_score = gr.Number(label="CIBIL Score")
        residential_assets_value = gr.Number(label="Residential Assets Value")
        commercial_assets_value = gr.Number(label="Commercial Assets Value")
        luxury_assets_value = gr.Number(label="Luxury Assets Value")
        bank_asset_value = gr.Number(label="Bank Asset Value")

        submit = gr.Button("Check Loan Risk")
        output = gr.HTML()

        submit.click(
            predict_loan_status,
            inputs=[
                no_of_dependents, education, self_employed,
                income_annum, loan_amount, loan_term,
                cibil_score, residential_assets_value,
                commercial_assets_value, luxury_assets_value,
                bank_asset_value
            ],
            outputs=output
        )

demo.launch()
